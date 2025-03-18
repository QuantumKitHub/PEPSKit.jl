"""
    struct PEPSOptimize{G}
    PEPSOptimize(; kwargs...)

Algorithm struct for PEPS ground-state optimization using AD. See [`fixedpoint`](@ref) for details.

## Keyword arguments

* `boundary_alg::Union{NamedTuple,<:CTMRGAlgorithm}` : Supply boundary algorithm parameters using either a `NamedTuple` of keyword arguments or a `CTMRGAlgorithm` directly. See [`leading_boundary`](@ref) for a description of all possible keyword arguments.
* `gradient_alg::Union{NamedTuple,Nothing,<:GradMode}` : Supply gradient algorithm parameters using either a `NamedTuple` of keyword arguments, `nothing`, or a `GradMode` directly. See [`fixedpoint`](@ref) for a description of all possible keyword arguments.
* `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}` : Supply optimizer algorithm parameters using either a `NamedTuple` of keyword arguments, or a `OptimKit.OptimizationAlgorithm` directly. See [`fixedpoint`](@ref) for a description of all possible keyword arguments.
* `reuse_env::Bool=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing` : Accepts `nothing` or a `SymmetrizationStyle`, in which case the PEPS and PEPS gradient are symmetrized after each optimization iteration.
"""
struct PEPSOptimize{G}
    boundary_alg::CTMRGAlgorithm
    gradient_alg::G
    optimizer_alg::OptimKit.OptimizationAlgorithm
    reuse_env::Bool
    symmetrization::Union{Nothing,SymmetrizationStyle}

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
        boundary_alg::CTMRGAlgorithm,
        gradient_alg::G,
        optimizer_alg,
        reuse_env,
        symmetrization,
    ) where {G}
        if gradient_alg isa GradMode
            if boundary_alg isa SequentialCTMRG && iterscheme(gradient_alg) === :fixed
                msg = ":fixed was converted to :diffgauge since SequentialCTMRG does not \
                      support :fixed differentiation mode due to sequential application of \
                      SVDs; select SimultaneousCTMRG instead to use :fixed mode"
                throw(ArgumentError(msg))
            end
        end
        return new{G}(boundary_alg, gradient_alg, optimizer_alg, reuse_env, symmetrization)
    end
end

function PEPSOptimize(;
    boundary_alg=(;),
    gradient_alg=(;),
    optimizer_alg=(;),
    reuse_env=Defaults.reuse_env,
    symmetrization=nothing,
)
    boundary_algorithm = _alg_or_nt(CTMRGAlgorithm, boundary_alg)
    gradient_algorithm = _alg_or_nt(GradMode, gradient_alg)
    optimizer_algorithm = _alg_or_nt(OptimKit.OptimizationAlgorithm, optimizer_alg)

    return PEPSOptimize(
        boundary_algorithm,
        gradient_algorithm,
        optimizer_algorithm,
        reuse_env,
        symmetrization,
    )
end

const OPTIMIZATION_SYMBOLS = IdDict{Symbol,Type{<:OptimKit.OptimizationAlgorithm}}(
    :gradientdescent => GradientDescent,
    :conjugategradient => ConjugateGradient,
    :lbfgs => LBFGS,
)

# Should be OptimizationAlgorithm but piracy
function _alg_or_nt(::Type{<:OptimKit.OptimizationAlgorithm}, alg::NamedTuple)
    return _OptimizationAlgorithm(; alg...)
end

function _OptimizationAlgorithm(;
    alg=Defaults.optimizer_alg,
    tol=Defaults.optimizer_tol,
    maxiter=Defaults.optimizer_maxiter,
    verbosity=Defaults.optimizer_verbosity,
    ls_maxiter=Defaults.ls_maxiter,
    ls_maxfg=Defaults.ls_maxfg,
    lbfgs_memory=Defaults.lbfgs_memory,
    # TODO: add linesearch, ... to kwargs and defaults?
)
    # replace symbol with optimizer alg type
    haskey(OPTIMIZATION_SYMBOLS, alg) ||
        throw(ArgumentError("unknown optimizer algorithm: $alg"))
    alg_type = OPTIMIZATION_SYMBOLS[alg]

    # instantiate algorithm
    return if alg_type <: LBFGS
        alg_type(lbfgs_memory; gradtol=tol, maxiter, verbosity, ls_maxiter, ls_maxfg)
    else
        alg_type(; gradtol=tol, maxiter, verbosity, ls_maxiter, ls_maxfg)
    end
end

"""
    fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv; kwargs...)
    # expert version:
    fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv, alg::PEPSOptimize;
               finalize!=OptimKit._finalize!)
    
Find the fixed point of `operator` (i.e. the ground state) starting from `peps₀` according
to the supplied optimization parameters. The initial environment `env₀` serves as an
initial guess for the first CTMRG run. By default, a random initial environment is used.

The optimization parameters can be supplied via the keyword arguments or directly as a
`PEPSOptimize` struct. The following keyword arguments are supported:

## Keyword arguments

### General settings

* `tol::Real=$(Defaults.optimizer_tol)` : Overall tolerance for gradient norm convergence of the optimizer. Sets related tolerance such as the boundary and boundary-gradient tolerances to sensible defaults unless they are explictly specified.
* `verbosity::Int=1` : Overall output information verbosity level, should be one of the following:
    0. Suppress all output
    1. Optimizer output and warnings
    2. Additionally print boundary information
    3. All information including AD debug outputs
* `reuse_env::Bool=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing` : Accepts `nothing` or a `SymmetrizationStyle`, in which case the PEPS and PEPS gradient are symmetrized after each optimization iteration.
* `(finalize!)=OptimKit._finalize!` : Inserts a `finalize!` function call after each optimization step by utilizing the `finalize!` kwarg of `OptimKit.optimize`. The function maps `(peps, env), f, g = finalize!((peps, env), f, g, numiter)`.

### Boundary algorithm

Supply boundary algorithm parameters via `boundary_alg::Union{NamedTuple,<:CTMRGAlgorithm}`
using either a `NamedTuple` of keyword arguments or a `CTMRGAlgorithm` directly.
See [`leading_boundary`](@ref) for a description of all possible keyword arguments.
By default, a CTMRG tolerance of `tol=1e-4tol` and is used.

### Gradient algorithm

Supply gradient algorithm parameters via `gradient_alg::Union{NamedTuple,Nothing,<:GradMode}`
using either a `NamedTuple` of keyword arguments, `nothing`, or a `GradMode` struct directly.
Pass `nothing` to fully differentiate the CTMRG run, meaning that all iterations will be
taken into account, instead of differentiating the fixed point. The supported `NamedTuple`
keyword arguments are:

* `tol::Real=1e-2tol` : Convergence tolerance for the fixed-point gradient iteration.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient problem iterations.
* `alg::Symbol=:$(Defaults.gradient_alg)` : Gradient algorithm variant, can be one of the following:
    - `:geomsum` : Compute gradient directly from the geometric sum, see [`GeomSum`](@ref)
    - `:manualiter` : Iterate gradient geometric sum manually, see ['ManualIter'](@ref)
    - `:linsolver` : Solve fixed-point gradient linear problem using iterative solver, see ['LinSolver'](@ref)
    - `:eigsolver` : Determine gradient via eigenvalue formulation of its Sylvester equation, see [`EigSolver`](@ref)
* `verbosity::Int` : Gradient output verbosity, ≤0 by default to disable too verbose printing. Should only be >0 for debug purposes.
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)` : CTMRG iteration scheme determining mode of differentiation. This can be:
    - `:fixed` : the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge` : the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well

### Optimizer settings

Supply the optimizer algorithm via `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}`
using either a `NamedTuple` of keyword arguments or a `OptimKit.OptimizationAlgorithm` directly. By default,
`OptimKit.LBFGS` is used in combination with a `HagerZhangLineSearch`. The supported
keyword arguments are:

* `alg::Symbol=:$(Defaults.optimizer_alg)` : Optimizer algorithm, can be one of the following:
    - `:gradientdescent` : Gradient descent algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:conjugategradient` : Conjugate gradient algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:lbfgs` : L-BFGS algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
* `tol::Real=tol` : Gradient norm tolerance of the optimizer.
* `maxiter::Int=$(Defaults.optimizer_maxiter)` : Maximal number of optimization steps.
* `verbosity::Int=$(Defaults.optimizer_verbosity)` : Optimizer output verbosity.
* `lbfgs_memory::Int=$(Defaults.lbfgs_memory)` : Size of limited memory representation of BFGS Hessian matrix.

## Return values

The function returns the final PEPS, CTMRG environment and cost value, as well as an
information `NamedTuple` which contains the following entries:

* `last_gradient` : Last gradient of the cost function.
* `fg_evaluations` : Number of evaluations of the cost and gradient function.
* `costs` : History of cost values.
* `gradnorms` : History of gradient norms.
* `truncation_errors` : History of maximal truncation errors of the boundary algorithm.
* `condition_numbers` : History of maximal condition numbers of the CTMRG environments.
* `gradnorms_unitcell` : History of gradient norms for each respective unit cell entry.
* `times` : History of optimization step execution times.
"""
function fixedpoint(
    operator,
    peps₀::InfinitePEPS,
    env₀::CTMRGEnv;
    (finalize!)=OptimKit._finalize!,
    kwargs...,
)
    alg = select_algorithm(fixedpoint, env₀; kwargs...)
    return fixedpoint(operator, peps₀, env₀, alg; finalize!)
end
function fixedpoint(
    operator,
    peps₀::InfinitePEPS,
    env₀::CTMRGEnv,
    alg::PEPSOptimize;
    (finalize!)=OptimKit._finalize!,
)
    # setup retract and finalize! for symmetrization
    if isnothing(alg.symmetrization)
        retract = peps_retract
    else
        retract, finalize! = symmetrize_retract_and_finalize!(
            alg.symmetrization, peps_retract, finalize!
        )
    end

    # :fixed mode compatibility
    if !isnothing(alg.gradient_alg) && iterscheme(alg.gradient_alg) == :fixed
        if scalartype(env₀) <: Real # incompatible with real environments
            env₀ = complex(env₀)
            @warn "the provided real environment was converted to a complex environment \
            since :fixed mode generally produces complex gauges; use :diffgauge mode \
            instead by passing gradient_alg=(; iterscheme=:diffgauge) to the fixedpoint \
            keyword arguments to work with purely real environments"
        end
    end

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps₀)), real(scalartype(env₀)))
    truncation_errors = Vector{T}()
    condition_numbers = Vector{T}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # normalize the initial guess
    peps₀ = peps_normalize(peps₀)

    # optimize operator cost function
    (peps_final, env_final), cost, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀),
        alg.optimizer_alg;
        retract,
        inner=real_inner,
        finalize!,
        (transport!)=(peps_transport!),
    ) do (peps, env)
        start_time = time_ns()
        E, gs = withgradient(peps) do ψ
            env′, info = hook_pullback(
                leading_boundary,
                env,
                ψ,
                alg.boundary_alg;
                alg_rrule=alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && update!(env, env′)
                push!(truncation_errors, info.truncation_error)
                push!(condition_numbers, info.condition_number)
            end
            return cost_function(ψ, env′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        push!(gradnorms_unitcell, norm.(g.A))
        push!(times, (time_ns() - start_time) * 1e-9)
        return E, g
    end

    info = (;
        last_gradient=∂cost,
        fg_evaluations=numfg,
        costs=convergence_history[:, 1],
        gradnorms=convergence_history[:, 2],
        truncation_errors,
        condition_numbers,
        gradnorms_unitcell,
        times,
    )
    return peps_final, env_final, cost, info
end

"""
    peps_normalize(A::InfinitePEPS)

Normalize the individual tensors in the unit cell of an `InfinitePEPS` such that they each
have unit Euclidean norm.
"""
function peps_normalize(A::InfinitePEPS)
    normalized_tensors = normalize.(unitcell(A))
    return InfinitePEPS(normalized_tensors)
end

"""
    peps_retract(x, η, α)

Performs a norm-preserving retraction of an infinite PEPS `A = x[1]` along `η` with step
size `α`, giving a new PEPS `A´`,
```math
A' ← \\cos ( α ‖η‖ / ‖A‖ ) A + \\sin ( α ‖η‖ / ‖A‖ ) ‖A‖ η / ‖η‖,
```
and corresponding directional derivative `ξ`,
```math
ξ = \\cos ( α ‖η‖ / ‖A‖ ) η - \\sin ( α ‖η‖ / ‖A‖ ) ‖η‖ A / ‖A‖,
```
such that ``⟨ A', ξ ⟩ = 0`` and ``‖A'‖ = ‖A‖``.
"""
function peps_retract(x, η, α)
    peps = x[1]
    env = deepcopy(x[2])

    retractions = norm_preserving_retract.(unitcell(peps), unitcell(η), α)
    peps´ = InfinitePEPS(map(first, retractions))
    ξ = InfinitePEPS(map(last, retractions))

    return (peps´, env), ξ
end

"""
    peps_transport!(ξ, x, η, α, x′)

Transports a direction at `A = x[1]` to a valid direction at `A´ = x´[1]` corresponding to
the norm-preserving retraction of `A` along `η` with step size `α`. In particular, starting
from a direction `η` of the form
```math
ξ = ⟨ η / ‖η‖, ξ ⟩ η / ‖η‖ + Δξ
```
where ``⟨ Δξ, A ⟩ = ⟨ Δξ, η ⟩ = 0``, it returns
```math
ξ(α) = ⟨ η / ‖η‖, ξ ⟩ ( \\cos ( α ‖η‖ / ‖A‖ ) η / ‖η‖ - \\sin( α ‖η‖ / ‖A‖ ) A / ‖A‖ ) + Δξ
```
such that ``‖ξ(α)‖ = ‖ξ‖, ⟨ A', ξ(α) ⟩ = 0``.
"""
function peps_transport!(ξ, x, η, α, x´)
    peps = x[1]
    peps´ = x´[1]

    norm_preserving_transport!.(
        unitcell(ξ), unitcell(peps), unitcell(η), α, unitcell(peps´)
    )

    return ξ
end

# Take real valued part of dot product
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

"""
    symmetrize_retract_and_finalize!(symm::SymmetrizationStyle)

Return the `retract` and `finalize!` function for symmetrizing the `peps` and `grad` tensors.
"""
function symmetrize_retract_and_finalize!(
    symm::SymmetrizationStyle, retract=peps_retract, (finalize!)=OptimKit._finalize!
)
    function symmetrize_then_finalize!((peps, env), E, grad, numiter)
        # symmetrize the gradient
        grad_symm = symmetrize!(grad, symm)
        # then finalize
        return finalize!((peps, env), E, grad_symm, numiter)
    end
    function retract_then_symmetrize((peps, env), η, α)
        # retract
        (peps´, env´), ξ = retract((peps, env), η, α)
        # symmetrize retracted point and directional derivative
        peps´_symm = symmetrize!(peps´, symm)
        ξ_symm = symmetrize!(ξ, symm)
        return (peps´_symm, env′), ξ_symm
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end

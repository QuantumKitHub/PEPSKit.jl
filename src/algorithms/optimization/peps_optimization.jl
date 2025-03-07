"""
    struct PEPSOptimize{G}

Algorithm struct for PEPS ground-state optimization using AD. See [`fixedpoint`](@ref) for details.

## Keyword arguments

* `boundary_alg::Union{NamedTuple,<:CTMRGAlgorithm}`: Supply boundary algorithm parameters using either a `NamedTuple` of keyword arguments or a `CTMRGAlgorithm` directly. See [`leading_boundary`](@ref) for a description of all possible keyword arguments.
* `gradient_alg::Union{NamedTuple,Nothing,<:GradMode}`: Supply gradient algorithm parameters using either a `NamedTuple` of keyword arguments, `nothing`, or a `GradMode` directly. See [`fixedpoint`](@ref) for a description of all possible keyword arguments.
* `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}`: Supply optimizer algorithm parameters using either a `NamedTuple` of keyword arguments, or a `OptimKit.OptimizationAlgorithm` directly. See [`fixedpoint`](@ref) for a description of all possible keyword arguments.
* `reuse_env::Bool=$(Defaults.reuse_env)`: If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing`: Accepts `nothing` or a `SymmetrizationStyle`, in which case the PEPS and PEPS gradient are symmetrized after each optimization iteration.
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
PEPSOptimize(; kwargs...) = select_algorithm(PEPSOptimize; kwargs...)

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

* `tol::Real=$(Defaults.optimizer_tol)`: Overall tolerance for gradient norm convergence of the optimizer. Sets related tolerance such as the boundary and boundary-gradient tolerances to sensible defaults unless they are explictly specified.
* `verbosity::Int=1`: Overall output information verbosity level, should be one of the following:
    0. Suppress all output
    1. Optimizer output and warnings
    2. Additionally print boundary information
    3. All information including AD debug outputs
* `reuse_env::Bool=$(Defaults.reuse_env)`: If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing`: Accepts `nothing` or a `SymmetrizationStyle`, in which case the PEPS and PEPS gradient are symmetrized after each optimization iteration.
* `(finalize!)=OptimKit._finalize!`: Inserts a `finalize!` function call after each optimization step by utilizing the `finalize!` kwarg of `OptimKit.optimize`. The function maps `(peps, env), f, g = finalize!((peps, env), f, g, numiter)`.

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

* `tol::Real=1e-2tol`: Convergence tolerance for the fixed-point gradient iteration.
* `maxiter::Int=$(Defaults.gradient_maxiter)`: Maximal number of gradient problem iterations.
* `alg::Symbol=:$(Defaults.gradient_alg)`: Gradient algorithm variant, can be one of the following:
    - `:geomsum`: Compute gradient directly from the geometric sum, see [`GeomSum`](@ref)
    - `:manualiter`: Iterate gradient geometric sum manually, see ['ManualIter'](@ref)
    - `:linsolver`: Solve fixed-point gradient linear problem using iterative solver, see ['LinSolver'](@ref)
    - `:eigsolver`: Determine gradient via eigenvalue formulation of its Sylvester equation, see [`EigSolver`](@ref)
* `verbosity::Int`: Gradient output verbosity, ≤0 by default to disable too verbose printing. Should only be >0 for debug purposes.
* `iterscheme::Symbol=:$(Defaults.gradient_iterscheme)`: CTMRG iteration scheme determining mode of differentiation. This can be:
    - `:fixed`: the differentiated CTMRG iteration uses a pre-computed SVD with a fixed set of gauges
    - `:diffgauge`: the differentiated iteration consists of a CTMRG iteration and a subsequent gauge-fixing step such that the gauge-fixing procedure is differentiated as well

### Optimizer settings

Supply the optimizer algorithm via `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}`
using either a `NamedTuple` of keyword arguments or a `OptimKit.OptimizationAlgorithm` directly. By default,
`OptimKit.LBFGS` is used in combination with a `HagerZhangLineSearch`. The supported
keyword arguments are:

* `alg::Symbol=:$(Defaults.optimizer_alg)`: Optimizer algorithm, can be one of the following:
    - `:gradientdescent`: Gradient descent algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:conjugategradient`: Conjugate gradient algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:lbfgs`: L-BFGS algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
* `tol::Real=tol`: Gradient norm tolerance of the optimizer.
* `maxiter::Int=$(Defaults.optimizer_maxiter)`: Maximal number of optimization steps.
* `verbosity::Int=$(Defaults.optimizer_verbosity)`: Optimizer output verbosity.
* `lbfgs_memory::Int=$(Defaults.lbfgs_memory)`: Size of limited memory representation of BFGS Hessian matrix.

## Return values

The function returns the final PEPS, CTMRG environment and cost value, as well as an
information `NamedTuple` which contains the following entries:

* `last_gradient`: Last gradient of the cost function.
* `fg_evaluations`: Number of evaluations of the cost and gradient function.
* `costs`: History of cost values.
* `gradnorms`: History of gradient norms.
* `truncation_errors`: History of maximal truncation errors of the boundary algorithm.
* `condition_numbers`: History of maximal condition numbers of the CTMRG environments.
* `gradnorms_unitcell`: History of gradient norms for each respective unit cell entry.
* `times`: History of optimization step execution times.
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
        retract, symm_finalize! = symmetrize_retract_and_finalize!(alg.symmetrization)
        fin! = finalize!  # Previous finalize!
        finalize! = (x, f, g, numiter) -> fin!(symm_finalize!(x, f, g, numiter)..., numiter)
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
        if isnothing(alg.boundary_alg.projector_alg.svd_alg.rrule_alg) # incompatible with TensorKit SVD rrule
            G = Base.typename(typeof(alg.gradient_alg)).wrapper # simple type without iterscheme parameter
            gradient_alg = G{:diffgauge}(
                (getproperty(alg.gradient_alg, f) for f in fieldnames(G))...
            )
            @reset alg.gradient_alg = gradient_alg
            @warn ":fixed was converted to :diffgauge since :fixed mode and \
            rrule_alg=nothing are incompatible - nothing uses the TensorKit \
            reverse-rule requiring access to the untruncated SVD which FixedSVD does not \
            have; select GMRES, BiCGStab or Arnoldi instead to use :fixed mode"
        end
    end

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps₀)), real(scalartype(env₀)))
    truncation_errors = Vector{T}()
    condition_numbers = Vector{T}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # optimize operator cost function
    (peps_final, env_final), cost, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀), alg.optimizer_alg; retract, inner=real_inner, finalize!
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

# Update PEPS unit cell in non-mutating way
# Note: Both x and η are InfinitePEPS during optimization
function peps_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env = deepcopy(x[2])
    return (peps, env), η
end

# Take real valued part of dot product
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

"""
    symmetrize_retract_and_finalize!(symm::SymmetrizationStyle)

Return the `retract` and `finalize!` function for symmetrizing the `peps` and `grad` tensors.
"""
function symmetrize_retract_and_finalize!(symm::SymmetrizationStyle)
    finf = function symmetrize_finalize!((peps, env), E, grad, _)
        grad_symm = symmetrize!(grad, symm)
        return (peps, env), E, grad_symm
    end
    retf = function symmetrize_retract((peps, env), η, α)
        peps_symm = deepcopy(peps)
        peps_symm.A .+= η.A .* α
        env′ = deepcopy(env)
        symmetrize!(peps_symm, symm)
        return (peps_symm, env′), η
    end
    return retf, finf
end

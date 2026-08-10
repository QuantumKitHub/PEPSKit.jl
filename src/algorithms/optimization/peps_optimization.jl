"""
$(TYPEDEF)

Algorithm struct for PEPS ground-state optimization using AD. See [`fixedpoint`](@ref) for details.

## Fields

$(TYPEDFIELDS)

## Constructors

    PEPSOptimize(; kwargs...)

Construct a PEPS optimization algorithm struct based on keyword arguments.
For a full description, see [`fixedpoint`](@ref). The supported keywords are:

* `boundary_alg::Union{NamedTuple,<:CTMRGAlgorithm,...}`
* `gradient_alg::Union{NamedTuple,Nothing,<:GradientAlgorithm}`
* `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}`
* `reuse_env::Bool=$(Defaults.reuse_env)`
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing`
"""
struct PEPSOptimize{B, G}
    boundary_alg::B
    gradient_alg::G
    optimizer_alg::OptimKit.OptimizationAlgorithm
    reuse_env::Bool
    symmetrization::Union{Nothing, SymmetrizationStyle}

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
            boundary_alg::B, gradient_alg::G, optimizer_alg,
            reuse_env, symmetrization,
        ) where {B, G}
        _check_algorithm_combination(
            parent_alg(boundary_alg), parent_alg(gradient_alg), symmetrization
        )
        return new{B, G}(boundary_alg, gradient_alg, optimizer_alg, reuse_env, symmetrization)
    end
end

function PEPSOptimize(;
        boundary_alg = (;), gradient_alg = (;), optimizer_alg = (;),
        reuse_env = Defaults.reuse_env, symmetrization = nothing,
    )
    boundary_algorithm = _alg_or_nt(CTMRGAlgorithm, boundary_alg)
    gradient_algorithm = _alg_or_nt(GradientAlgorithm, gradient_alg)
    optimizer_algorithm = _alg_or_nt(OptimKit.OptimizationAlgorithm, optimizer_alg)

    return PEPSOptimize(
        boundary_algorithm, gradient_algorithm, optimizer_algorithm,
        reuse_env, symmetrization,
    )
end

const OPTIMIZATION_SYMBOLS = IdDict{Symbol, Type{<:OptimKit.OptimizationAlgorithm}}(
    :GradientDescent => GradientDescent,
    :ConjugateGradient => ConjugateGradient,
    :LBFGS => LBFGS,
)

# Should be OptimizationAlgorithm but piracy
function _alg_or_nt(::Type{<:OptimKit.OptimizationAlgorithm}, alg::NamedTuple)
    return _OptimizationAlgorithm(; alg...)
end

function _OptimizationAlgorithm(;
        alg = Defaults.optimizer_alg,
        tol = Defaults.optimizer_tol,
        maxiter = Defaults.optimizer_maxiter,
        verbosity = Defaults.optimizer_verbosity,
        ls_maxiter = Defaults.ls_maxiter,
        ls_maxfg = Defaults.ls_maxfg,
        lbfgs_memory = Defaults.lbfgs_memory,
        # TODO: add linesearch, ... to kwargs and defaults?
    )
    # replace symbol with optimizer alg type
    haskey(OPTIMIZATION_SYMBOLS, alg) ||
        throw(ArgumentError("unknown optimizer algorithm: $alg"))
    alg_type = OPTIMIZATION_SYMBOLS[alg]

    # instantiate algorithm
    return if alg_type <: LBFGS
        alg_type(lbfgs_memory; gradtol = tol, maxiter, verbosity, ls_maxiter, ls_maxfg)
    else
        alg_type(; gradtol = tol, maxiter, verbosity, ls_maxiter, ls_maxfg)
    end
end

"""
    fixedpoint(operator, peps₀::InfinitePEPS, env₀; kwargs...) -> peps_final, env_final, cost_final, info
    # expert version:
    fixedpoint(operator, peps₀::InfinitePEPS, env₀, alg::PEPSOptimize; finalize!=OptimKit._finalize!)
    
Find the fixed point of `operator` (i.e. the ground state) starting from `peps₀` according
to the supplied optimization parameters. The initial environment `env₀` serves as an
initial guess for the first boundary contraction run. By default, a random initial environment is used.

The optimization parameters can be supplied via the keyword arguments or directly as a
`PEPSOptimize` struct. The following keyword arguments are supported:

## Keyword arguments

### General settings

* `tol::Real=$(Defaults.optimizer_tol)` : Overall tolerance for gradient norm convergence of the optimizer. Sets related tolerance such as the boundary and boundary-gradient tolerances to sensible defaults unless they are explictly specified.
* `verbosity::Int=1` : Overall output information verbosity level, should be one of the following:
    0. Suppress all output
    1. Only print warnings
    2. Initialization and convergence info
    3. Iteration info
    4. Debug info including AD outputs
* `reuse_env::Bool=$(Defaults.reuse_env)` : If `true`, the current optimization step is initialized on the previous environment, otherwise a random environment is used.
* `symmetrization::Union{Nothing,SymmetrizationStyle}=nothing` : Accepts `nothing` or a `SymmetrizationStyle`, in which case the PEPS and PEPS gradient are symmetrized after each optimization iteration.
* `hasconverged=OptimKit.DefaultHasConverged(optimizer_alg.gradtol)` : Function specifying the convergence criterion with signature `bool = hasconverged(state, cost, grad, gradnorm)`, see [OptimKit.optimize](https://github.com/Jutho/OptimKit.jl/blob/master/src/OptimKit.jl) for specifics. Note that this overrides the default convergence criterion.
* `shouldstop=OptimKit.DefaultShouldStop(optimizer_alg.maxiter)` : Function specifying the stopping criterion with signature `bool = shouldstop(state, cost, grad, numfg, iter, timespent)`, see [OptimKit.optimize](https://github.com/Jutho/OptimKit.jl/blob/master/src/OptimKit.jl) for specifics. Note that this overrides the default stopping criterion.
* `(finalize!)=OptimKit._finalize!` : Inserts a `finalize!` function call after each optimization step by utilizing the `finalize!` kwarg of `OptimKit.optimize`. The function maps `(state, env), f, g = finalize!((state, env), cost, grad, numiter)`.

### Boundary algorithm

Supply boundary algorithm parameters via `boundary_alg`
using either a `NamedTuple` of keyword arguments or a boundary algorithm instance directly.
See [`leading_boundary`](@ref) for a description of all possible keyword arguments.
By default, a CTMRG tolerance of `tol=1e-4tol` and is used.

### Gradient algorithm

Supply gradient algorithm parameters via `gradient_alg::Union{NamedTuple,Nothing,<:GradientAlgorithm}`
using either a `NamedTuple` of keyword arguments, `nothing`, or a `GradientAlgorithm` struct directly.
Pass `nothing` to fully differentiate the CTMRG run, meaning that all iterations will be
taken into account, instead of differentiating the fixed point. The supported `NamedTuple`
keyword arguments are:

* `tol::Real=1e-2tol` : Convergence tolerance for the fixed-point gradient iteration.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient problem iterations.
* `verbosity::Int` : Gradient output verbosity, ≤0 by default to disable too verbose printing. Should only be >0 for debug purposes.
* `alg::Symbol=:$(Defaults.gradient_alg)` : Implicit gradient algorithm variant, can be one of the following:
    - `:FixedPointGradient` : Compute the gradient via fixed-point differentiation, see [`FixedPointGradient`](@ref)
* `solver_alg::Union{Algorithm,NamedTuple}`: Solver algorithm for computing the implicit gradient; see [`FixedPointGradient`](@ref) for supported algorithms.

### Optimizer settings

Supply the optimizer algorithm via `optimizer_alg::Union{NamedTuple,<:OptimKit.OptimizationAlgorithm}`
using either a `NamedTuple` of keyword arguments or a `OptimKit.OptimizationAlgorithm` directly. By default,
`OptimKit.LBFGS` is used in combination with a `HagerZhangLineSearch`. The supported
keyword arguments are:

* `alg::Symbol=:$(Defaults.optimizer_alg)` : Optimizer algorithm, can be one of the following:
    - `:GradientDescent` : Gradient descent algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:ConjugateGradient` : Conjugate gradient algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
    - `:LBFGS` : L-BFGS algorithm, see the [OptimKit README](https://github.com/Jutho/OptimKit.jl)
* `tol::Real=tol` : Gradient norm tolerance of the optimizer.
* `maxiter::Int=$(Defaults.optimizer_maxiter)` : Maximal number of optimization steps.
* `verbosity::Int=$(Defaults.optimizer_verbosity)` : Optimizer output verbosity.
* `ls_maxiter::Int=$(Defaults.ls_maxiter)` : Maximal number of linesearch iterations.
* `ls_maxfg::Int=$(Defaults.ls_maxfg)` : Maximal number of function-gradient evaluations during linesearch.
* `lbfgs_memory::Int=$(Defaults.lbfgs_memory)` : Size of limited memory representation of BFGS Hessian matrix.

### Dynamic tolerances

The boundary and gradient algorithms each additionally accept the keyword arguments below,
which wrap the corresponding algorithm in an `MPSKit.DynamicTols.DynamicTol` that
rescales its tolerance over the course of the optimization. This allows the intermediate
problems to be solved only as accurately as the current optimization step requires, which
can significantly reduce the total runtime. The boundary tolerance is scaled relative to the
current gradient norm, while the gradient tolerance is in turn scaled relative to the
effective boundary tolerance. These settings are only available within a
variational optimization, not for standalone [`leading_boundary`](@ref) calls.

* `dynamic_tols::Bool` : Enable dynamic tolerance scaling for this algorithm. Defaults to `$(Defaults.ctmrg_dynamic_tols)` and `$(Defaults.gradient_dynamic_tols)` for the boundary and gradient algorithm respectively.
* `tol_min::Real` : Lower clamp on the dynamically scaled tolerance.
* `tol_max::Real` : Upper clamp on the dynamically scaled tolerance.
* `tol_factor::Real` : Prefactor of the dynamically scaled tolerance.

## Return values

The function returns the final PEPS, CTMRG environment and cost value, as well as an
information `NamedTuple` which contains the following entries:

* `last_gradient` : Last gradient of the cost function.
* `fg_evaluations` : Number of evaluations of the cost and gradient function.
* `costs` : History of cost values.
* `gradnorms` : History of gradient norms.
* `contraction_metrics` : History of boundary-algorithm-specific contraction information, e.g. truncation errors and condition numbers.
* `gradnorms_unitcell` : History of gradient norms for each respective unit cell entry.
* `times` : History of optimization step execution times.
"""
function fixedpoint(operator, peps₀::InfinitePEPS, env₀; kwargs...)
    extra_kwarg_keys = (:finalize!, :hasconverged, :shouldstop) # these will not be passed to `select_algorithm`, only to the 2nd `fixedpoint` call
    alg = select_algorithm(fixedpoint, env₀; filter(kw -> !(first(kw) in extra_kwarg_keys), kwargs)...)
    return fixedpoint(operator, peps₀, env₀, alg; filter(kw -> first(kw) in extra_kwarg_keys, kwargs)...)
end
function fixedpoint(
        operator, peps₀::InfinitePEPS, env₀, alg::PEPSOptimize;
        (finalize!) = OptimKit._finalize!,
        hasconverged = OptimKit.DefaultHasConverged(alg.optimizer_alg.gradtol),
        shouldstop = OptimKit.DefaultShouldStop(alg.optimizer_alg.maxiter),
    )
    # validate inputs
    check_input(fixedpoint, peps₀, env₀, alg)

    # setup retract and finalize! for symmetrization
    if isnothing(alg.symmetrization)
        retract = peps_retract
    else
        retract, finalize! = symmetrize_retract_and_finalize!(
            alg.symmetrization, peps_retract, finalize!
        )
    end

    T = promote_type(real(scalartype(peps₀)), real(scalartype(env₀)))

    # `tol_state` tracks (iter, gradnorm) of the last accepted optimization step
    # (updated only in `finalize!`), used to adjust `alg.boundary_alg`/`alg.gradient_alg`
    # via `MPSKit.updatetol` if they are wrapped in a `MPSKit.DynamicTol`. `latest_*`
    # hold the values produced by the current `fg` call, and are only recorded into
    # their respective history vectors once a step is accepted.
    tol_state = Ref((iter = 0, gradnorm = one(T)))
    latest_metrics = Ref{NamedTuple}()
    latest_gradnorms = Ref{Matrix{T}}()
    latest_time = Ref(0.0)

    # initialize info collection vectors
    contraction_metrics = Vector{NamedTuple}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()
    finalize! = track_state_and_finalize!(
        tol_state, latest_metrics, latest_gradnorms, latest_time,
        contraction_metrics, gradnorms_unitcell, times, finalize!,
    )

    # normalize the initial guess
    peps₀ = peps_normalize(peps₀)

    # optimize operator cost function
    (peps_final, env_final), cost_final, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀), alg.optimizer_alg;
        retract, inner = real_inner, (transport!) = (peps_transport!),
        hasconverged, shouldstop, finalize!,
    ) do (peps, env)
        start_time = time_ns()
        boundary_alg = updatetol(alg.boundary_alg, tol_state[].iter, tol_state[].gradnorm)
        # gradient tolerance is scaled relative to the boundary algorithm's own
        # (just-updated) effective tolerance, not directly to the gradient norm
        gradient_alg = updatetol(alg.gradient_alg, tol_state[].iter, boundary_alg.tol)
        E, gs = withgradient(peps) do ψ
            env′, info = hook_pullback(
                leading_boundary, env, ψ, boundary_alg;
                alg_rrule = gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && update!(env, env′)
                latest_metrics[] = info.contraction_metrics
            end
            return cost_function(ψ, env′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        latest_gradnorms[] = norm.(unitcell(g))
        latest_time[] = (time_ns() - start_time) * 1.0e-9
        return E, g
    end

    info = (;
        last_gradient = ∂cost,
        fg_evaluations = numfg,
        costs = convergence_history[:, 1],
        gradnorms = convergence_history[:, 2],
        contraction_metrics,
        gradnorms_unitcell,
        times,
    )
    return peps_final, env_final, cost_final, info
end

"""
    check_input(::typeof(fixedpoint), peps₀, env₀, alg::PEPSOptimize)

Check compatibility of an initial PEPS and environment with a specified PEPS optimization algorithm.
"""
function check_input(::typeof(fixedpoint), peps₀, env₀, alg::PEPSOptimize)
    if parent_alg(alg.boundary_alg) isa SimultaneousCTMRG &&
            parent_alg(alg.gradient_alg) isa FixedPointGradient &&
            scalartype(env₀) <: Real # :fixed mode gauge fixing is incompatible with real environments
        msg = "the provided real environment is incompatible with :fixed mode \
        since :fixed mode generally produces complex gauges"
        throw(ArgumentError(msg))
    end
    return nothing
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
$(SIGNATURES)

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
$(SIGNATURES)

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
    symmetrize_retract_and_finalize!(symm::SymmetrizationStyle, [retract, finalize!])

Return the `retract` and `finalize!` function for symmetrizing the `peps` and `grad` tensors.
"""
function symmetrize_retract_and_finalize!(
        symm::SymmetrizationStyle, retract = peps_retract, (finalize!) = OptimKit._finalize!
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
        return (peps´_symm, env´), ξ_symm
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end

"""
    track_state_and_finalize!(
        tol_state::Base.RefValue, latest_metrics::Base.RefValue, latest_gradnorms::Base.RefValue,
        latest_time::Base.RefValue, contraction_metrics::Vector, gradnorms_unitcell::Vector,
        times::Vector, [finalize!],
    )

Return a `finalize!` function that, after calling `finalize!` (defaulting to
`OptimKit._finalize!`):
* updates `tol_state[]` to the `(iter, gradnorm)` of the now-accepted optimization step,
  used to drive `MPSKit.updatetol` for any `alg.boundary_alg`/`alg.gradient_alg` wrapped
  in a `MPSKit.DynamicTol`
* records `latest_metrics[]`/`latest_gradnorms[]`/`latest_time[]`, i.e. the values
  produced by the `fg` call corresponding to the accepted step, into
  `contraction_metrics`/`gradnorms_unitcell`/`times`
"""
function track_state_and_finalize!(
        tol_state::Base.RefValue, latest_metrics::Base.RefValue, latest_gradnorms::Base.RefValue,
        latest_time::Base.RefValue, contraction_metrics::Vector, gradnorms_unitcell::Vector,
        times::Vector, (finalize!) = OptimKit._finalize!,
    )
    function commit_state_and_finalize!((peps, env), E, grad, numiter)
        (peps, env), E, grad = finalize!((peps, env), E, grad, numiter)
        gradnorm = sqrt(real_inner((peps, env), grad, grad))
        tol_state[] = (; iter = numiter, gradnorm)
        push!(contraction_metrics, latest_metrics[])
        push!(gradnorms_unitcell, latest_gradnorms[])
        push!(times, latest_time[])
        return (peps, env), E, grad
    end
    return commit_state_and_finalize!
end

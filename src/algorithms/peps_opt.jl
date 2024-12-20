using Manifolds: Manifolds
using Manifolds:
    AbstractManifold,
    AbstractRetractionMethod,
    Euclidean,
    default_retraction_method,
    retract
using Manopt

abstract type GradMode{F} end

iterscheme(::GradMode{F}) where {F} = F

"""
    struct GeomSum(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol,
                   verbosity=0, iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode for CTMRG using explicit evaluation of the geometric sum.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct GeomSum{F} <: GradMode{F}
    maxiter::Int
    tol::Real
    verbosity::Int
end
function GeomSum(;
    maxiter=Defaults.fpgrad_maxiter,
    tol=Defaults.fpgrad_tol,
    verbosity=0,
    iterscheme=Defaults.iterscheme,
)
    return GeomSum{iterscheme}(maxiter, tol, verbosity)
end

"""
    struct ManualIter(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol,
                      verbosity=0, iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode for CTMRG using manual iteration to solve the linear problem.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct ManualIter{F} <: GradMode{F}
    maxiter::Int
    tol::Real
    verbosity::Int
end
function ManualIter(;
    maxiter=Defaults.fpgrad_maxiter,
    tol=Defaults.fpgrad_tol,
    verbosity=0,
    iterscheme=Defaults.iterscheme,
)
    return ManualIter{iterscheme}(maxiter, tol, verbosity)
end

"""
    struct LinSolver(; solver=KrylovKit.GMRES(), iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode wrapper around `KrylovKit.LinearSolver` for solving the gradient linear
problem using iterative solvers.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct LinSolver{F} <: GradMode{F}
    solver::KrylovKit.LinearSolver
end
function LinSolver(;
    solver=KrylovKit.BiCGStab(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    iterscheme=Defaults.iterscheme,
)
    return LinSolver{iterscheme}(solver)
end

"""
    mutable struct RecordTruncationError <: RecordAction

Record the maximal truncation error of all `boundary_alg` runs of the corresponding
optimization iteration.
"""
mutable struct RecordTruncationError <: RecordAction
    recorded_values::Vector{Float64}
    RecordTruncationError() = new(Vector{Float64}())
end
function (r::RecordTruncationError)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, cache.env_info.truncation_error, i)
end

"""
    mutable struct RecordConditionNumber <: RecordAction

Record the maximal condition number of all `boundary_alg` runs of the corresponding
optimization iteration.
"""
mutable struct RecordConditionNumber <: RecordAction
    recorded_values::Vector{Float64}
    RecordConditionNumber() = new(Vector{Float64}())
end
function (r::RecordConditionNumber)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, cache.env_info.condition_number, i)
end

"""
    mutable struct RecordUnitCellGradientNorm <: RecordAction
        
Record the PEPS gradient norms unit cell entry-wise, i.e. an array 
of norms `norm.(peps.A)`.
"""
mutable struct RecordUnitCellGradientNorm <: RecordAction
    recorded_values::Vector{Matrix{Float64}}
    RecordUnitCellGradientNorm() = new(Vector{Matrix{Float64}}())
end
function (r::RecordUnitCellGradientNorm)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    cache = Manopt.get_cost_function(get_objective(p))
    grad = cache.from_vec(get_gradient(s))
    return Manopt.record_or_reset!(r, norm.(grad.A), i)
end

"""
    struct PEPSOptimize{G}

Algorithm struct for PEPS optimization using automatic differentiation.

# Fields
- `boundary_alg::CTMRGAlgorithm`: algorithm for determining the PEPS environment
- `optim_alg::Function`: Manopt optimization algorithm
- `optim_kwargs::NamedTuple`: Keyword arguments provided to the Manopt `optim_alg` call
- `gradient_alg::G`: Algorithm computing the cost function gradient in reverse-mode
- `reuse_env::Bool`: If `true` the previous environment is used to initialize the next
    `leading_boundary` call
- `symmetrization::Union{Nothing,SymmetrizationStyle}`: Symmetrize the PEPS and PEPS
    gradient after each optimization iteration (does nothing if `nothing` is provided)
"""
struct PEPSOptimize{G}
    boundary_alg::CTMRGAlgorithm
    optim_alg::Function
    optim_kwargs::NamedTuple
    gradient_alg::G
    reuse_env::Bool
    # reuse_env_tol::Float64  # TODO: add option for reuse tolerance
    symmetrization::Union{Nothing,SymmetrizationStyle}

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
        boundary_alg::CTMRGAlgorithm,
        optim_alg,
        optim_kwargs,
        gradient_alg::G,
        reuse_env,
        symmetrization,
    ) where {G}
        if gradient_alg isa GradMode
            if boundary_alg isa SequentialCTMRG && iterscheme(gradient_alg) === :fixed
                throw(ArgumentError(":sequential and :fixed are not compatible"))
            end
        end
        return new{G}(
            boundary_alg, optim_alg, optim_kwargs, gradient_alg, reuse_env, symmetrization
        )
    end
end

"""
    PEPSOptimize(;
        boundary_alg=Defaults.ctmrg_alg,
        optim_alg=Defaults.optim_alg,
        maxiter=Defaults.optim_maxiter,
        tol=Defaults.optim_tol,
        gradient_alg=Defaults.gradient_alg,
        reuse_env=Defaults.reuse_env,
        symmetrization=nothing,
        kwargs...,
    )

Convenience keyword argument constructor for `PEPSOptimize` algorithms.
Here, `maxiter` and `tol` are passed onto `StopAfterIteration` and `StopWhenGradientNormLess`
stopping criteria, respectively. Additionally, any keyword arguments can be provided which 
are then stored inside `optim_kwargs` and passed to the Manopt optimization call, such that
that all arguments of the respective `optim_alg` can be used.
"""
function PEPSOptimize(;
    boundary_alg=Defaults.ctmrg_alg,
    optim_alg=Defaults.optim_alg,
    maxiter=Defaults.optim_maxiter,
    tol=Defaults.optim_tol,
    gradient_alg=Defaults.gradient_alg,
    reuse_env=Defaults.reuse_env,
    symmetrization=nothing,
    kwargs...,
)
    stopping_criterion = StopAfterIteration(maxiter) | StopWhenGradientNormLess(tol)
    optim_kwargs = merge(Defaults.optim_kwargs, (; stopping_criterion, kwargs...))
    return PEPSOptimize(
        boundary_alg, optim_alg, optim_kwargs, gradient_alg, reuse_env, symmetrization
    )
end

"""
    mutable struct PEPSCostFunctionCache{T}

Stores objects used for computing PEPS cost functions during optimization that are
needed apart from the PEPS that is being optimized.

# Fields
- `operator::LocalOperator`: cost function operator
- `alg::PEPSOptimize`: optimization parameters
- `env::CTMRGEnv`: environment of the current PEPS
- `from_vec`: map which returns vectorized PEPS as an `InfinitePEPS`
- `peps_vec::Vector{T}`: current vectorized PEPS
- `grad_vec::Vector{T}`: current vectorized gradient
- `cost::Float64`: current cost function value
- `env_info::NamedTuple`: return info of `leading_boundary` used by `RecordAction`s
"""
mutable struct PEPSCostFunctionCache{T}
    operator::LocalOperator
    alg::PEPSOptimize
    env::CTMRGEnv
    from_vec
    peps_vec::Vector{T}
    grad_vec::Vector{T}
    cost::Float64
    env_info::NamedTuple
end

"""
    PEPSCostFunctionCache(
        operator::LocalOperator, alg::PEPSOptimize, peps_vec::Vector, from_vec, env::CTMRGEnv
    )

Initialize a `PEPSCostFunctionCache` using `peps_vec` from which the vector dimension
and scalartype are derived.
"""
function PEPSCostFunctionCache(
    operator::LocalOperator, alg::PEPSOptimize, peps_vec::Vector, from_vec, env::CTMRGEnv
)
    return PEPSCostFunctionCache(
        operator,
        alg,
        env,
        from_vec,
        similar(peps_vec),
        similar(peps_vec),
        0.0,
        (; truncation_error=0.0, condition_number=1.0),
    )
end

"""
    cost_and_grad!(cache::PEPSCostFunctionCache{T}, peps_vec::Vector{T}) where {T}

Update the cost and gradient of the `PEPSCostFunctionCache` with respect to the new point
`peps_vec`.
"""
function cost_and_grad!(cache::PEPSCostFunctionCache{T}, peps_vec::Vector{T}) where {T}
    cache.peps_vec .= peps_vec  # update point in manifold
    peps = cache.from_vec(peps_vec)  # convert back to InfinitePEPS
    env₀ =
        cache.alg.reuse_env ? cache.env : CTMRGEnv(randn, scalartype(cache.env), cache.env)

    # compute cost and gradient
    cost, grads = withgradient(peps) do ψ
        env, info = hook_pullback(
            leading_boundary,
            env₀,
            ψ,
            cache.alg.boundary_alg;
            alg_rrule=cache.alg.gradient_alg,
        )
        cost = expectation_value(ψ, cache.operator, env)
        ignore_derivatives() do
            update!(cache.env, env)  # update environment in-place
            cache.env_info = info  # update environment information (truncation error, ...)
            isapprox(imag(cost), 0; atol=sqrt(eps(real(cost)))) ||
                @warn "Expectation value is not real: $cost."
        end
        return real(cost)
    end
    grad = only(grads)  # `withgradient` returns tuple of gradients `grads`

    # symmetrize gradient
    if !isnothing(cache.alg.symmetrization)
        grad = symmetrize!(grad, cache.alg.symmetrization)
    end

    cache.cost = cost  # update cost function value
    cache.grad_vec .= to_vec(grad)[1]  # update vectorized gradient
    return cache.cost, cache.grad_vec
end

"""
    (cache::PEPSCostFunctionCache{T})(::Euclidean, peps_vec::Vector{T}) where {T}
    
Return the cost of `cache` and recompute if `peps_vec` is a new point.
"""
function (cache::PEPSCostFunctionCache{T})(::Euclidean, peps_vec::Vector{T}) where {T}
    # Note that it is necessary to implement the cost function as a functor so that the
    # `PEPSCostFunctionCache` is available through `get_objective(::AbstractManoptProblem)`
    # to the `RecordAction`s during optimization
    if !(peps_vec == cache.peps_vec) # update cache if at new point
        cost_and_grad!(cache, peps_vec)
    end
    return cache.cost
end

"""
    gradient_function(cache::PEPSCostFunctionCache{T}) where {T}

Get the gradient function of `cache` which returns the gradient vector
and recomputes it if the provided point is new.
"""
function gradient_function(cache::PEPSCostFunctionCache{T}) where {T}
    return function gradient_function(::Euclidean, peps_vec::Vector{T})
        if !(peps_vec == cache.peps_vec) # update cache if at new point
            cost_and_grad!(cache, peps_vec)
        end
        return cache.grad_vec
    end
end

"""
    SymmetrizeExponentialRetraction <: AbstractRetractionMethod
    
Exponential retraction followed by a symmetrization step.
"""
struct SymmetrizeExponentialRetraction <: AbstractRetractionMethod
    symmetrization::SymmetrizationStyle
    from_vec::Function
end

function Manifolds.retract(
    M::AbstractManifold, p, X, t::Number, sr::SymmetrizeExponentialRetraction
)
    q = retract(M, p, X, t, ExponentialRetraction())
    q_symm_peps = symmetrize!(sr.from_vec(q), sr.symmetrization)
    return to_vec(q_symm_peps)
end

"""
    fixedpoint(
        peps₀::InfinitePEPS{T},
        operator::LocalOperator,
        alg::PEPSOptimize,
        env₀::CTMRGEnv=CTMRGEnv(peps₀, field(T)^20);
    ) where {T}

Optimize a PEPS starting from `peps₀` and `env₀` by minimizing the cost function given
by expectation value of `operator`. All optimization parameters are provided through `alg`.
"""
function fixedpoint(
    peps₀::InfinitePEPS{T},
    operator::LocalOperator,
    alg::PEPSOptimize,
    env₀::CTMRGEnv=CTMRGEnv(peps₀, field(T)^20);
) where {T}
    if scalartype(env₀) <: Real
        env₀ = complex(env₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    # construct cost and grad functions
    peps₀_vec, from_vec = to_vec(peps₀)
    cache = PEPSCostFunctionCache(operator, alg, peps₀_vec, from_vec, deepcopy(env₀))
    cost = cache
    grad = gradient_function(cache)

    # optimize
    # fld = scalartype(peps₀) <: Real ? Manifolds.ℝ : Manifolds.ℂ  # Manopt can't optimize over ℂ?
    M = Euclidean(length(peps₀_vec))
    retraction_method = if isnothing(alg.symmetrization)
        default_retraction_method(M)
    else
        SymmetrizeExponentialRetraction(alg.symmetrization, from_vec)
    end
    result = alg.optim_alg(
        M, cost, grad, peps₀_vec; alg.optim_kwargs..., retraction_method, return_state=true
    )

    # extract final result
    peps_final = from_vec(get_solver_result(result))
    return peps_final, cost.env, cache.cost, result
end

#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

function _rrule(
    gradmode::GradMode{:diffgauge},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::CTMRGAlgorithm,
)
    envs, info = leading_boundary(envinit, state, alg)

    function leading_boundary_diffgauge_pullback((Δenvs′, Δinfo))
        Δenvs = unthunk(Δenvs′)

        # find partial gradients of gauge_fixed single CTMRG iteration
        f(A, x) = gauge_fix(x, ctmrg_iteration(A, x, alg)[1])[1]
        _, env_vjp = rrule_via_ad(config, f, state, envs)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[3]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂envs, NoTangent()
    end

    return (envs, info), leading_boundary_diffgauge_pullback
end

# Here f is differentiated from an pre-computed SVD with fixed U, S and V
function _rrule(
    gradmode::GradMode{:fixed},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::SimultaneousCTMRG,
)
    @assert !isnothing(alg.projector_alg.svd_alg.rrule_alg)
    envs, = leading_boundary(envinit, state, alg)
    envs_conv, info = ctmrg_iteration(state, envs, alg)
    envs_fixed, signs = gauge_fix(envs, envs_conv)

    # Fix SVD
    Ufix, Vfix = fix_relative_phases(info.U, info.V, signs)
    svd_alg_fixed = SVDAdjoint(;
        fwd_alg=FixedSVD(Ufix, info.S, Vfix), rrule_alg=alg.projector_alg.svd_alg.rrule_alg
    )
    alg_fixed = @set alg.projector_alg.svd_alg = svd_alg_fixed
    alg_fixed = @set alg_fixed.projector_alg.trscheme = notrunc()

    function leading_boundary_fixed_pullback((Δenvs′, Δinfo))
        Δenvs = unthunk(Δenvs′)

        f(A, x) = fix_global_phases(x, ctmrg_iteration(A, x, alg_fixed)[1])
        _, env_vjp = rrule_via_ad(config, f, state, envs_fixed)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[3]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂envs, NoTangent()
    end

    return (envs_fixed, info), leading_boundary_fixed_pullback
end

@doc """
    fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y0, alg)

Compute the gradient of the cost function for CTMRG by solving the following equation:

dx = ∑ₙ (∂f∂x)ⁿ ∂f∂A dA = (1 - ∂f∂x)⁻¹ ∂f∂A dA

where `∂F∂x` is the gradient of the cost function with respect to the PEPS tensors, `∂f∂x`
is the partial gradient of the CTMRG iteration with respect to the environment tensors,
`∂f∂A` is the partial gradient of the CTMRG iteration with respect to the PEPS tensors, and
`y0` is the initial guess for the fixed-point iteration. The function returns the gradient
`dx` of the fixed-point iteration.
"""
fpgrad

# TODO: can we construct an implementation that does not need to evaluate the vjp
# twice if both ∂f∂A and ∂f∂x are needed?
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, _, alg::GeomSum)
    g = ∂F∂x
    dx = ∂f∂A(g) # n = 0 term: ∂F∂x ∂f∂A
    ϵ = 2 * alg.tol
    for i in 1:(alg.maxiter)
        g = ∂f∂x(g)
        Σₙ = ∂f∂A(g)
        dx += Σₙ
        ϵnew = norm(Σₙ)  # TODO: normalize this error?
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 &&
            @printf("Gradient iter: %3d   ‖Σₙ‖: %.2e   Δ‖Σₙ‖: %.2e\n", i, ϵnew, Δϵ)
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Σₙ‖ = $ϵ"
        end
    end
    return dx
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::ManualIter)
    y = deepcopy(y₀)  # Do not mutate y₀
    dx = ∂f∂A(y)
    ϵ = 1.0
    for i in 1:(alg.maxiter)
        y′ = ∂F∂x + ∂f∂x(y)

        dxnew = ∂f∂A(y′)
        ϵnew = norm(dxnew - dx)
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 && @printf(
            "Gradient iter: %3d   ‖Cᵢ₊₁-Cᵢ‖/N: %.2e   Δ‖Cᵢ₊₁-Cᵢ‖/N: %.2e\n", i, ϵnew, Δϵ
        )
        y = y′
        dx = dxnew
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Cᵢ₊₁-Cᵢ‖ = $ϵ"
        end
    end
    return dx
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::LinSolver)
    y, info = linsolve(∂f∂x, ∂F∂x, y₀, alg.solver, 1, -1)
    if alg.solver.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

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
    truncation_error::Float64
    condition_number::Float64
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
        operator, alg, env, from_vec, similar(peps_vec), similar(peps_vec), 0.0, 0.0, 1.0
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
        env, truncation_error, condition_number = hook_pullback(
            leading_boundary,
            env₀,
            ψ,
            cache.alg.boundary_alg;
            alg_rrule=cache.alg.gradient_alg,
        )
        cost = expectation_value(ψ, cache.operator, env)
        ignore_derivatives() do
            update!(cache.env, env)  # update environment in-place
            cache.truncation_error = truncation_error  # update environment information
            cache.condition_number = condition_number
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

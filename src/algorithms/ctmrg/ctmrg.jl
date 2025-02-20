"""
    CTMRGAlgorithm

Abstract super type for the corner transfer matrix renormalization group (CTMRG) algorithm
for contracting infinite PEPS.
"""
abstract type CTMRGAlgorithm end

"""
    ctmrg_iteration(state, env, alg::CTMRGAlgorithm) -> env′, info

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(state, env, alg::CTMRGAlgorithm) end

"""
    MPSKit.leading_boundary([envinit], state, alg::CTMRGAlgorithm)

Contract `state` using CTMRG and return the CTM environment. Per default, a random
initial environment is used.

Each CTMRG run is converged up to `alg.tol` where the singular value convergence
of the corners and edges is checked. The maximal and minimal number of CTMRG
iterations is set with `alg.maxiter` and `alg.miniter`.

Different levels of output information are printed depending on `alg.verbosity`, where `0`
suppresses all output, `1` only prints warnings, `2` gives information at the start and
end, and `3` prints information every iteration.
"""
function MPSKit.leading_boundary(
    envinit, network::InfiniteSquareNetwork, alg::CTMRGAlgorithm
)
    CS = map(x -> tsvd(x)[2], envinit.corners)
    TS = map(x -> tsvd(x)[2], envinit.edges)

    η = one(real(scalartype(network)))
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    return LoggingExtras.withlevel(; alg.verbosity) do
        ctmrg_loginit!(log, η, network, envinit)
        local info
        for iter in 1:(alg.maxiter)
            env, info = ctmrg_iteration(network, env, alg)  # Grow and renormalize in all 4 directions
            η, CS, TS = calc_convergence(env, CS, TS)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, network, env)
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, network, env)
            else
                ctmrg_logiter!(log, iter, η, network, env)
            end
        end
        return env, info
    end
end
function MPSKit.leading_boundary(envinit, state, alg::CTMRGAlgorithm)
    return MPSKit.leading_boundary(envinit, InfiniteSquareNetwork(state), alg)
end

# custom CTMRG logging
function ctmrg_loginit!(log, η, network, env)
    @infov 2 loginit!(log, η, network_value(network, env))
end
function ctmrg_logiter!(log, iter, η, network, env)
    @infov 3 logiter!(log, iter, η, network_value(network, env))
end
function ctmrg_logfinish!(log, iter, η, network, env)
    @infov 2 logfinish!(log, iter, η, network_value(network, env))
end
function ctmrg_logcancel!(log, iter, η, network, env)
    @warnv 1 logcancel!(log, iter, η, network_value(network, env))
end

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

#=
In order to compute an error measure, we compare the singular values of the current iteration with the previous one.
However, when the virtual spaces change, this comparison is not directly possible.
Instead, we project both tensors into the smaller space and then compare the difference.

TODO: we might want to consider embedding the smaller tensor into the larger space and then compute the difference
=#
function _singular_value_distance((S₁, S₂))
    V₁ = space(S₁, 1)
    V₂ = space(S₂, 1)
    if V₁ == V₂
        return norm(S₁ - S₂)
    else
        V = infimum(V₁, V₂)
        e1 = isometry(V₁, V)
        e2 = isometry(V₂, V)
        return norm(e1' * S₁ * e1 - e2' * S₂ * e2)
    end
end

"""
    calc_convergence(env, CS_old, TS_old)
    calc_convergence(env_new::CTMRGEnv, env_old::CTMRGEnv)

Given a new environment `env`, compute the maximal singular value distance.
This determined either from the previous corner and edge singular values
`CS_old` and `TS_old`, or alternatively, directly from the old environment.
"""
function calc_convergence(env, CS_old, TS_old)
    CS_new = map(x -> tsvd(x)[2], env.corners)
    ΔCS = maximum(_singular_value_distance, zip(CS_old, CS_new))

    TS_new = map(x -> tsvd(x)[2], env.edges)
    ΔTS = maximum(_singular_value_distance, zip(TS_old, TS_new))

    @debug "maxᵢ|Cⁿ⁺¹ - Cⁿ|ᵢ = $ΔCS   maxᵢ|Tⁿ⁺¹ - Tⁿ|ᵢ = $ΔTS"

    return max(ΔCS, ΔTS), CS_new, TS_new
end
function calc_convergence(env_new::CTMRGEnv, env_old::CTMRGEnv)
    CS_old = map(x -> tsvd(x)[2], env_old.corners)
    TS_old = map(x -> tsvd(x)[2], env_old.edges)
    return calc_convergence(env_new, CS_old, TS_old)
end
@non_differentiable calc_convergence(args...)

abstract type CTMRGAlgorithmTriangular end

function leading_boundary(env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return leading_boundary(env₀, network, alg)
end
function leading_boundary(
        env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork, alg::CTMRGAlgorithmTriangular
    )
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))
    return LoggingExtras.withlevel(; alg.verbosity) do
        env = deepcopy(env₀)
        S_old = DiagonalTensorMap.(id.(domain.(env₀.C)))
        η = one(real(scalartype(network)))
        ctmrg_loginit!(log, η, network, env₀)
        local info
        for iter in 1:(alg.maxiter)
            env, S = ctmrg_iteration(network, env, alg)  # Grow and renormalize in all 4 directions
            η = calc_convergence(S, S_old)
            S_old = copy(S)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, network, env)
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, network, env)
            else
                ctmrg_logiter!(log, iter, η, network, env)
            end
            info = (; convergence_metric = η)
        end
        return env, info
    end
end
function leading_boundary(env₀::CTMRGEnvTriangular, state, args...; kwargs...)
    return leading_boundary(env₀, InfiniteTriangularNetwork(state), args...; kwargs...)
end

function calc_convergence(S_new::Array{T, 3}, S_old::Array{T, 3}) where {E, S, T <: DiagonalTensorMap{E, S}}
    ε = Inf
    function dist(S1, S2)
        if space(S1) == space(S2)
            return norm(S1^4 - S2^4)
        else
            return Inf
        end
    end
    return maximum(dist.(S_new, S_old))
end

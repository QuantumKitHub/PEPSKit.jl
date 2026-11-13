abstract type CTMRGAlgorithmTriangular end

function CTMRGAlgorithmTriangular(; alg = :SimultaneousCTMRGTriangular, kwargs...)
    alg == :SimultaneousCTMRGTriangular || throw(ArgumentError("unknown triangular CTMRG algorithm: $alg"))
    return SimultaneousCTMRGTriangular(; kwargs...)
end

"""
    leading_boundary(env₀::CTMRGEnvTriangular, network; alg=:SimultaneousCTMRGTriangular, kwargs...)
    leading_boundary(env₀::CTMRGEnvTriangular, network, alg::CTMRGAlgorithmTriangular)

Keyword arguments are forwarded to `SimultaneousCTMRGTriangular`: `tol`, `maxiter`,
`miniter`, `verbosity`, `decomposition_alg`, `conditioning`, `projector_alg`, and `trunctype`.

Return the contracted environment and convergence information. `info.converged` and
`info.convergence_error` describe convergence; `info.convergence_metric` is retained as
an alias for the latter. Convergence is the maximum singular-value distance of all
six corners and twelve edge halves at every site, following PEPSKit's native tensor
bipartitions and sector-aware distance. Spectral convergence does not imply an
elementwise fixed point. `info.contraction_metrics` contains `corner_truncation_error`
and `edge_truncation_error`, maximized over all sites and directions of the final
iteration, and their maximum `truncation_error`.

Errors are discarded norms divided by the retained SVD spectrum norm, following
PEPSKit's corner-projector convention. They include singular values suppressed by
the inverse-square-root cutoff, but exclude preliminary seed and conditioning SVDs.
These are local truncation diagnostics, not errors on the final observables.
"""
function leading_boundary(env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return leading_boundary(env₀, network, alg)
end
function leading_boundary(
        env₀::CTMRGEnvTriangular, network::InfiniteTriangularNetwork, alg::CTMRGAlgorithmTriangular
    )
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))
    return with_verbosity(; alg.verbosity) do
        env = deepcopy(env₀)
        CS, TS = ignore_derivatives() do
            return convergence_spectra(env₀, alg)
        end
        η = one(real(scalartype(network)))
        ctmrg_loginit!(log, η, network, env₀)
        local info_iter
        converged = false
        for iter in 1:(alg.maxiter)
            env, info_iter = ctmrg_iteration(network, env, alg)
            η, CS, TS = calc_convergence(env, CS, TS, alg)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, network, env)
                converged = true
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, network, env)
            else
                ctmrg_logiter!(log, iter, η, network, env)
            end
        end
        info = (;
            converged, convergence_error = η, convergence_metric = η,
            contraction_metrics = info_iter.contraction_metrics,
        )
        return env, info
    end
end
function leading_boundary(env₀::CTMRGEnvTriangular, state, args...; kwargs...)
    return leading_boundary(env₀, InfiniteTriangularNetwork(state), args...; kwargs...)
end

convergence_tensors(env::CTMRGEnvTriangular, alg) = (env.C, [env.Ea; env.Eb])

function convergence_spectra(env::CTMRGEnvTriangular, alg)
    corners, edges = convergence_tensors(env, alg)
    return corner_spectra(corners, alg), edge_spectra(edges, alg)
end

"""
    struct SUGauge

Algorithm for fixing gauge of an iPEPS using trivial simple update
(with identity gates).

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SUGauge
    "Stopping criterion for the trivial SU iterations in weight difference"
    tol::Float64 = 1.0e-10
    "Minimal number of SU iterations"
    miniter::Int = 2
    "Maximal number of SU iterations"
    maxiter::Int = 100
end

function _trivial_gates(Nrow::Int, Ncol::Int)
    gates = map(Iterators.product(1:2, 1:Ncol, 1:Nrow)) do (d, c, r)
        site1 = CartesianIndex(r, c)
        site2 = (d == 1) ? CartesianIndex(r, c + 1) : CartesianIndex(r - 1, c)
        return [site1, site2] => nothing
    end
    return TrotterGates(vec(gates))
end

"""
    gauge_fix(psi::Union{InfinitePEPS, InfinitePEPO}, alg::SUGauge)

Fix the gauge of `psi` using trivial simple update.
"""
function gauge_fix(psi::InfiniteState, alg::SUGauge)
    Nr, Nc, = size(psi)
    gates = _trivial_gates(Nr, Nc)
    su_alg = SimpleUpdate(; trunc = FixedSpaceTruncation(), bipartite = _state_bipartite_check(psi))
    wts0 = SUWeight(psi)
    # use default constructor to avoid calculation of exp(-H * 0)
    evolver = TimeEvolver(su_alg, 0.0, alg.maxiter, gates, SUState(0, 0.0, psi, wts0))
    for (i, (psi′, wts, info)) in enumerate(evolver)
        ϵ = compare_weights(wts, wts0)
        if i >= alg.miniter && ϵ < alg.tol
            @info "Trivial SU conv $i: |Δλ| = $ϵ."
            return psi′, wts, ϵ
        end
        if i == alg.maxiter
            @warn "Trivial SU cancel $i: |Δλ| = $ϵ."
            return psi′, wts, ϵ
        end
        wts0 = deepcopy(wts)
    end
    return
end

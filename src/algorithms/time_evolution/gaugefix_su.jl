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

"""
A LocalOperator consisting of identity gates on all nearest neighbor bonds.
"""
function _trivial_gates(elt::Type{<:Number}, lattice::Matrix{S}) where {S <: ElementarySpace}
    terms = []
    for site1 in CartesianIndices(lattice)
        r1, c1 = mod1.(Tuple(site1), size(lattice))
        for d in (CartesianIndex(1, 0), CartesianIndex(0, 1))
            site2 = site1 + d
            r2, c2 = mod1.(Tuple(site2), size(lattice))
            V1, V2 = lattice[r1, c1], lattice[r2, c2]
            h = TensorKit.id(elt, V1 ⊗ V2)
            push!(terms, (site1, site2) => h)
        end
    end
    return LocalOperator(lattice, terms...)
end

function gauge_fix_su(peps0::InfinitePEPS, alg::SUGauge)
    gates = _trivial_gates(scalartype(peps0), physicalspace(peps0))
    su_alg = SimpleUpdate(; trunc = FixedSpaceTruncation())
    wts0 = SUWeight(peps0)
    # use default constructor to avoid calculation of exp(-H * 0)
    evolver = TimeEvolver(su_alg, 0.0, alg.maxiter, gates, SUState(0, 0.0, peps0, wts0))
    for (i, (peps0, wts, info)) in enumerate(evolver)
        ϵ = compare_weights(wts, wts0)
        if i >= alg.miniter && ϵ < alg.tol
            @info "Trivial SU conv $i: |Δλ| = $ϵ."
            return peps0, wts, ϵ
        end
        if i == alg.maxiter
            @warn "Trivial SU cancel $i: |Δλ| = $ϵ."
            return peps0, wts, ϵ
        end
        wts0 = deepcopy(wts)
    end
    return
end

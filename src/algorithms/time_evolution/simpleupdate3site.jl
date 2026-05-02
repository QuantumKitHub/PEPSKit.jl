function _fuse_physicalspaces(O::GenericMPSTensor{S, 5}) where {S <: ElementarySpace}
    V1, V2 = codomain(O, 2), codomain(O, 3)
    F = isomorphism(Int, fuse(V1, V2), V1 ⊗ V2)
    @plansor O_fused[-1 -2 -4 -5; -6] := F[-2; 2 3] * O[-1 2 3 -4 -5; -6]
    return O_fused, F
end

function _unfuse_physicalspace(
        O::GenericMPSTensor{S, 4}, Vout::ElementarySpace, Vin::ElementarySpace = Vout'
    ) where {S <: ElementarySpace}
    F = isomorphism(Int, Vout ⊗ Vin, fuse(Vout ⊗ Vin))
    @plansor O_unfused[-1 -2 -3 -4 -5; -6] := F[-2 -3; 1] * O[-1 1 -4 -5; -6]
    return O_unfused, F
end

function _get_cluster_with_weights(
    state::InfiniteState, sites::Vector{CartesianIndex{2}}, env::SUWeight
)
    Ms, open_vaxs, perms = _get_cluster(state, sites)
    _absorb_weight!(Ms, sites, open_vaxs, env; inv = false)
    Np = (state isa InfinitePEPS) ? Val(1) : Val(2)
    invperms = map(p -> _inv_mpo_perm(p, Np), perms)
    return _permute_cluster(Ms, perms), open_vaxs, invperms
end

"""
Simple update with an N-site MPO `gate` (N ≥ 2).
"""
function _su_iter!(
        state::InfiniteState, gate::Vector{T}, env::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::SimpleUpdate
    ) where {T <: AbstractTensorMap}
    Nr, Nc = size(state)
    truncs = _get_cluster_trunc(alg.trunc, sites, (Nr, Nc))
    Ms, open_vaxs, invperms = _get_cluster_with_weights(state, sites, env)
    # flip virtual arrows in `Ms` to ←
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips)
    # apply gate MPOs and truncate
    ϵ = 0.0
    local wts
    for gate_ax in 1:2
        _apply_gatempo!(Ms, gate; gate_ax)
        wts, ϵs, = _cluster_truncate!(Ms, truncs)
        ϵ = max(ϵ, maximum(ϵs))
        alg.purified && break
    end
    # restore virtual arrows in `Ms`
    _flip_virtuals!(Ms, flips)
    # update env weights
    bond_revs = map(sites, Iterators.drop(sites, 1)) do site1, site2
        _nn_bondrev(site1, site2, (Nr, Nc))
    end
    for (wt, (bond, rev), flip) in zip(wts, bond_revs, flips)
        wt_new = flip ? _fliptwist_s(wt) : wt
        wt_new = rev ? transpose(wt_new) : wt_new
        env[CartesianIndex(bond)] = normalize!(wt_new, Inf)
    end
    # update state tensors
    for (M, s, invperm, vaxs) in zip(Ms, sites, invperms, open_vaxs)
        s′ = CartesianIndex(mod1(s[1], Nr), mod1(s[2], Nc))
        # restore original axes order and remove weights on open axes of the cluster
        M′ = absorb_weight(permute(M, invperm), env, s′, vaxs; inv = true)
        state[s′] = normalize!(M′, Inf)
    end
    return ϵ
end

"""
Get the `TruncationStrategy` for each bond in the cluster
updated by the Trotter evolution MPO.
"""
function _get_cluster_trunc(
        trunc::TruncationStrategy, sites::Vector{CartesianIndex{2}},
        unitcell::NTuple{2, Int}
    )
    return map(sites, Iterators.drop(sites, 1)) do site1, site2
        (d, r, c), rev = _nn_bondrev(site1, site2, unitcell)
        t = truncation_strategy(trunc, d, r, c)
        if rev && isa(t, TruncationSpace)
            t = truncspace(flip(t.space)')
        end
        return t
    end
end

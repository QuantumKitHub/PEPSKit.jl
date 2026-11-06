function _fuse_physicalspaces(O::GenericMPSTensor{S, 5}) where {S <: ElementarySpace}
    V1, V2 = codomain(O, 2), codomain(O, 3)
    F = isomorphism(similarstoragetype(O, Int), fuse(V1, V2), V1 ⊗ V2)
    @plansor O_fused[-1 -2 -4 -5; -6] := F[-2; 2 3] * O[-1 2 3 -4 -5; -6]
    return O_fused, F
end

function _unfuse_physicalspace(
        O::GenericMPSTensor{S, 4}, Vout::ElementarySpace, Vin::ElementarySpace = Vout'
    ) where {S <: ElementarySpace}
    F = isomorphism(similarstoragetype(O, Int), Vout ⊗ Vin, fuse(Vout ⊗ Vin))
    @plansor O_unfused[-1 -2 -3 -4 -5; -6] := F[-2 -3; 1] * O[-1 1 -4 -5; -6]
    return O_unfused, F
end

"""
Obtain the cluster along `sites` with weights absorbed into its open virtual legs,
and permuted to MPS axis order.
"""
function _get_cluster_with_weights(
        state::InfiniteState, sites::Vector{CartesianIndex{2}}, env::SUWeight
    )
    Ms, open_vaxs, perms = _get_cluster(state, sites)
    Np = (state isa InfinitePEPS) ? Val(1) : Val(2)
    Ms_mps = map(Ms, sites, open_vaxs, perms) do M, site, vaxs, perm
        return permute(absorb_weight(M, env, site, vaxs), perm)
    end
    invperms = map(p -> _inv_mpo_perm(p, Np), perms)
    return Ms_mps, open_vaxs, invperms
end

"""
Factor out irrelevant virtual legs out of the cluster endpoints
for more efficient application of gate MPOs.
"""
function _reduce_mpo_endpoints!(Ms::Vector{<:GenericMPSTensor})
    N = numout(first(Ms))
    physical = ntuple(i -> i + 1, N - 3)
    Ql, R = left_orth!(permute(first(Ms), ((1, N - 1, N), (physical..., N + 1))); positive = true)
    R = permute(R, ((1, physical...), (N - 1,)))
    L, Qr = right_orth!(permute(last(Ms), ((1, physical...), (N - 1, N, N + 1))); positive = true)
    Ms[1] = insertrightunit(insertrightunit(R, Val(N - 2)), Val(N - 1))
    Ms[end] = insertrightunit(insertrightunit(L, Val(N - 2)), Val(N - 1))
    return Ql, Qr
end

"""
Restore virtual legs of the cluster endpoints after gate MPO application.
"""
function _restore_mpo_endpoints!(Ms::Vector{<:GenericMPSTensor}, Ql, Qr)
    N = numout(first(Ms))
    R = removeunit(removeunit(first(Ms), Val(N)), Val(N - 1))
    L = removeunit(removeunit(last(Ms), Val(N)), Val(N - 1))
    p = ((1, ntuple(i -> i + 3, N - 3)..., 2, 3), (N + 1,))
    Ms[1] = tensorcontract(Ql, ((1, 2, 3), (4,)), false, R, ((1,), Tuple(2:(N - 1))), false, p)
    Ms[end] = tensorcontract(L, (codomainind(L), domainind(L)), false, Qr, (codomainind(Qr), domainind(Qr)), false, (ntuple(identity, N), (N + 1,)))
    return Ms
end

"""
Simple update with an N-site MPO `gate` (N ≥ 2).

Temporary MPO contractions and QR/LQ inputs use `allocator`.
"""
function _su_iter!(
        state::InfiniteState, gate::Vector{T}, env::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::SimpleUpdate;
        allocator = DefaultAllocator()
    ) where {T <: AbstractTensorMap}
    truncs = _get_cluster_trunc(alg.trunc, sites)
    Ms, open_vaxs, invperms = _get_cluster_with_weights(state, sites, env)
    # flip virtual arrows in `Ms` to ←
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips)
    Ql, Qr = _reduce_mpo_endpoints!(Ms)
    # apply gate MPOs and truncate
    ϵ = 0.0
    local wts
    for gate_ax in 1:2
        _apply_gatempo!(Ms, gate; gate_ax, allocator)
        wts, ϵs, = _cluster_truncate!(Ms, truncs; allocator)
        ϵ = max(ϵ, maximum(ϵs))
        alg.purified && break
    end
    _restore_mpo_endpoints!(Ms, Ql, Qr)
    # restore virtual arrows in `Ms`
    _flip_virtuals!(Ms, flips)
    # update env weights
    bond_dirs = map(sites, Iterators.drop(sites, 1)) do site1, site2
        _nn_bonddir(site1, site2)
    end
    for (wt, (dir, r, c), flip) in zip(wts, bond_dirs, flips)
        wt_new = flip ? _fliptwist_s!(wt) : wt
        wt_new = (dir in (WEST, SOUTH)) ? transpose(wt_new) : wt_new
        d = dir in (EAST, WEST) ? 1 : 2
        env[d, r, c] = normalize!(wt_new, Inf)
    end
    # update state tensors
    for (M, s, invperm, vaxs) in zip(Ms, sites, invperms, open_vaxs)
        # restore original axes order and remove weights on open axes of the cluster
        M′ = absorb_weight!(permute(M, invperm), env, s, vaxs; inv = true)
        state[s] = normalize!(M′, Inf)
    end
    return ϵ
end

"""
Get the `TruncationStrategy` for each bond in the cluster
updated by the Trotter evolution MPO.
"""
function _get_cluster_trunc(
        trunc::TruncationStrategy, sites::Vector{CartesianIndex{2}}
    )
    return map(sites, Iterators.drop(sites, 1)) do site1, site2
        (dir, r, c) = _nn_bonddir(site1, site2)
        d = dir in (EAST, WEST) ? 1 : 2
        t = truncation_strategy(trunc, d, r, c)
        if (dir in (WEST, SOUTH)) && isa(t, TruncationSpace)
            t = truncspace(flip(t.space)')
        end
        return t
    end
end

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

"""
Convert nearest neighbor vector `nn_vec` to direction labels.
```
                NORTH
                (-1,0)
                  ↑
    WEST (0,-1)-←-∘-→-(0,+1) EAST
                  ↓
                (+1,0)
                SOUTH
```
"""
function _nn_vec_direction(nn_vec::CartesianIndex{2})
    if nn_vec == CartesianIndex(-1, 0)
        return NORTH
    elseif nn_vec == CartesianIndex(0, 1)
        return EAST
    elseif nn_vec == CartesianIndex(1, 0)
        return SOUTH
    elseif nn_vec == CartesianIndex(0, -1)
        return WEST
    else
        error("Input is not a nearest neighbor vector")
    end
end

"""
Given `site1`, `site2` connected by a nearest neighbor bond,
return the bond index and whether it is reversed from the
standard orientation (`site1` on the west/south of `site2`).
"""
function _nn_bondrev(site1::CartesianIndex{2}, site2::CartesianIndex{2}, (Nrow, Ncol)::NTuple{2, Int})
    diff = site1 - site2
    if diff == CartesianIndex(0, -1)
        r, c = mod1(site1[1], Nrow), mod1(site1[2], Ncol)
        return (1, r, c), false
    elseif diff == CartesianIndex(0, 1)
        r, c = mod1(site2[1], Nrow), mod1(site2[2], Ncol)
        return (1, r, c), true
    elseif diff == CartesianIndex(1, 0)
        r, c = mod1(site1[1], Nrow), mod1(site1[2], Ncol)
        return (2, r, c), false
    elseif diff == CartesianIndex(-1, 0)
        r, c = mod1(site2[1], Nrow), mod1(site2[2], Ncol)
        return (2, r, c), true
    else
        error("`site1` and `site2` are not nearest neighbors.")
    end
end

"""
Return a size N-k tuple with values 1 to N but the missing ones. Accept k=1 and k=2.
"""
function _filtered_oneto(i, ::Val{N}) where {N}
    return ntuple(k -> k < i ? k : k + 1, N - 1)
end
function _filtered_oneto(i, j, ::Val{N}) where {N}
    lo, hi = minmax(i, j)
    return ntuple(k -> k < lo ? k : k < hi - 1 ? k + 1 : k + 2, N - 2)
end
"""
Find the permutation to permute `out_ax`, `in_ax` legs to
the first and the last position of a tensor with `Nax` legs,
then assign the last leg to domain, and the others to codomain.
"""
function _get_mpo_perm(out_ax::Integer, in_ax::Integer, ::Val{Nax}) where {Nax}
    perm = _filtered_oneto(out_ax, in_ax, Val(Nax))
    return (out_ax, perm...), (in_ax,)
end

"""
Obtain a middle cluster tensor from `state` at `site`,
where `out_ax` (`in_ax`) is the virtual axis connecting to the previous (next) tensor.
The tensor is permuted to MPS axis order.
"""
function _get_mid(
        state::InfiniteState, site::CartesianIndex{2}, out_ax::Int, in_ax::Int,
        env::SUWeight
    )
    Nr, Nc = size(state)
    n_physical_axes = numout(eltype(unitcell(state)))
    Nax = Val(4 + n_physical_axes)
    open_vaxs = _filtered_oneto(out_ax, in_ax, Val(4))
    perm = _get_mpo_perm(out_ax + n_physical_axes, in_ax + n_physical_axes, Nax)
    invperm = invbiperm(perm, Val(n_physical_axes))
    s = mod1(site[1], Nr), mod1(site[2], Nc)
    t = absorb_weight(state[s...], env, s[1], s[2], open_vaxs)
    return permute(t, perm), open_vaxs, invperm
end


function invbiperm(bituple::Tuple{Tuple, Tuple}, ::Val{N}) where {N}
    return invbiperm((first(bituple)..., last(bituple)...), Val(N))
end
function invbiperm(t::Tuple, ::Val{N}) where {N}
    p = invperm(t)
    return p[begin:N], p[(N + 1):end]
end
function cluster_truncate!(vertices, truncs, ::InfinitePEPO)
    Vphys = codomain.(vertices, 2)
    fused_vertices = [first(_fuse_physicalspaces(v)) for v in vertices]
    wts, ϵs, = _cluster_truncate!(fused_vertices, truncs)
    new_vertices = [first(_unfuse_physicalspace(v, Vphy)) for (v, Vphy) in zip(fused_vertices, Vphys)]
    return new_vertices, wts, ϵs
end

function cluster_truncate!(Ms, truncs, ::InfinitePEPS)
    wts, ϵs, = _cluster_truncate!(Ms2, truncs)
    return Ms, wts, ϵs
end
"""
Simple update with an N-site MPO `gate` (N ≥ 2).
"""
function _su_iter_mpo!(
        state::InfiniteState, gates::Vector{T}, env::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::SimpleUpdate
    ) where {T <: AbstractTensorMap}
    Nr, Nc = size(state)
    n_physical_axes = numout(eltype(unitcell(state)))
    Nax = Val(4 + n_physical_axes)
    n_sites = length(sites)
    truncs = _get_cluster_trunc(alg.trunc, sites, (Nr, Nc))
    out_axs = map(i -> _nn_vec_direction(sites[i - 1] - sites[i]), 2:n_sites)
    in_axs = map(i -> _nn_vec_direction(sites[i + 1] - sites[i]), 1:(n_sites - 1))
    # left and right: get tensor without permutation, then permute to MPS form
    left_M0, left_vaxs, = _get_left(state, sites[1], in_axs[1], env)
    right_M0, right_vaxs, = _get_right(state, sites[end], out_axs[end], env)
    left_perm = _get_mpo_perm(mod1(2 + in_axs[1], 4) + n_physical_axes, in_axs[1] + n_physical_axes, Nax)
    right_perm = _get_mpo_perm(out_axs[end] + n_physical_axes, mod1(2 + out_axs[end], 4) + n_physical_axes, Nax)
    left_M = permute(left_M0, left_perm)
    right_M = permute(right_M0, right_perm)
    left_invperm = invbiperm(left_perm, Val(n_physical_axes))
    right_invperm = invbiperm(right_perm, Val(n_physical_axes))
    # middle tensors: permuted to MPS form in _get_mid
    mids = map(i -> _get_mid(state, sites[i], out_axs[i - 1], in_axs[i], env), 2:(n_sites - 1))
    vertices = [left_M, getindex.(mids, 1)..., right_M]  # TODO remove
    # Ms has well defined eltype Here
    # issue it is redefined later with Any eltype
    open_vaxs = [left_vaxs, getindex.(mids, 2)..., right_vaxs] # TODO removve
    # open_vaxs however cannot be stable
    invperms = [left_invperm, getindex.(mids, 3)..., right_invperm]
    flips = push!([isdual(space(first(x), 1)) for x in mids], isdual(space(right_M, 1)))
    # flip virtual arrows in `vertices` to ←
    _flip_virtuals!(vertices, flips)

    # apply gate MPOs and truncate
    _apply_gatempo!(vertices, gates; gate_ax = 1)
    new_vertices, wts, ϵs = cluster_truncate!(vertices, truncs, state)
    if !alg.purified
        _apply_gatempo!(new_vertices, gates; gate_ax = 2)
        new_vertices, wts, ϵs = cluster_truncate!(new_vertices, truncs, state)
    end

    # restore virtual arrows in `new_vertices`
    _flip_virtuals!(new_vertices, flips)
    # update env weights
    bond_revs = map(zip(sites, Iterators.drop(sites, 1))) do (site1, site2)
        _nn_bondrev(site1, site2, (Nr, Nc))
    end
    for (wt, (bond, rev), flip) in zip(wts, bond_revs, flips)
        wt_new = flip ? _fliptwist_s(wt) : wt
        wt_new = rev ? transpose(wt_new) : wt_new
        @assert all(wt_new.data .>= 0)
        env[CartesianIndex(bond)] = normalize!(wt_new, Inf)
    end
    for (vertex, s, invperm, vaxs) in zip(new_vertices, sites, invperms, open_vaxs)
        s′ = CartesianIndex(mod1(s[1], Nr), mod1(s[2], Nc))
        # restore original axes order
        permuted = permute(vertex, invperm)
        # remove weights on open axes of the cluster and update state
        state[s′] = absorb_weight(permuted, env, s′[1], s′[2], vaxs; inv = true)
    end
    return maximum(ϵs)
end

"""
Get the `TruncationStrategy` for each bond in the cluster
updated by the Trotter evolution MPO.
"""
function _get_cluster_trunc(
        trunc::TruncationStrategy, sites::Vector{CartesianIndex{2}},
        (Nrow, Ncol)::NTuple{2, Int}
    )
    return map(zip(sites, Iterators.drop(sites, 1))) do (site1, site2)
        diff = site2 - site1
        if diff == CartesianIndex(0, 1)
            r, c = mod1(site1[1], Nrow), mod1(site1[2], Ncol)
            return truncation_strategy(trunc, 1, r, c)
        elseif diff == CartesianIndex(0, -1)
            r, c = mod1(site2[1], Nrow), mod1(site2[2], Ncol)
            return truncation_strategy(trunc, 1, r, c)
        elseif diff == CartesianIndex(1, 0)
            r, c = mod1(site2[1], Nrow), mod1(site2[2], Ncol)
            return truncation_strategy(trunc, 2, r, c)
        elseif diff == CartesianIndex(-1, 0)
            r, c = mod1(site1[1], Nrow), mod1(site1[2], Ncol)
            return truncation_strategy(trunc, 2, r, c)
        else
            error("The path `sites` contains a long-range bond.")
        end
    end
end

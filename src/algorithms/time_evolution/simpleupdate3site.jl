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
Obtain the cluster `Ms` along the (open) path `sites` in `state`. 

When the `SUWeight` environment `env` is provided,
it will be absorbed into tensors of `Ms`.

When `permute = true`, permute tensors in `Ms` to MPS axis order
```
    PEPS:           PEPO:
           3             3  4
          ╱              | ╱
    o -- M -- i     o -- M -- i
       ╱ |             ╱ |
      4  2            5  2
    M[o 2 3 4; i]  M[o 2 3 4 5; i]
```
where `o` (`i`) connects to the previous (next) tensor.
Otherwise, axes order of each tensor in `Ms` are preserved.

## Returns

- `vertices`: Tensors in the cluster.
- `open_vaxs`: Open virtual axes (1 to 4) of each cluster tensor before permutation.
- `invperms`: Permutations to restore the axes order of each cluster tensor.
"""
function _get_cluster(
        state::InfiniteState, sites::Vector{CartesianIndex{2}},
        env::SUWeight; permute::Bool = true
    )
    Nr, Nc = size(state)
    n_sites = length(sites)
    n_physical_axes = numout(eltype(unitcell(state)))
    # number of axes of each state tensor
    Nax = Val(4 + n_physical_axes)
    out_axs = map(2:n_sites) do i
        return _nn_vec_direction(sites[i - 1] - sites[i])
    end
    in_axs = map(1:(n_sites - 1)) do i
        return _nn_vec_direction(sites[i + 1] - sites[i])
    end
    first_open_vaxs = _filtered_oneto(in_axs[1], Val(4))
    last_open_vaxs = _filtered_oneto(out_axs[n_sites - 1], Val(4))
    mid_vaxs = map(i -> _filtered_oneto(out_axs[i - 1], in_axs[i], Val(4)), 2:(n_sites - 1))
    # use direction opposite to `in` as `out`
    first_perm = _get_mpo_perm(mod1(2 + in_axs[1], 4) + n_physical_axes, in_axs[1] + n_physical_axes, Nax)
    # use direction opposite to `out` as `in`
    last_perm = _get_mpo_perm(out_axs[n_sites - 1] + n_physical_axes, mod1(2 + out_axs[n_sites - 1], 4) + n_physical_axes, Nax)
    mid_perms = map(2:(n_sites - 1)) do i
        return _get_mpo_perm(out_axs[i - 1] + n_physical_axes, in_axs[i] + n_physical_axes, Nax)
    end

    open_vaxs = [first_open_vaxs, mid_vaxs..., last_open_vaxs]
    perms = [first_perm, mid_perms..., last_perm]
    invperms = invbiperm.(perms, Val(n_physical_axes))
    vertices = map(
        zip(sites, open_vaxs, perms)
    ) do (site, vaxs, perm)
        s = CartesianIndex(mod1(site[1], Nr), mod1(site[2], Nc))
        t = absorb_weight(state[s], env, s[1], s[2], vaxs)
        return permute ? TensorKit.permute(t, perm) : t
    end
    return vertices, open_vaxs, invperms
end

function invbiperm(bituple::Tuple{Tuple, Tuple}, ::Val{N}) where {N}
    return invbiperm((first(bituple)..., last(bituple)...), Val(N))
end
function invbiperm(t::Tuple, ::Val{N}) where {N}
    p = invperm(t)
    return p[begin:N], p[(N + 1):end]
end
"""
Simple update with an N-site MPO `gate` (N ≥ 2).
"""
function _su_iter_mpo!(
        state::InfiniteState, gates::Vector{T}, env::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::SimpleUpdate
    ) where {T <: AbstractTensorMap}
    Nr, Nc = size(state)
    truncs = _get_cluster_trunc(alg.trunc, sites, (Nr, Nc))
    Ms, open_vaxs, invperms = _get_cluster(state, sites, env)
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    Vphys = [codomain(M, 2) for M in Ms]
    normalize!.(Ms, Inf)
    # flip virtual arrows in `Ms` to ←
    _flip_virtuals!(Ms, flips)
    # apply gate MPOs and truncate
    gate_axs = alg.purified ? (1:1) : (1:2)
    global wts, ϵs
    for gate_ax in gate_axs
        _apply_gatempo!(Ms, gates; gate_ax)
        if isa(state, InfinitePEPO)
            Ms = [first(_fuse_physicalspaces(M)) for M in Ms]
        end
        wts, ϵs, = _cluster_truncate!(Ms, truncs)
        if isa(state, InfinitePEPO)
            Ms = [first(_unfuse_physicalspace(M, Vphy)) for (M, Vphy) in zip(Ms, Vphys)]
        end
    end
    # restore virtual arrows in `Ms`
    _flip_virtuals!(Ms, flips)
    # update env weights
    bond_revs = map(zip(sites, Iterators.drop(sites, 1))) do (site1, site2)
        _nn_bondrev(site1, site2, (Nr, Nc))
    end
    for (wt, (bond, rev), flip) in zip(wts, bond_revs, flips)
        wt_new = flip ? _fliptwist_s(wt) : wt
        wt_new = rev ? transpose(wt_new) : wt_new
        @assert all(wt_new.data .>= 0)
        env[CartesianIndex(bond)] = normalize(wt_new, Inf)
    end
    for (M, s, invperm, vaxs) in zip(Ms, sites, invperms, open_vaxs)
        s′ = CartesianIndex(mod1(s[1], Nr), mod1(s[2], Nc))
        # restore original axes order
        M = permute(M, invperm)
        # remove weights on open axes of the cluster
        M = absorb_weight(M, env, s′[1], s′[2], vaxs; inv = true)
        # update state tensors
        state[s′] = normalize(M, Inf)
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

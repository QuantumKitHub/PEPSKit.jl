# code to extract and permute tensors (the cluster) updated by a gate

"""
Find the permutation to permute `out_ax`, `in_ax` legs to
the first and the last position of a tensor with `N` legs,
then assign the last leg to domain, and the others to codomain
to follow the `GenericMPSTensor` convention.
"""
function _mpo_perm(out_ax::Integer, in_ax::Integer, ::Val{N}) where {N}
    perm = TupleTools.deleteat(ntuple(identity, N), (out_ax, in_ax))
    return (out_ax, perm...), (in_ax,)
end

"""
Given the permutation `perm` that converts a PEPSTensor (N = 1)
or PEPOTensor (N = 2) to MPS axis order, find the inverse permutation.
"""
function _inv_mpo_perm(perm::Tuple{Tuple, Tuple}, ::Val{N}) where {N}
    return _inv_mpo_perm((first(perm)..., last(perm)...), Val(N))
end
function _inv_mpo_perm(perm::Tuple, ::Val{N}) where {N}
    p = invperm(perm)
    return (p[begin:N], p[(N + 1):end])
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
function _nn_bondrev(site1::CartesianIndex{2}, site2::CartesianIndex{2})
    diff = site1 - site2
    if diff == CartesianIndex(0, -1)
        r, c = site1[1], site1[2]
        return (1, r, c), false
    elseif diff == CartesianIndex(0, 1)
        r, c = site2[1], site2[2]
        return (1, r, c), true
    elseif diff == CartesianIndex(1, 0)
        r, c = site1[1], site1[2]
        return (2, r, c), false
    elseif diff == CartesianIndex(-1, 0)
        r, c = site2[1], site2[2]
        return (2, r, c), true
    else
        error("`site1` and `site2` are not nearest neighbors.")
    end
end

function _bond_rotation(x, bonddir::Int, rev::Bool; inv::Bool = false)
    return if bonddir == 1 # x-bond
        rev ? rot180(x) : x
    elseif bonddir == 2 # y-bond
        if rev
            inv ? rotr90(x) : rotl90(x)
        else
            inv ? rotl90(x) : rotr90(x)
        end
    else
        error("`bonddir` must be 1 (for x-bonds) or 2 (for y-bonds).")
    end
end
function _bond_rotation(x::CartesianIndex{2}, bonddir::Int, rev::Bool, unitcell::NTuple{2, Int})
    return if bonddir == 1
        rev ? siterot180(x, unitcell) : x
    else
        rev ? siterotl90(x, unitcell) : siterotr90(x, unitcell)
    end
end

"""
Obtain the cluster `Ms` along the (open) path `sites` in `state`.

## Returns

- `Ms`: Tensors in the cluster (not permuted to MPS axis order).
- `open_vaxs`: Open virtual axes (1 to 4) of each cluster tensor before permutation.
- `perms`: Permutations to change each tensor to MPS axis order.
"""
function _get_cluster(
        state::InfiniteState, sites::Vector{CartesianIndex{2}}
    )
    # number of sites
    Ns = length(sites)
    # number of physical axes
    Np = isa(state, InfinitePEPS) ? 1 : 2
    # number of axes of each state tensor
    Nax = Val(4 + Np)
    out_axs = map(2:Ns) do i
        return _nn_vec_direction(sites[i - 1] - sites[i])
    end
    in_axs = map(1:(Ns - 1)) do i
        return _nn_vec_direction(sites[i + 1] - sites[i])
    end
    open_vaxs = map(1:Ns) do i
        return if i == 1
            TupleTools.deleteat((1, 2, 3, 4), in_axs[i])
        elseif i == Ns
            TupleTools.deleteat((1, 2, 3, 4), out_axs[i - 1])
        else
            TupleTools.deleteat((1, 2, 3, 4), (out_axs[i - 1], in_axs[i]))
        end
    end
    perms = map(1:Ns) do i
        out_ax, in_ax = if i == 1
            # use direction opposite to `in` as `out`
            mod1(2 + in_axs[i], 4), in_axs[i]
        elseif i == Ns
            # use direction opposite to `out` as `in`
            out_axs[i - 1], mod1(2 + out_axs[i - 1], 4)
        else
            out_axs[i - 1], in_axs[i]
        end
        return _mpo_perm(out_ax + Np, in_ax + Np, Nax)
    end
    Ms = map(site -> state[site], sites)
    return Ms, open_vaxs, perms
end

"""
Absorb weights into the open virtual legs of
the cluster `Ms` of PEPS/PEPO tensors.
"""
function _absorb_weight!(
        Ms::Vector{T}, sites::Vector{CartesianIndex{2}},
        open_vaxs::Vector{<:Tuple}, env::SUWeight; inv::Bool = false
    ) where {T}
    for (idx, (M, site, vaxs)) in enumerate(zip(Ms, sites, open_vaxs))
        Ms[idx] = absorb_weight(M, env, site, vaxs; inv)
    end
    return Ms
end

"""
Given a vector `Ms` of `AbstractTensorMap`s,
apply permutations `perms` to each tensor.
"""
function _permute_cluster(Ms::Vector{<:AbstractTensorMap}, perms::Vector{<:Tuple})
    return map(Ms, perms) do M, perm
        return permute(M, perm)
    end
end

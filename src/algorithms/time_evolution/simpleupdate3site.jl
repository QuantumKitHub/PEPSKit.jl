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

const openaxs_se = [(NORTH, SOUTH, WEST), (EAST, SOUTH), (NORTH, EAST, WEST)]
const invperms_se_peps = [((2,), (3, 5, 4, 1)), ((2,), (5, 3, 4, 1)), ((2,), (5, 3, 1, 4))]
const perms_se_peps = map(invperms_se_peps) do (p1, p2)
    p = invperm((p1..., p2...))
    return (p[1:(end - 1)], (p[end],))
end
const invperms_se_pepo = [((2, 3), (4, 6, 5, 1)), ((2, 3), (6, 4, 5, 1)), ((2, 3), (6, 4, 1, 5))]
const perms_se_pepo = map(invperms_se_pepo) do (p1, p2)
    p = invperm((p1..., p2...))
    return (p[1:(end - 1)], (p[end],))
end
"""
Obtain the 3-site cluster in the "southeast corner" of a square plaquette.
``` 
    r-1         M3
                |
                ↓
    r   M1 -←- M2
        c      c+1
```
"""
function get_3site_se(state::InfiniteState, env::SUWeight, row::Int, col::Int)
    Nr, Nc = size(state)
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    coords_se = [(row, col), (row, cp1), (rm1, cp1)]
    perms_se = isa(state, InfinitePEPS) ? perms_se_peps : perms_se_pepo
    Ms = map(zip(coords_se, perms_se, openaxs_se)) do (coord, perm, openaxs)
        M = absorb_weight(state.A[CartesianIndex(coord)], env, coord[1], coord[2], openaxs)
        # permute to MPS axes order
        return permute(M, perm)
    end
    return Ms
end

function _su3site_se!(
        state::InfiniteState, gs::Vector{T}, env::SUWeight,
        row::Int, col::Int, truncs::Vector{E};
        purified::Bool = true
    ) where {T <: AbstractTensorMap, E <: TruncationStrategy}
    Nr, Nc = size(state)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    # southwest 3-site cluster and arrow direction within it
    Ms = get_3site_se(state, env, row, col)
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    Vphys = [codomain(M, 2) for M in Ms]
    normalize!.(Ms, Inf)
    # flip virtual arrows in `Ms` to ←
    _flip_virtuals!(Ms, flips)
    # sites in the cluster
    coords = ((row, col), (row, cp1), (rm1, cp1))
    # weights in the cluster
    wt_idxs = ((1, row, col), (2, row, cp1))
    # apply gate MPOs and truncate
    gate_axs = purified ? (1:1) : (1:2)
    ϵs = nothing
    for gate_ax in gate_axs
        _apply_gatempo!(Ms, gs; gate_ax)
        if isa(state, InfinitePEPO)
            Ms = [first(_fuse_physicalspaces(M)) for M in Ms]
        end
        wts, ϵs, = _cluster_truncate!(Ms, truncs)
        if isa(state, InfinitePEPO)
            Ms = [first(_unfuse_physicalspace(M, Vphy)) for (M, Vphy) in zip(Ms, Vphys)]
        end
        for (wt, wt_idx, flip) in zip(wts, wt_idxs, flips)
            env[CartesianIndex(wt_idx)] = normalize(flip ? _fliptwist_s(wt) : wt, Inf)
        end
    end
    # restore virtual arrows in `Ms`
    _flip_virtuals!(Ms, flips)
    # update `state` from `Ms`
    invperms_se = isa(state, InfinitePEPS) ? invperms_se_peps : invperms_se_pepo
    for (M, coord, invperm, openaxs, Vphy) in zip(Ms, coords, invperms_se, openaxs_se, Vphys)
        # restore original axes order
        M = permute(M, invperm)
        # remove weights on open axes of the cluster
        M = absorb_weight(M, env, coord[1], coord[2], openaxs; inv = true)
        state.A[CartesianIndex(coord)] = normalize(M, Inf)
    end
    return maximum(ϵs)
end

function su_iter(
        state::InfiniteState, gatempos::Vector{G}, alg::SimpleUpdate, env::SUWeight
    ) where {G <: AbstractMatrix}
    if state isa InfinitePEPO
        @assert size(state, 3) == 1
    end
    Nr, Nc = size(state)[1:2]
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError(
            "iPEPS unit cell size for simple update should be no smaller than (2, 2)."
        ),
    )
    state2, env2, ϵ = deepcopy(state), deepcopy(env), 0.0
    trunc = alg.trunc
    for i in 1:4
        Nr, Nc = size(state2)[1:2]
        for r in 1:Nr, c in 1:Nc
            gs = gatempos[i][r, c]
            truncs = [
                truncation_strategy(trunc, 1, r, c)
                truncation_strategy(trunc, 2, r, _next(c, Nc))
            ]
            ϵ = _su3site_se!(state2, gs, env2, r, c, truncs; alg.purified)
        end
        state2, env2 = rotl90(state2), rotl90(env2)
        trunc = rotl90(trunc)
    end
    return state2, env2, ϵ
end

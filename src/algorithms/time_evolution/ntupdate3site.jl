function _get_cluster_permute(
        state::InfiniteState, sites::Vector{CartesianIndex{2}}
    )
    Ms, _, perms = _get_cluster(state, sites)
    Np = (state isa InfinitePEPS) ? Val(1) : Val(2)
    invperms = map(p -> _inv_mpo_perm(p, Np), perms)
    return _permute_cluster(Ms, perms), invperms
end

"""
Apply an N-site MPO `gate` (N ≥ 2) to the owned NTU working state and weights.
"""
function _ntu_iter!(
        state::InfiniteState, gate::Vector{T}, wts::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::NeighbourUpdate;
        reverse_trunc::Bool = false
    ) where {T <: AbstractTensorMap}
    truncs = _get_cluster_trunc(alg.opt_alg.trunc, sites)

    Ms, invperms = _get_cluster_permute(state, sites)
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←

    # apply gate MPO without truncation
    _apply_gatempo!(Ms, gate)
    _flip_virtuals!(Ms, flips) # restore virtual arrows in `Ms`
    for (M, s, invperm) in zip(Ms, sites, invperms)
        state[s] = permute(M, invperm)
    end

    # truncate each bond sequentially along the path
    # if reverse_trunc, truncate in the reverse order
    info = (; fid = 1.0)
    nbond = length(sites) - 1
    for i in 1:nbond
        stype1 = (i == 1) ? :first : :middle
        stype2 = (i == nbond) ? :last : :middle
        # the bond to be updated
        bond_id = reverse_trunc ? (nbond + 1 - i) : i
        # the sites involved
        bondsites = if reverse_trunc
            (sites[bond_id + 1], sites[bond_id])
        else
            (sites[bond_id], sites[bond_id + 1])
        end
        # truncate the bond
        trunc = truncs[bond_id]
        alg′ = (@set alg.opt_alg.trunc = trunc)
        state, wts, info′ = _bond_truncate!(state, wts, bondsites, (stype1, stype2), alg′)
        # record the worst fidelity
        (info′.fid < info.fid) && (info = info′)
    end
    return state, wts, info
end

"""
Truncate a nearest neighbor bond between `site1` and `site2` in place,
using the standard x direction `A ← B` for the local environment contractions.
"""
function _bond_truncate!(
        state::InfiniteState, wts::SUWeight,
        (site1, site2)::NTuple{2, CartesianIndex{2}},
        (stype1, stype2)::NTuple{2, Symbol},
        alg::NeighbourUpdate; gate::Union{NNGate, Nothing} = nothing
    )
    # Rotate only the endpoints; environment neighbors are rotated when collected.
    (dir, r, c) = _nn_bonddir(site1, site2)
    A = _bond_rotation(state[site1], dir, EAST)
    B = _bond_rotation(state[site2], dir, EAST)

    # create bond environment
    qrtrunc = trunctol(; rtol = 1.0e-12)
    a, X = if stype1 == :first
        bond_tensor_first(A; trunc = qrtrunc)
    else
        @assert stype1 == :middle
        bond_tensor_midnext(A; trunc = qrtrunc)
    end
    b, Y = if stype2 == :last
        bond_tensor_last(B; trunc = qrtrunc)
    else
        @assert stype2 == :middle
        bond_tensor_midprev(B; trunc = qrtrunc)
    end
    benv = bondenv_ntu(site1[1], site1[2], dir, X, Y, state, alg.bondenv_alg)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(benv))"
    if alg.fixgauge
        Z = positive_approx(benv)
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X = _fixgauge_benvX(X, Rinv)
        Y = _fixgauge_benvY(Y, Linv)
        benv = Z' * Z
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(benv))"
    end

    # (optional) apply the NN gate without truncation
    if !(gate === nothing)
        a, _, b, = _apply_bond_gate(a, b, gate, truncerror(; atol = 1.0e-15))
    end
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)

    A = if stype1 == :first
        undo_bond_tensor_first(a, X)
    else
        undo_bond_tensor_midnext(a, X)
    end
    B = if stype2 == :last
        undo_bond_tensor_last(b, Y)
    else
        undo_bond_tensor_midprev(b, Y)
    end

    state[site1] = normalize!(_bond_rotation(A, EAST, dir), Inf)
    state[site2] = normalize!(_bond_rotation(B, EAST, dir), Inf)
    d = dir in (EAST, WEST) ? 1 : 2
    wts[d, r, c] = normalize!(dir in (WEST, SOUTH) ? transpose(s) : s, Inf)
    return state, wts, info
end

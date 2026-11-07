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
        reverse_trunc::Bool = false, allocator = DefaultAllocator()
    ) where {T <: AbstractTensorMap}
    truncs = _get_cluster_trunc(alg.opt_alg.trunc, sites)

    Ms, invperms = _get_cluster_permute(state, sites)
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←

    # apply gate MPO without truncation
    _apply_gatempo!(Ms, gate; allocator)
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
        state, wts, info′ = _bond_truncate!(state, wts, bondsites, (stype1, stype2), alg′; allocator)
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
        alg::NeighbourUpdate; gate::Union{NNGate, Nothing} = nothing,
        allocator = DefaultAllocator()
    )
    # Reduce in the original frame; orient only the orthogonal factors for the environment.
    (dir, r, c) = _nn_bonddir(site1, site2)
    A = state[site1]
    B = state[site2]
    dir_opposite = mod1(dir + 2, 4)

    # create bond environment
    # MPO-expanded tensors need rank compression; NN gates are applied after reduction.
    orth_kwargs = isnothing(gate) ? (; trunc = trunctol(; rtol = 1.0e-12)) : (; positive = true)
    X, a = if stype1 == :first
        bond_tensor_end(A, dir; orth_kwargs...)
    else
        @assert stype1 == :middle
        bond_tensor_mid(A, dir; orth_kwargs...)
    end
    Y, b = if stype2 == :last
        bond_tensor_end(B, dir_opposite; orth_kwargs...)
    else
        @assert stype2 == :middle
        bond_tensor_mid(B, dir_opposite; orth_kwargs...)
    end
    X, Y = _bond_rotation(X, dir, EAST), _bond_rotation(Y, dir, EAST)
    # Gauge fixing and bond truncation use the MPS-chain layouts on both sides.
    a_chain = permute(a, ((1, 2), (3,)))
    b_chain = permute(b, ((3, 2), (1,)))
    benv = bondenv_ntu(site1[1], site1[2], dir, X, Y, state, alg.bondenv_alg; allocator)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(benv))"
    if alg.fixgauge
        Z = positive_approx(benv)
        Z, a_chain, b_chain, (Linv, Rinv) = fixgauge_benv(Z, a_chain, b_chain)
        X = _fixgauge_benvX(X, Rinv)
        Y = _fixgauge_benvY(Y, Linv)
        benv = Z' * Z
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(benv))"
    end

    # (optional) apply the NN gate without truncation
    if !(gate === nothing)
        a = permute(a_chain, ((1,), (2, 3)))
        b = permute(b_chain, ((3,), (2, 1)))
        a, _, b, = _apply_bond_gate(a, b, gate, truncerror(; atol = 1.0e-15))
        a_chain = permute(a, ((1, 2), (3,)))
        b_chain = permute(b, ((3, 2), (1,)))
    end
    a_chain, s, b_chain, info = bond_truncate(a_chain, b_chain, benv, alg.opt_alg)

    X, Y = _bond_rotation(X, EAST, dir), _bond_rotation(Y, EAST, dir)
    a = permute(a_chain, ((1,), (2, 3)))
    A = if stype1 == :first
        undo_bond_tensor_end(a, X, dir)
    else
        undo_bond_tensor_mid(a, X, dir)
    end
    b = permute(b_chain, ((3,), (2, 1)))
    B = if stype2 == :last
        undo_bond_tensor_end(b, Y, dir_opposite)
    else
        undo_bond_tensor_mid(b, Y, dir_opposite)
    end

    state[site1] = normalize!(A, Inf)
    state[site2] = normalize!(B, Inf)
    d = dir in (EAST, WEST) ? 1 : 2
    wts[d, r, c] = normalize!(dir in (WEST, SOUTH) ? transpose(s) : s, Inf)
    return state, wts, info
end

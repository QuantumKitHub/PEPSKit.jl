"""
Neighbourhood tensor update with N-site MPO `gate` (N ≥ 2).
"""
function _ntu_iter(
        state::InfiniteState, gate::Vector{T}, wts::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::NeighbourUpdate
    ) where {T <: AbstractTensorMap}
    Nr, Nc = size(state)
    truncs = _get_cluster_trunc(alg.opt_alg.trunc, sites, (Nr, Nc))
    state, wts = copy(state), deepcopy(wts)

    Ms, _, invperms = _get_cluster(state, sites)
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←

    # apply gate MPO without truncation
    _apply_gatempo!(Ms, gate)
    _flip_virtuals!(Ms, flips) # restore virtual arrows in `Ms`
    for (M, s, invperm) in zip(Ms, sites, invperms)
        s′ = CartesianIndex(mod1(s[1], Nr), mod1(s[2], Nc))
        state[s′] = permute(M, invperm)
    end

    # truncate each bond sequentially along the path
    info = (; fid = 1.0)
    nbond = length(sites) - 1
    for (i, bondsites) in enumerate(zip(sites, Iterators.drop(sites, 1)))
        trunc = truncs[i]
        alg′ = (@set alg.opt_alg.trunc = trunc)
        stype1 = (i == 1) ? :first : :middle
        stype2 = (i == nbond) ? :last : :middle
        state, wts, info′ = _bond_truncate(state, wts, bondsites, (stype1, stype2), alg′)
        # record the worst fidelity
        (info′.fid < info.fid) && (info = info′)
    end
    return state, wts, info
end

"""
Truncate a nearest neighbor bond between `site1` and `site2`
after rotating the bond to standard x direction `A ← B`.

`bondtype` takes values in (1, 2, 3), meaning that the current bond is
(the first, a middle, the last) bond in the updated cluster.
"""
function _bond_truncate(
        state::InfiniteState, wts::SUWeight,
        (site1, site2)::NTuple{2, CartesianIndex{2}},
        (stype1, stype2)::NTuple{2, Symbol},
        alg::NeighbourUpdate; gate::Union{NNGate, Nothing} = nothing
    )
    # rotate bond to standard x direction `A ← B`
    ucell = size(state)[1:2]
    bond, rev = _nn_bondrev(site1, site2, ucell)
    state2 = _bond_rotation(state, bond[1], rev; inv = false)
    wts2 = _bond_rotation(wts, bond[1], rev; inv = false)

    # rotated bond tensors
    siteA = _bond_rotation(site1, bond[1], rev, ucell)
    row, col = mod1.(Tuple(siteA), size(state2)[1:2])
    cp1 = _next(col, size(state2, 2))
    A, B = state2[row, col], state2[row, cp1]

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
    benv = bondenv_ntu(row, col, X, Y, state2, alg.bondenv_alg)
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
        a, s, b, = _apply_gate(a, b, gate, truncerror(; atol = 1.0e-15))
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

    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    state2[row, col], state2[row, cp1], wts2[1, row, col] = A, B, s

    # rotate back tensors and bond weight
    state2 = _bond_rotation(state2, bond[1], rev; inv = true)
    wts2 = _bond_rotation(wts2, bond[1], rev; inv = true)
    return state2, wts2, info
end

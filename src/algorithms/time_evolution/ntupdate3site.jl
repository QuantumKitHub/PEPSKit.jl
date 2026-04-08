"""
Neighbourhood tensor update with N-site MPO `gate` (N ≥ 2).
"""
function _ntu_iter(
        state::InfiniteState, gate::Vector{T}, wts::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::NeighbourUpdate
    ) where {T <: AbstractTensorMap}
    Nr, Nc = size(state)

    # apply gate MPO without truncation
    Ms, _, invperms = _get_cluster(state, sites)
    flips = [isdual(space(M, 1)) for M in Ms[2:end]]
    _flip_virtuals!(Ms, flips) # flip virtual arrows in `Ms` to ←
    _apply_gatempo!(Ms, gate)
    _flip_virtuals!(Ms, flips) # restore virtual arrows in `Ms`
    state2, wts2 = deepcopy(state), deepcopy(wts)
    for (M, s, invperm) in zip(Ms, sites, invperms)
        s′ = CartesianIndex(mod1(s[1], Nr), mod1(s[2], Nc))
        state2[s′] = permute(M, invperm)
    end

    # truncate each bond sequentially along the path
    info = (; fid = 1.0)
    for bondsites in zip(sites, Iterators.drop(sites, 1))
        state2, wts2, info′ = _bond_truncate(state2, wts2, bondsites, alg)
        # record the worst fidelity
        (info′.fid < info.fid) && (info = info′)
    end
    return state2, wts2, info
end

"""
Truncate a nearest neighbor bond between `site1` and `site2`
after rotating the bond to standard x direction `A ← B`.
"""
function _bond_truncate(
        state::InfiniteState, wts::SUWeight,
        (site1, site2)::NTuple{2, CartesianIndex{2}},
        alg::NeighbourUpdate; gate::Union{NNGate, Nothing} = nothing
    )
    # rotate bond to standard x direction `A ← B`
    ucell = size(state)[1:2]
    bond, rev = _nn_bondrev(site1, site2, ucell)
    state2 = _bond_rotation(state, bond[1], rev; inv = false)
    wts2 = _bond_rotation(wts, bond[1], rev; inv = false)

    # rotated bond tensors
    siteA = if bond[1] == 1
        rev ? siterot180(site1, ucell) : site1
    else
        rev ? siterotl90(site1, ucell) : siterotr90(site1, ucell)
    end
    row, col = mod1.(Tuple(siteA), size(state2)[1:2])
    cp1 = _next(col, size(state2, 2))
    A, B = state2[row, col], state2[row, cp1]

    # create bond environment
    X, a, b, Y = _qr_bond(A, B; trunc = trunctol(; rtol = 1.0e-12))
    benv = bondenv_ntu(row, col, X, Y, state2, alg.bondenv_alg)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(benv))"
    if alg.fixgauge
        Z = positive_approx(benv)
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
        benv = Z' * Z
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(benv))"
    end

    # (optional) apply the NN gate
    if !(gate === nothing)
        a, s, b, = _apply_gate(a, b, gate, truncerror(; atol = 1.0e-15))
    else
        a = permute(a, ((1, 2), (3,)))
        b = permute(b, ((1,), (2, 3)))
    end

    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    A, B = _qr_bond_undo(X, a, b, Y)
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    state2[row, col], state2[row, cp1], wts2[1, row, col] = A, B, s

    # rotate back tensors and bond weight
    state2 = _bond_rotation(state2, bond[1], rev; inv = true)
    wts2 = _bond_rotation(wts2, bond[1], rev; inv = true)
    return state2, wts2, info
end

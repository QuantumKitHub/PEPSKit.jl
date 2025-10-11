function _transfer_left(
        vec::GenericMPSTensor{S, 2}, d::MPSKit.SingleTransferMatrix
    ) where {S}
    return @tensor y[-1 -2; -3] := vec[1 2; 4] *
        d.above[4 5; -3] * d.middle[2 3; 5 -2] * conj(d.below[1 3; -1])
end

function _transfer_left(
        vec::GenericMPSTensor{S, 3}, d::MPSKit.SingleTransferMatrix
    ) where {S}
    return @tensor y[-1 dstring -2; -3] := vec[1 dstring 2; 4] *
        d.above[4 5; -3] * d.middle[2 3; 5 -2] * conj(d.below[1 3; -1])
end

function correlator_horizontal(
        ρ::InfinitePEPO, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    all(==(i[1]) ∘ first ∘ Tuple, js) ||
        throw(ArgumentError("Not a horizontal correlation function"))
    issorted(vcat(i, js); by = last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))
    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))
    # preallocate with correct scalartype
    G = similar(
        js,
        TensorOperations.promote_contract(
            scalartype(ρ), scalartype(env), scalartype.(O)...
        ),
    )
    # left start for operator and norm contractions
    Vn, Vo = start_correlator(i, ρ, O[1], env)
    i += CartesianIndex(0, 1)
    for (k, j) in enumerate(js)
        # transfer until left of site j
        while j > i
            Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
            Amid = trace_physicalspaces(ρ[mod1(i[1], end), mod1(i[2], end)])
            Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
            T = TransferMatrix(Atop, Amid, _dag(Abot))
            Vn = _transfer_left(Vn, T)
            Vo = _transfer_left(Vo, T)
            i += CartesianIndex(0, 1)
        end
        # compute overlap with operator
        numerator = end_correlator_numerator(j, Vo, ρ, O[2], env)
        # transfer right of site j
        Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
        Amid = trace_physicalspaces(ρ[mod1(i[1], end), mod1(i[2], end)])
        Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
        T = TransferMatrix(Atop, Amid, _dag(Abot))
        Vn = _transfer_left(Vn, T)
        if k < length(js)
            Vo = _transfer_left(Vo, T)
        end
        i += CartesianIndex(0, 1)
        # compute overlap without operator
        denominator = end_correlator_denominator(j, Vn, env)
        G[k] = numerator / denominator
    end
    return G
end

function start_correlator(
        i::CartesianIndex{2}, ρ::InfinitePEPO,
        O::MPOTensor, env::CTMRGEnv
    )
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(i)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    E_west = env.edges[WEST, mod1(r, end), _prev(c, end)]
    C_northwest = env.corners[NORTHWEST, _prev(r, end), _prev(c, end)]
    C_southwest = env.corners[SOUTHWEST, _next(r, end), _prev(c, end)]
    t = twistdual(ρ[mod1(r, end), mod1(c, end)], 1:2)
    # TODO: part of these contractions is duplicated between the two output tensors,
    # so could be optimized further
    @autoopt @tensor Vn[χSE De; χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        t[d d; Dn De Ds Dw] * E_north[χN Dn; χNE]
    @autoopt @tensor Vo[χSE dstring De; χNE] :=
        E_south[χSE Ds; χSW2] * C_southwest[χSW2; χSW] *
        E_west[χSW Dw; χNW] * C_northwest[χNW; χN] *
        removeunit(O, 1)[d2; d1 dstring] *
        t[d1 d2; Dn De Ds Dw] * E_north[χN Dn; χNE]
    return Vn, Vo
end

function end_correlator_numerator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 3, 1},
        ρ::InfinitePEPO,
        O::MPOTensor, env::CTMRGEnv
    ) where {T, S}
    (size(ρ, 3) == 1) ||
        throw(ArgumentError("The input PEPO ρ must have only one layer."))
    r, c = Tuple(j)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    t = twistdual(ρ[mod1(r, end), mod1(c, end)], 1:2)
    return @autoopt @tensor V[χSW dstring DW; χNW] *
        E_south[χSSE DS; χSW] * E_east[χNEE DE; χSEE] * E_north[χNW DN; χNNE] *
        C_northeast[χNNE; χNEE] * C_southeast[χSEE; χSSE] *
        t[d1 d2; DN DE DS DW] * removeunit(O, 4)[dstring d2; d1]
end

function end_correlator_denominator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 2, 1}, env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    return @autoopt @tensor V[χS DE; χN] * C_northeast[χN; χNE] *
        E_east[χNE DE; χSE] * C_southeast[χSE; χS]
end

function correlator_vertical(
        ρ::InfinitePEPO, operator,
        i::CartesianIndex{2}, js::AbstractVector{CartesianIndex{2}},
        env::CTMRGEnv,
    )
    rotated_ρ = rotl90(ρ)
    unitcell = size(ρ)[1:2]
    rotated_i = siterotl90(i, unitcell)
    rotated_js = map(j -> siterotl90(j, unitcell), js)
    return correlator_horizontal(
        rotated_ρ, operator, rotated_i, rotated_js, rotl90(env)
    )
end

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, js::CoordCollection{2},
        env::CTMRGEnv,
    )
    js = vec(js) # map CartesianIndices to Vector instead of Matrix
    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return correlator_horizontal(ρ, O, i, js, env)
    elseif all(==(i[2]) ∘ last ∘ Tuple, js)
        return correlator_vertical(ρ, O, i, js, env)
    else
        error("Only horizontal or vertical correlators are implemented")
    end
end

function MPSKit.correlator(
        ρ::InfinitePEPO, O,
        i::CartesianIndex{2}, j::CartesianIndex{2},
        env::CTMRGEnv,
    )
    return only(correlator(ρ, O, i, j:j, env))
end

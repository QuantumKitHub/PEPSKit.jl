function correlator_horizontal(
    bra::InfinitePEPS,
    operator,
    i::CartesianIndex{2},
    js::AbstractVector{CartesianIndex{2}},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    size(ket) == size(bra) ||
        throw(DimensionMismatch("The ket and bra must have the same unit cell."))
    all(==(i[1]) ∘ first ∘ Tuple, js) ||
        throw(ArgumentError("Not a horizontal correlation function"))
    issorted(vcat(i, js); by=last ∘ Tuple) ||
        throw(ArgumentError("Not an increasing sequence of coordinates"))

    O = FiniteMPO(operator)
    length(O) == 2 || throw(ArgumentError("Operator must act on two sites"))

    # preallocate with correct scalartype
    G = similar(
        js,
        TensorOperations.promote_contract(
            scalartype(bra), scalartype(ket), scalartype(env), scalartype.(O)...
        ),
    )

    # left start for operator and norm contractions
    Vn, Vo = start_correlator(i, bra, O[1], ket, env)
    i += CartesianIndex(0, 1)

    for (k, j) in enumerate(js)
        # transfer until left of site j
        while j > i
            Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
            Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
            sandwich = (
                ket[mod1(i[1], end), mod1(i[2], end)], bra[mod1(i[1], end), mod1(i[2], end)]
            )
            T = TransferMatrix(Atop, sandwich, _dag(Abot))
            Vo = Vo * T
            Vn = Vn * T
            i += CartesianIndex(0, 1)
        end

        # compute overlap with operator
        numerator = end_correlator_numerator(j, Vo, bra, O[2], ket, env)

        # transfer right of site j
        Atop = env.edges[NORTH, _prev(i[1], end), mod1(i[2], end)]
        Abot = env.edges[SOUTH, _next(i[1], end), mod1(i[2], end)]
        sandwich = (
            ket[mod1(i[1], end), mod1(i[2], end)], bra[mod1(i[1], end), mod1(i[2], end)]
        )
        T = TransferMatrix(Atop, sandwich, _dag(Abot))
        Vo = Vo * T
        Vn = Vn * T
        i += CartesianIndex(0, 1)

        # compute overlap without operator
        denominator = end_correlator_denominator(j, Vn, env)

        G[k] = numerator / denominator
    end

    return G
end

function start_correlator(
    i::CartesianIndex{2},
    below::InfinitePEPS,
    O::MPOTensor,
    above::InfinitePEPS,
    env::CTMRGEnv,
)
    r, c = Tuple(i)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    E_west = env.edges[WEST, mod1(r, end), _prev(c, end)]
    C_northwest = env.corners[NORTHWEST, _prev(r, end), _prev(c, end)]
    C_southwest = env.corners[SOUTHWEST, _next(r, end), _prev(c, end)]
    sandwich = (below[mod1(r, end), mod1(c, end)], above[mod1(r, end), mod1(c, end)])

    # TODO: part of these contractions is duplicated between the two output tensors,
    # so could be optimized further
    @autoopt @tensor Vn[χSE Detop Debot; χNE] :=
        E_south[χSE Dstop Dsbot; χSW2] *
        C_southwest[χSW2; χSW] *
        E_west[χSW Dwtop Dwbot; χNW] *
        C_northwest[χNW; χN] *
        conj(bra(sandwich)[d; Dnbot Debot Dsbot Dwbot]) *
        ket(sandwich)[d; Dntop Detop Dstop Dwtop] *
        E_north[χN Dntop Dnbot; χNE]

    @autoopt @tensor Vo[χSE Detop dstring Debot; χNE] :=
        E_south[χSE Dstop Dsbot; χSW2] *
        C_southwest[χSW2; χSW] *
        E_west[χSW Dwtop Dwbot; χNW] *
        C_northwest[χNW; χN] *
        conj(bra(sandwich)[d1; Dnbot Debot Dsbot Dwbot]) *
        removeunit(O, 1)[d1; d2 dstring] *
        ket(sandwich)[d2; Dntop Detop Dstop Dwtop] *
        E_north[χN Dntop Dnbot; χNE]

    return Vn, Vo
end

function end_correlator_numerator(
    j::CartesianIndex{2},
    V,
    above::InfinitePEPS,
    O::MPOTensor,
    below::InfinitePEPS,
    env::CTMRGEnv,
)
    r, c = Tuple(j)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    sandwich = (above[mod1(r, end), mod1(c, end)], below[mod1(r, end), mod1(c, end)])

    return @autoopt @tensor contractcheck = true V[χSW DWt dstring DWb; χNW] *
        E_south[χSSE DSt DSb; χSW] *
        E_east[χNEE DEt DEb; χSEE] *
        E_north[χNW DNt DNb; χNNE] *
        C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        conj(bra(sandwich)[db; DNb DEb DSb DWb]) *
        ket(sandwich)[dt; DNt DEt DSt DWt] *
        removeunit(O, 4)[dstring db; dt]
end

function end_correlator_denominator(j::CartesianIndex{2}, V, env::CTMRGEnv)
    r, c = Tuple(j)
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]

    return @autoopt @tensor V[χS DEt DEb; χN] *
        C_northeast[χN; χNE] *
        E_east[χNE DEt DEb; χSE] *
        C_southeast[χSE; χS]
end

function correlator_vertical(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    js::AbstractVector{CartesianIndex{2}},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    rotated_bra = rotl90(bra)
    rotated_ket = bra === ket ? rotated_bra : rotl90(ket)

    rotated_i = rotl90(i)
    rotated_j = map(rotl90, js)

    return correlator_horizontal(
        rotated_bra, O, rotated_i, rotated_j, rotated_ket, rotl90(env)
    )
end

const CoordCollection{N} = Union{AbstractVector{CartesianIndex{N}},CartesianIndices{N}}

function MPSKit.correlator(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    js::CoordCollection{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    js = vec(js) # map CartesianIndices to actual Vector instead of Matrix

    if all(==(i[1]) ∘ first ∘ Tuple, js)
        return correlator_horizontal(bra, O, i, js, ket, env)
    elseif all(==(i[2]) ∘ last ∘ Tuple, js)
        return correlator_vertical(bra, O, i, js, ket, env)
    else
        error("Only horizontal or vertical correlators are implemented")
    end
end

function MPSKit.correlator(
    bra::InfinitePEPS,
    O,
    i::CartesianIndex{2},
    j::CartesianIndex{2},
    ket::InfinitePEPS,
    env::CTMRGEnv,
)
    return only(correlator(bra, O, i, j:j, ket, env))
end

function MPSKit.correlator(state::InfinitePEPS, O, i::CartesianIndex{2}, j, env::CTMRGEnv)
    return MPSKit.correlator(state, O, i, j, state, env)
end

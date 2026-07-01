function start_correlator_left(
        i::CartesianIndex{2},
        below::InfinitePEPS,
        O::MPOTensor,
        above::InfinitePEPS,
        env::CTMRGEnv,
    )
    r, c = Tuple(i)
    E_north = edge(env, NORTH, r - 1, c)
    E_south = edge(env, SOUTH, r + 1, c)
    E_west = edge(env, WEST, r, c - 1)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    sandwich = (below[r, c], above[r, c])

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

function end_correlator_right_numerator(
        j::CartesianIndex{2},
        V::AbstractTensorMap{T, S, 4, 1},
        above::InfinitePEPS,
        O::MPOTensor,
        below::InfinitePEPS,
        env::CTMRGEnv,
    ) where {T, S}
    r, c = Tuple(j)
    E_north = edge(env, NORTH, r - 1, c)
    E_east = edge(env, EAST, r, c + 1)
    E_south = edge(env, SOUTH, r + 1, c)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    sandwich = (above[r, c], below[r, c])

    return @autoopt @tensor V[χSW DWt dstring DWb; χNW] *
        E_south[χSSE DSt DSb; χSW] *
        E_east[χNEE DEt DEb; χSEE] *
        E_north[χNW DNt DNb; χNNE] *
        C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        conj(bra(sandwich)[db; DNb DEb DSb DWb]) *
        ket(sandwich)[dt; DNt DEt DSt DWt] *
        removeunit(O, 4)[dstring db; dt]
end

function end_correlator_right_denominator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 3, 1},
        env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    E_east = edge(env, EAST, r, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)

    return @autoopt @tensor V[χS DEt DEb; χN] *
        C_northeast[χN; χNE] *
        E_east[χNE DEt DEb; χSE] *
        C_southeast[χSE; χS]
end

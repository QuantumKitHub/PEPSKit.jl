# transfer with excited GL
function MPSKit.transfer_left(
        GL::GenericMPSTensor{S, 4}, O::PEPSSandwich,
        A::GenericMPSTensor{S, 3}, Ā::GenericMPSTensor{S, 3},
    ) where {S}
    return @autoopt @tensor GL′[χ_SE D_E_above d_string D_E_below; χ_NE] :=
        GL[χ_SW D_W_above d_string D_W_below; χ_NW] *
        conj(Ā[χ_SW D_S_above D_S_below; χ_SE]) *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below]) *
        A[χ_NW D_N_above D_N_below; χ_NE]
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
        V::AbstractTensorMap{T, S, 4, 1},
        above::InfinitePEPS,
        O::MPOTensor,
        below::InfinitePEPS,
        env::CTMRGEnv,
    ) where {T, S}
    r, c = Tuple(j)
    E_north = env.edges[NORTH, _prev(r, end), mod1(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    E_south = env.edges[SOUTH, _next(r, end), mod1(c, end)]
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]
    sandwich = (above[mod1(r, end), mod1(c, end)], below[mod1(r, end), mod1(c, end)])

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

function end_correlator_denominator(
        j::CartesianIndex{2}, V::AbstractTensorMap{T, S, 3, 1},
        env::CTMRGEnv
    ) where {T, S}
    r, c = Tuple(j)
    C_northeast = env.corners[NORTHEAST, _prev(r, end), _next(c, end)]
    E_east = env.edges[EAST, mod1(r, end), _next(c, end)]
    C_southeast = env.corners[SOUTHEAST, _next(r, end), _next(c, end)]

    return @autoopt @tensor V[χS DEt DEb; χN] *
        C_northeast[χN; χNE] *
        E_east[χNE DEt DEb; χSE] *
        C_southeast[χSE; χS]
end

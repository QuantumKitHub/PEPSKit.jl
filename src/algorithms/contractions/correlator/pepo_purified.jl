# -------- For left-to-right correlator contraction --------

function start_correlator_left(
        i::CartesianIndex{2},
        below::InfinitePEPO,
        O::MPOTensor,
        above::InfinitePEPO,
        env::CTMRGEnv,
    )
    r, c = Tuple(i)
    E_north = edge(env, NORTH, r - 1, c)
    E_south = edge(env, SOUTH, r + 1, c)
    E_west = edge(env, WEST, r, c - 1)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    ket = twistdual(below[r, c], (1, 2))
    bra = above[r, c]

    @autoopt @tensor Vn[χSE Detop Debot; χNE] :=
        E_south[χSE Dstop Dsbot; χSW2] *
        C_southwest[χSW2; χSW] *
        E_west[χSW Dwtop Dwbot; χNW] *
        C_northwest[χNW; χN] *
        conj(bra[d a; Dnbot Debot Dsbot Dwbot]) *
        ket[d a; Dntop Detop Dstop Dwtop] *
        E_north[χN Dntop Dnbot; χNE]

    @autoopt @tensor Vo[χSE Detop dstring Debot; χNE] :=
        E_south[χSE Dstop Dsbot; χSW2] *
        C_southwest[χSW2; χSW] *
        E_west[χSW Dwtop Dwbot; χNW] *
        C_northwest[χNW; χN] *
        conj(bra[d1 a; Dnbot Debot Dsbot Dwbot]) *
        removeunit(O, 1)[d1; d2 dstring] *
        ket[d2 a; Dntop Detop Dstop Dwtop] *
        E_north[χN Dntop Dnbot; χNE]

    return Vn, Vo
end

function end_correlator_right_numerator(
        j::CartesianIndex{2},
        V::AbstractTensorMap{T, S, 4, 1},
        above::InfinitePEPO,
        O::MPOTensor,
        below::InfinitePEPO,
        env::CTMRGEnv,
    ) where {T, S}
    r, c = Tuple(j)
    E_north = edge(env, NORTH, r - 1, c)
    E_east = edge(env, EAST, r, c + 1)
    E_south = edge(env, SOUTH, r + 1, c)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    ket = twistdual(above[r, c], (1, 2))
    bra = below[r, c]

    return @autoopt @tensor V[χSW DWt dstring DWb; χNW] *
        E_south[χSSE DSt DSb; χSW] *
        E_east[χNEE DEt DEb; χSEE] *
        E_north[χNW DNt DNb; χNNE] *
        C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        conj(bra[db a; DNb DEb DSb DWb]) *
        ket[dt a; DNt DEt DSt DWt] *
        removeunit(O, 4)[dstring db; dt]
end

# -------- For right-to-left correlator contraction --------

function start_correlator_right(
        i::CartesianIndex{2},
        below::InfinitePEPO,
        O::MPOTensor,
        above::InfinitePEPO,
        env::CTMRGEnv,
    )
    r, c = Tuple(i)
    E_north = edge(env, NORTH, r - 1, c)
    E_east = edge(env, EAST, r, c + 1)
    E_south = edge(env, SOUTH, r + 1, c)
    C_northeast = corner(env, NORTHEAST, r - 1, c + 1)
    C_southeast = corner(env, SOUTHEAST, r + 1, c + 1)
    ket = twistdual(below[r, c], (1, 2))
    bra = above[r, c]

    @autoopt @tensor Vn[χNW DWtop DWbot; χSW] :=
        E_south[χSSE Dstop Dsbot; χSW] *
        E_east[χNEE Detop Debot; χSEE] *
        E_north[χNW Dntop Dnbot; χNNE] *
        C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        conj(bra[d a; Dnbot Debot Dsbot DWbot]) *
        ket[d a; Dntop Detop Dstop DWtop]

    @autoopt @tensor Vo[χNW DWtop dstring DWbot; χSW] :=
        E_south[χSSE Dstop Dsbot; χSW] *
        E_east[χNEE Detop Debot; χSEE] *
        E_north[χNW Dntop Dnbot; χNNE] *
        C_northeast[χNNE; χNEE] *
        C_southeast[χSEE; χSSE] *
        conj(bra[d1 a; Dnbot Debot Dsbot DWbot]) *
        removeunit(O, 1)[d1; d2 dstring] *
        ket[d2 a; Dntop Detop Dstop DWtop]

    return Vn, Vo
end

function end_correlator_left_numerator(
        j::CartesianIndex{2},
        V::AbstractTensorMap{T, S, 4, 1},
        above::InfinitePEPO,
        O::MPOTensor,
        below::InfinitePEPO,
        env::CTMRGEnv,
    ) where {T, S}
    r, c = Tuple(j)
    E_north = edge(env, NORTH, r - 1, c)
    E_south = edge(env, SOUTH, r + 1, c)
    E_west = edge(env, WEST, r, c - 1)
    C_northwest = corner(env, NORTHWEST, r - 1, c - 1)
    C_southwest = corner(env, SOUTHWEST, r + 1, c - 1)
    ket = twistdual(above[r, c], (1, 2))
    bra = below[r, c]

    return @autoopt @tensor V[χNE DEt dstring DEb; χSE] *
        E_south[χSE DSt DSb; χSW2] *
        C_southwest[χSW2; χSW] *
        E_west[χSW DWt DWb; χNW] *
        C_northwest[χNW; χN] *
        E_north[χN DNt DNb; χNE] *
        conj(bra[db a; DNb DEb DSb DWb]) *
        ket[dt a; DNt DEt DSt DWt] *
        removeunit(O, 4)[dstring db; dt]
end

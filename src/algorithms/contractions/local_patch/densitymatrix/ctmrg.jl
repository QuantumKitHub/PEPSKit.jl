# Fast path reduced density matrices using CTMRG environments
# -----------------------------------------------------------

# Keep contraction order but try to optimize intermediate permutations:
# EE_SWA is largest object so keep largest legs to the front there
function reduced_densitymatrix1x1(
        inds::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(inds)

    # Unpack variables and absorb corners
    A = ket[row, col]
    Ā = bra[row, col]

    E_north = absorb_right(
        edge(env, NORTH, row - 1, col), corner(env, NORTHEAST, row - 1, col + 1)
    )
    E_east = absorb_right(
        edge(env, EAST, row, col + 1), corner(env, SOUTHEAST, row + 1, col + 1)
    )
    E_south = absorb_right(
        edge(env, SOUTH, row + 1, col), corner(env, SOUTHWEST, row + 1, col - 1)
    )
    E_west = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_SW[χSE χNW DSb DWb; DSt DWt] :=
        E_south[χSE DSt DSb; χSW] * E_west[χSW DWt DWb; χNW]

    @tensor EE_SWA[χSE χNW DNt DEt; dt DSb DWb] :=
        EE_SW[χSE χNW DSb DWb; DSt DWt] * A[dt; DNt DEt DSt DWt]

    @tensor EE_NE[DNb DEb; χSE χNW DNt DEt] :=
        E_north[χNW DNt DNb; χNE] * E_east[χNE DEt DEb; χSE]

    @tensor EEAEE[dt; DNb DEb DSb DWb] :=
        EE_NE[DNb DEb; χSE χNW DNt DEt] * EE_SWA[χSE χNW DNt DEt; dt DSb DWb]

    @tensor ρ[dt; db] := EEAEE[dt; DNb DEb DSb DWb] * conj(Ā[db; DNb DEb DSb DWb])

    return ρ / str(ρ)
end

# Special case 2x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
function reduced_densitymatrix2x1(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_north = ket[row, col]
    Ā_north = bra[row, col]
    A_south = ket[row + 1, col]
    Ā_south = bra[row + 1, col]

    E_north = absorb_right(
        edge(env, NORTH, row - 1, col), corner(env, NORTHEAST, row - 1, col + 1)
    )
    E_northeast = edge(env, EAST, row, col + 1)
    E_southeast = absorb_right(
        edge(env, EAST, row + 1, col + 1), corner(env, SOUTHEAST, row + 2, col + 1)
    )
    E_south = absorb_right(
        edge(env, SOUTH, row + 2, col), corner(env, SOUTHWEST, row + 2, col - 1)
    )
    E_southwest = edge(env, WEST, row + 1, col - 1)
    E_northwest = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_NW[χW χNE DNWt DNt; DNWb DNb] :=
        E_northwest[χW DNWt DNWb; χNW] * E_north[χNW DNt DNb; χNE]
    @tensor EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] :=
        EE_NW[χW χNE DNWt DNt; DNWb DNb] * conj(Ā_north[dNb; DNb DNEb DMb DNWb])
    @tensor EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] :=
        EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] * A_north[dNt; DNt DNEt DMt DNWt]
    @tensor EEEAA_N[dNt dNb; χW DMt DMb χE] :=
        EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] * E_northeast[χNE DNEt DNEb; χE]

    @tensor EE_SE[χE χSW DSEt DSt; DSEb DSb] :=
        E_southeast[χE DSEt DSEb; χSE] * E_south[χSE DSt DSb; χSW]
    @tensor EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] :=
        EE_SE[χE χSW DSEt DSt; DSEb DSb] * conj(Ā_south[dSb; DMb DSEb DSb DSWb])
    @tensor EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] :=
        EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] * A_south[dSt; DMt DSEt DSt DSWt]
    @tensor EEEAA_S[χW DMt DMb χE; dSt dSb] :=
        EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] * E_southwest[χSW DSWt DSWb; χW]

    @tensor ρ[dNt dSt; dNb dSb] :=
        EEEAA_N[dNt dNb; χW DMt DMb χE] * EEEAA_S[χW DMt DMb χE; dSt dSb]

    return ρ / str(ρ)
end

function reduced_densitymatrix1x2(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_west = ket[row, col]
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    E_northwest = edge(env, NORTH, row - 1, col)
    E_northeast = absorb_right(
        edge(env, NORTH, row - 1, col + 1), corner(env, NORTHEAST, row - 1, col + 2)
    )
    E_east = absorb_right(
        edge(env, EAST, row, col + 2), corner(env, SOUTHEAST, row + 1, col + 2)
    )
    E_southeast = edge(env, SOUTH, row + 1, col + 1)
    E_southwest = absorb_right(
        edge(env, SOUTH, row + 1, col), corner(env, SOUTHWEST, row + 1, col - 1)
    )
    E_west = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_SW[χS χNW DSWt DWt; DSWb DWb] :=
        E_southwest[χS DSWt DSWb; χSW] * E_west[χSW DWt DWb; χNW]
    @tensor EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] :=
        EE_SW[χS χNW DSWt DWt; DSWb DWb] * conj(Ā_west[dWb; DNWb DMb DSWb DWb])
    @tensor EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] :=
        EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] * A_west[dWt; DNWt DMt DSWt DWt]
    @tensor EEEAA_W[dWt dWb; χS DMt DMb χN] :=
        EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] * E_northwest[χNW DNWt DNWb; χN]

    @tensor EE_NE[χN χSE DNEt DEt; DNEb DEb] :=
        E_northeast[χN DNEt DNEb; χNE] * E_east[χNE DEt DEb; χSE]
    @tensor EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] :=
        EE_NE[χN χSE DNEt DEt; DNEb DEb] * conj(Ā_east[dEb; DNEb DEb DSEb DMb])
    @tensor EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] :=
        EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] * A_east[dEt; DNEt DEt DSEt DMt]
    @tensor EEEAA_E[χS DMt DMb χN; dEt dEb] :=
        EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] * E_southeast[χSE DSEt DSEb; χS]

    @tensor ρ[dWt dEt; dWb dEb] :=
        EEEAA_W[dWt dWb; χS DMt DMb χN] * EEEAA_E[χS DMt DMb χN; dEt dEb]

    return ρ / str(ρ)
end

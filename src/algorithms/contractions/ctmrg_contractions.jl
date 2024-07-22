const CTMRGEdgeTensor{S} = AbstractTensorMap{S,3,1}
const CTMRGCornerTensor{S} = AbstractTensorMap{S,1,1}

# Enlarged corner contractions
# ----------------------------

"""
    enlarge_northwest_corner((row, col), envs, ket, bra)
    enlarge_northwest_corner(E_west, C_northwest, E_north, ket, bra)

Contract the enlarged northwest corner of the CTMRG environment, either by specifying the
coordinates, environments and state, or by directly providing the tensors.

```
    C_northwest -- E_north --
         |            ||
      E_west    == ket-bra ==
         |            ||
```
"""
function enlarge_northwest_corner(
    (row, col), envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    E_west = envs.edges[WEST, row, _prev(col, end)]
    C_northwest = envs.corners[NORTHWEST, _prev(row, end), _prev(col, end)]
    E_north = envs.edges[NORTH, _prev(row, end), col]
    return enlarge_northwest_corner(
        E_west, C_northwest, E_north, ket[row, col], bra[row, col]
    )
end
function enlarge_northwest_corner(
    E_west::CTMRGEdgeTensor,
    C_northwest::CTMRGCornerTensor,
    E_north::CTMRGEdgeTensor,
    ket::PEPSTensor,
    bra::PEPSTensor=ket,
)
    return @autoopt @tensor corner[χ_S D_Sabove D_Sbelow; χ_E D_Eabove D_Ebelow] :=
        E_west[χ_S D1 D2; χ1] *
        C_northwest[χ1; χ2] *
        E_north[χ2 D3 D4; χ_E] *
        ket[d; D3 D_Eabove D_Sabove D1] *
        conj(bra[d; D4 D_Ebelow D_Sbelow D2])
end

"""
    enlarge_northeast_corner((row, col), envs, ket, bra)
    enlarge_northeast_corner(E_north, C_northeast, E_east, ket, bra)

Contract the enlarged northeast corner of the CTMRG environment, either by specifying the
coordinates, environments and state, or by directly providing the tensors.

```
    -- E_north -- C_northeast
          ||            |
    == ket-bra ==    E_east
          ||            |
```
"""
function enlarge_northeast_corner(
    (row, col), envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    E_north = envs.edges[NORTH, _prev(row, end), col]
    C_northeast = envs.corners[NORTHEAST, _prev(row, end), _next(col, end)]
    E_east = envs.edges[EAST, row, _next(col, end)]
    return enlarge_northeast_corner(
        E_north, C_northeast, E_east, ket[row, col], bra[row, col]
    )
end
function enlarge_northeast_corner(
    E_north::CTMRGEdgeTensor,
    C_northeast::CTMRGCornerTensor,
    E_east::CTMRGEdgeTensor,
    ket::PEPSTensor,
    bra::PEPSTensor=ket,
)
    return @autoopt @tensor corner[χ_W D_Wabove D_Wbelow; χ_S D_Sabove D_Sbelow] :=
        E_north[χ_W D1 D2; χ1] *
        C_northeast[χ1; χ2] *
        E_east[χ2 D3 D4; χ_S] *
        ket[d; D1 D3 D_Sabove D_Wabove] *
        conj(bra[d; D2 D4 D_Sbelow D_Wbelow])
end

# TODO: also bring other corners in same form
function southeast_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        env.edges[EAST, row, _next(col, end)][χ_N D1 D2; χ1] *
        env.corners[SOUTHEAST, _next(row, end), _next(col, end)][χ1; χ2] *
        env.edges[SOUTH, _next(row, end), col][χ2 D3 D4; χ_W] *
        peps_above[row, col][d; D_Nabove D1 D3 D_Wabove] *
        conj(peps_below[row, col][d; D_Nbelow D2 D4 D_Wbelow])
end
function southwest_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        env.edges[SOUTH, _next(row, end), col][χ_E D1 D2; χ1] *
        env.corners[SOUTHWEST, _next(row, end), _prev(col, end)][χ1; χ2] *
        env.edges[WEST, row, _prev(col, end)][χ2 D3 D4; χ_N] *
        peps_above[row, col][d; D_Nabove D_Eabove D1 D3] *
        conj(peps_below[row, col][d; D_Nbelow D_Ebelow D2 D4])
end

# Projector contractions
# ----------------------

function halfinfinite_environment(quadrant1::AbstractTensorMap{S,3,3}, quadrant2::AbstractTensorMap{S,3,3})
    return @autoopt @tensor half[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
            quadrant1[χ_in D_inabove D_inbelow; χ D1 D2] *
            quadrant2[χ D1 D2; χ_out D_outabove D_outbelow]
end

# Renormalization contractions
# ----------------------------

# corner normalizations are the same contractions everywhere
function renormalize_corner(quadrant, P_left, P_right)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] * quadrant[χ1 D1 D2; χ2 D3 D4] * P_left[χ2 D3 D4; χ_out]
end
function rightrenormalize_corner(C, E, P)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        E[χ_in D1 D2; χ1] * C[χ1; χ2] * P[χ2 D1 D2; χ_out]
end
function leftrenormalize_corner(C, E, P)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P[χ_in D1 D2; χ1] * C[χ1; χ2] * E[χ2 D1 D2; χ_out]
end

"""
    renormalize_west_edge((row, col), envs, P_top, P_bottom, ket, bra)
    renormalize_west_edge(E_west, P_top, P_bottom, ket, bra)

Absorb a bra-ket pair into the west edge using the given projectors and environment tensors.

```
    TODO: diagram
```
"""
function renormalize_west_edge((row, col), envs, P_top, P_bottom, ket, bra)
    return renormalize_west_edge(envs.edges[WEST, row, _prev(col, end)],
            P_top[WEST, row, col], P_bottom[WEST, _next(row, end), col],
            ket[row, col], bra[row, col])
end
function renormalize_west_edge(E_west, P_top, P_bottom, ket, bra)
    return @autoopt @tensor edge[χ_S D_Eab D_Ebe; χ_N] :=
        E_west[χ1 D1 D2; χ2] *
        ket[d; D3 D_Eab D5 D1] *
        conj(bra[d; D4 D_Ebe D6 D2]) *
        P_bottom[χ2 D3 D4; χ_N] *
        P_top[χ_S; χ1 D5 D6]
end

# TODO: docstring
function renormalize_north_edge((row, col), envs, P_left, P_right, ket, bra)
    return renormalize_north_edge(envs.edges[NORTH, _prev(row, end), col],
            P_left[NORTH, row, col], P_right[NORTH, row, _prev(col, end)],
            ket[row, col], bra[row, col])
end
function renormalize_north_edge(E_north, P_left, P_right, ket, bra)
    return @autoopt @tensor edge[χ_W D_Eab D_Ebe; χ_E] :=
        E_north[χ1 D1 D2; χ2] *
        ket[d; D3 D_Eab D5 D1] *
        conj(bra[d; D4 D_Ebe D6 D2]) *
        P_right[χ2 D3 D4; χ_E] *
        P_left[χ_W; χ1 D5 D6]
end

# TODO: add other contractions

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

"""
    enlarge_southeast_corner((row, col), envs, ket, bra)
    enlarge_southeast_corner(E_east, C_southeast, E_south, ket, bra)

Contract the enlarged southeast corner of the CTMRG environment, either by specifying the
coordinates, environments and state, or by directly providing the tensors.

```
          ||            |
    == ket-bra ==    E_east
          ||            |
    -- E_south -- C_southeast
```
"""
function enlarge_southeast_corner(
    (row, col), envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    E_east = envs.edges[EAST, row, _next(col, end)]
    C_southeast = envs.corners[SOUTHEAST, _next(row, end), _next(col, end)]
    E_south = envs.edges[SOUTH, _next(row, end), col]
    return enlarge_southeast_corner(
        E_east, C_southeast, E_south, ket[row, col], bra[row, col]
    )
end
function enlarge_southeast_corner(
    E_east::CTMRGEdgeTensor,
    C_southeast::CTMRGCornerTensor,
    E_south::CTMRGEdgeTensor,
    ket::PEPSTensor,
    bra::PEPSTensor=ket,
)
    return @autoopt @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        E_east[χ_N D1 D2; χ1] *
        C_southeast[χ1; χ2] *
        E_south[χ2 D3 D4; χ_W] *
        ket[d; D_Nabove D1 D3 D_Wabove] *
        conj(bra[d; D_Nbelow D2 D4 D_Wbelow])
end

"""
    enlarge_southwest_corner((row, col), envs, ket, bra)
    enlarge_southwest_corner(E_south, C_southwest, E_west, ket, bra)

Contract the enlarged southwest corner of the CTMRG environment, either by specifying the
coordinates, environments and state, or by directly providing the tensors.

```
          |           ||      
       E_west   == ket-bra == 
          |           ||      
    C_southwest -- E_south -- 
```
"""
function enlarge_southwest_corner(
    (row, col), envs::CTMRGEnv, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    E_south = envs.edges[SOUTH, _next(row, end), col]
    C_southwest = envs.corners[SOUTHWEST, _next(row, end), _prev(col, end)]
    E_west = envs.edges[WEST, row, _prev(col, end)]
    return enlarge_southwest_corner(
        E_south, C_southwest, E_west, ket[row, col], bra[row, col]
    )
end
function enlarge_southwest_corner(
    E_south::CTMRGEdgeTensor,
    C_southwest::CTMRGCornerTensor,
    E_west::CTMRGEdgeTensor,
    ket::PEPSTensor,
    bra::PEPSTensor=ket,
)
    return @autoopt @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        E_south[χ_E D1 D2; χ1] *
        C_southwest[χ1; χ2] *
        E_west[χ2 D3 D4; χ_N] *
        ket[d; D_Nabove D_Eabove D1 D3] *
        conj(bra[d; D_Nbelow D_Ebelow D2 D4])
end

# Projector contractions
# ----------------------

"""
    halfinfinite_environment(quadrant1::AbstractTensorMap{S,3,3}, quadrant2::AbstractTensorMap{S,3,3})

Contract two quadrants (enlarged corners) to form a half-infinite environment.

```
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant1|    |quadrant2|
    |~~~~~~~~~| == |~~~~~~~~~|
      |    ||        ||    |
```
"""
function halfinfinite_environment(
    quadrant1::AbstractTensorMap{S,3,3}, quadrant2::AbstractTensorMap{S,3,3}
) where {S}
    return @autoopt @tensor half[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
        quadrant1[χ_in D_inabove D_inbelow; χ D1 D2] *
        quadrant2[χ D1 D2; χ_out D_outabove D_outbelow]
end

# Renormalization contractions
# ----------------------------

# corners

"""
    renormalize_corner(quadrant, P_left, P_right)

Apply projectors to each side of a quadrant.

```
    |~~~~~~~~| -- |~~~~~~|
    |quadrant|    |P_left| --
    |~~~~~~~~| == |~~~~~~|
     |    ||
    [P_right]
        |
```
"""
function renormalize_corner(quadrant::AbstractTensorMap{S,3,3}, P_left, P_right) where {S}
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] * quadrant[χ1 D1 D2; χ2 D3 D4] * P_left[χ2 D3 D4; χ_out]
end

"""
    renormalize_northwest_corner((row, col), enlarged_envs::CTMRGEnv, P_left, P_right)

Apply `renormalize_corner` to the enlarged northwest corner.
"""
function renormalize_northwest_corner((row, col), enlarged_envs, P_left, P_right)
    return renormalize_corner(
        enlarged_envs[NORTHWEST, row, col],
        P_left[NORTH, row, col],
        P_right[WEST, _next(row, end), col],
    )
end

"""
    renormalize_northeast_corner((row, col), enlarged_envs::CTMRGEnv, P_left, P_right)

Apply `renormalize_corner` to the enlarged northeast corner.
"""
function renormalize_northeast_corner((row, col), enlarged_envs, P_left, P_right)
    return renormalize_corner(
        enlarged_envs[NORTHEAST, row, col],
        P_left[EAST, row, col],
        P_right[NORTH, row, _prev(col, end)],
    )
end

"""
    renormalize_southeast_corner((row, col), enlarged_envs::CTMRGEnv, P_left, P_right)

Apply `renormalize_corner` to the enlarged southeast corner.
"""
function renormalize_southeast_corner((row, col), enlarged_envs, P_left, P_right)
    return renormalize_corner(
        enlarged_envs[SOUTHEAST, row, col],
        P_left[SOUTH, row, col],
        P_right[EAST, _prev(row, end), col],
    )
end

"""
    renormalize_southwest_corner((row, col), enlarged_envs::CTMRGEnv, P_left, P_right)

Apply `renormalize_corner` to the enlarged southwest corner.
"""
function renormalize_southwest_corner((row, col), enlarged_envs, P_left, P_right)
    return renormalize_corner(
        enlarged_envs[SOUTHWEST, row, col],
        P_left[WEST, row, col],
        P_right[SOUTH, row, _next(col, end)],
    )
end

"""
    renormalize_bottom_corner((r, c), envs, projectors)

Apply bottom projector to southwest corner and south edge.
```
        | 
    [P_bottom]
     |     ||
     C --  E -- in
```
"""
function renormalize_bottom_corner((row, col), envs::CTMRGEnv, projectors)
    C_southwest = envs.corners[SOUTHWEST, row, _prev(col, end)]
    E_south = envs.edges[SOUTH, row, col]
    P_bottom = projectors[1][row, col]
    return @autoopt @tensor corner[χ_in; χ_out] :=
        E_south[χ_in D1 D2; χ1] * C_southwest[χ1; χ2] * P_bottom[χ2 D1 D2; χ_out]
end

"""
    renormalize_top_corner((row, col), envs::CTMRGEnv, projectors)

Apply top projector to northwest corner and north edge.
```
     C -- E -- 
     |    ||
    [~P_top~]
        | 
```
"""
function renormalize_top_corner((row, col), envs::CTMRGEnv, projectors)
    C_northwest = envs.corners[NORTHWEST, row, _prev(col, end)]
    E_north = envs.edges[NORTH, row, col]
    P_top = projectors[2][_next(row, end), col]
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_top[χ_in; χ1 D1 D2] * C_northwest[χ1; χ2] * E_north[χ2 D1 D2; χ_out]
end

# edges

"""
    renormalize_north_edge((row, col), envs, P_left, P_right, ket, bra)
    renormalize_north_edge(E_north, P_left, P_right, ket, bra)

Absorb a bra-ket pair into the north edge using the given projectors and environment tensors.

```
       |~~~~~~| -- E_north -- |~~~~~~~| 
    -- |P_left|      ||       |P_right| --
       |~~~~~~| == ket-bra == |~~~~~~~| 

```
"""
function renormalize_north_edge(
    (row, col), envs::CTMRGEnv, P_left, P_right, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    return renormalize_north_edge(
        envs.edges[NORTH, _prev(row, end), col],
        P_left[NORTH, row, col],
        P_right[NORTH, row, _prev(col, end)],
        ket[row, col],
        bra[row, col],
    )
end
function renormalize_north_edge(
    E_north::CTMRGEdgeTensor, P_left, P_right, ket::PEPSTensor, bra::PEPSTensor=ket
)
    return @autoopt @tensor edge[χ_W D_Sab D_Sbe; χ_E] :=
        E_north[χ1 D1 D2; χ2] *
        ket[d; D1 D3 D_Sab D5] *
        conj(bra[d; D2 D4 D_Sbe D6]) *
        P_left[χ2 D3 D4; χ_E] *
        P_right[χ_W; χ1 D5 D6]
end

"""
    renormalize_east_edge((row, col), envs, P_top, P_bottom, ket, bra)
    renormalize_east_edge(E_east, P_top, P_bottom, ket, bra)

Absorb a bra-ket pair into the east edge using the given projectors and environment tensors.

```
            |
     [~~P_bottom~~]
      |         ||
    E_east == ket-bra
      |         ||
     [~~~~P_top~~~]
            |
```
"""
function renormalize_east_edge(
    (row, col), envs::CTMRGEnv, P_bottom, P_top, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    return renormalize_east_edge(
        envs.edges[EAST, row, _next(col, end)],
        P_bottom[EAST, row, col, end],
        P_top[EAST, _prev(row, end), col],
        ket[row, col],
        bra[row, col],
    )
end
function renormalize_east_edge(
    E_east::CTMRGEdgeTensor, P_bottom, P_top, ket::PEPSTensor, bra::PEPSTensor=ket
)
    return @autoopt @tensor edge[χ_N D_Wab D_Wbe; χ_S] :=
        E_east[χ1 D1 D2; χ2] *
        ket[d; D5 D1 D3 D_Wab] *
        conj(bra[d; D6 D2 D4 D_Wbe]) *
        P_bottom[χ2 D3 D4; χ_S] *
        P_top[χ_N; χ1 D5 D6]
end

"""
    renormalize_south_edge((row, col), envs, P_left, P_right, ket, bra)
    renormalize_south_edge(E_south, P_left, P_right, ket, bra)

Absorb a bra-ket pair into the south edge using the given projectors and environment tensors.

```
       |~~~~~~~| == ket-bra == |~~~~~~| 
    -- |P_right|      ||       |P_left| --
       |~~~~~~~| -- E_south -- |~~~~~~| 

```
"""
function renormalize_south_edge(
    (row, col), envs::CTMRGEnv, P_left, P_right, ket::InfinitePEPS, bra::InfinitePEPS=ket
)
    return renormalize_south_edge(
        envs.edges[SOUTH, _next(row, end), col],
        P_left[SOUTH, row, col],
        P_right[SOUTH, row, _next(col, end)],
        ket[row, col],
        bra[row, col],
    )
end
function renormalize_south_edge(
    E_south::CTMRGEdgeTensor, P_left, P_right, ket::PEPSTensor, bra::PEPSTensor=ket
)
    return @autoopt @tensor edge[χ_E D_Nab D_Nbe; χ_W] :=
        E_south[χ1 D1 D2; χ2] *
        bra[d; D_Nab D5 D1 D3] *
        conj(ket[d; D_Nbe D6 D2 D4]) *
        P_left[χ2 D3 D4; χ_W] *
        P_right[χ_E; χ1 D5 D6]
end

"""
    renormalize_west_edge((row, col), envs, P_top, P_bottom, ket, bra)
    renormalize_west_edge(E_west, P_top, P_bottom, ket, bra)

Absorb a bra-ket pair into the west edge using the given projectors and environment tensors.

```
            |
     [~~P_bottom~~]
      |         ||
    E_west == ket-bra
      |         ||
     [~~~~P_top~~~]
            |
```
"""
function renormalize_west_edge(  # For simultaneous CTMRG scheme
    (row, col),
    envs::CTMRGEnv,
    P_bottom::Array{Pb,3},
    P_top::Array{Pt,3},
    ket::InfinitePEPS,
    bra::InfinitePEPS=ket,
) where {Pt,Pb}
    return renormalize_west_edge(
        envs.edges[WEST, row, _prev(col, end)],
        P_bottom[WEST, row, col],
        P_top[WEST, _next(row, end), col],
        ket[row, col],
        bra[row, col],
    )
end
function renormalize_west_edge(  # For sequential CTMRG scheme
    (row, col),
    envs::CTMRGEnv,
    projectors,
    ket::InfinitePEPS,
    bra::InfinitePEPS=ket,
)
    return renormalize_west_edge(
        envs.edges[WEST, row, _prev(col, end)],
        projectors[1][row, col],
        projectors[2][_next(row, end), col],
        ket[row, col],
        bra[row, col],
    )
end
function renormalize_west_edge(
    E_west::CTMRGEdgeTensor, P_bottom, P_top, ket::PEPSTensor, bra::PEPSTensor=ket
)
    return @autoopt @tensor edge[χ_S D_Eab D_Ebe; χ_N] :=
        E_west[χ1 D1 D2; χ2] *
        ket[d; D3 D_Eab D5 D1] *
        conj(bra[d; D4 D_Ebe D6 D2]) *
        P_bottom[χ2 D3 D4; χ_N] *
        P_top[χ_S; χ1 D5 D6]
end

# Gauge fixing contractions
# -------------------------

# corners 

"""
    fix_gauge_corner(corner, σ_in, σ_out)

Multiply corner tensor with incoming and outgoing gauge signs.

```
    corner -- σ_out --
      |  
     σ_in
      |
```
"""
function fix_gauge_corner(
    corner::CTMRGCornerTensor, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
)
    @autoopt @tensor corner_fix[χ_in; χ_out] :=
        σ_in[χ_in; χ1] * corner[χ1; χ2] * conj(σ_out[χ_out; χ2])
end

"""
    fix_gauge_northwest_corner((row, col), envs, signs)

Apply `fix_gauge_corner` to the northwest corner with appropriate row and column indices.
"""
function fix_gauge_northwest_corner((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_corner(
        envs.corners[NORTHWEST, row, col],
        signs[WEST, row, col],
        signs[NORTH, row, _next(col, end)],
    )
end

"""
    fix_gauge_northeast_corner((row, col), envs, signs)

Apply `fix_gauge_corner` to the northeast corner with appropriate row and column indices.
"""
function fix_gauge_northeast_corner((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_corner(
        envs.corners[NORTHEAST, row, col],
        signs[NORTH, row, col],
        signs[EAST, _next(row, end), col],
    )
end

"""
    fix_gauge_southeast_corner((row, col), envs, signs)

Apply `fix_gauge_corner` to the southeast corner with appropriate row and column indices.
"""
function fix_gauge_southeast_corner((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_corner(
        envs.corners[SOUTHEAST, row, col],
        signs[EAST, row, col],
        signs[SOUTH, row, _prev(col, end)],
    )
end

"""
    fix_gauge_southwest_corner((row, col), envs, signs)

Apply `fix_gauge_corner` to the southwest corner with appropriate row and column indices.
"""
function fix_gauge_southwest_corner((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_corner(
        envs.corners[SOUTHWEST, row, col],
        signs[SOUTH, row, col],
        signs[WEST, _prev(row, end), col],
    )
end

# edges

"""
    fix_gauge_edge(edge, σ_in, σ_out)

Multiply edge tensor with incoming and outgoing gauge signs.

```
    -- σ_in -- edge -- σ_out --
```
"""
function fix_gauge_edge(
    edge::CTMRGEdgeTensor, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
)
    @autoopt @tensor edge_fix[χ_in D_above D_below; χ_out] :=
        σ_in[χ_in; χ1] * edge[χ1 D_above D_below; χ2] * conj(σ_out[χ_out; χ2])
end

"""
    fix_gauge_north_edge((row, col), envs, signs)

Apply `fix_gauge_edge` to the north edge with appropriate row and column indices.
"""
function fix_gauge_north_edge((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_edge(
        envs.edges[NORTH, row, col],
        signs[NORTH, row, col],
        signs[NORTH, row, _next(col, end)],
    )
end

"""
    fix_gauge_east_edge((row, col), envs, signs)

Apply `fix_gauge_edge` to the east edge with appropriate row and column indices.
"""
function fix_gauge_east_edge((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_edge(
        envs.edges[EAST, row, col], signs[EAST, row, col], signs[EAST, _next(row, end), col]
    )
end

"""
    fix_gauge_south_edge((row, col), envs, signs)

Apply `fix_gauge_edge` to the south edge with appropriate row and column indices.
"""
function fix_gauge_south_edge((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_edge(
        envs.edges[SOUTH, row, col],
        signs[SOUTH, row, col],
        signs[SOUTH, row, _prev(col, end)],
    )
end

"""
    fix_gauge_south_edge((row, col), envs, signs)

Apply `fix_gauge_edge` to the west edge with appropriate row and column indices.
"""
function fix_gauge_west_edge((row, col), envs::CTMRGEnv, signs)
    return fix_gauge_edge(
        envs.edges[WEST, row, col], signs[WEST, row, col], signs[WEST, _prev(row, end), col]
    )
end

# left singular vectors

"""
    fix_gauge_north_left_vecs((row, col), U, signs)

Multiply north left singular vectors with gauge signs from the right.
"""
function fix_gauge_north_left_vecs((row, col), U, signs)
    return U[NORTH, row, col] * signs[NORTH, row, _next(col, end)]
end

"""
    fix_gauge_east_left_vecs((row, col), U, signs)

Multiply east left singular vectors with gauge signs from the right.
"""
function fix_gauge_east_left_vecs((row, col), U, signs)
    return U[EAST, row, col] * signs[EAST, _next(row, end), col]
end

"""
    fix_gauge_south_left_vecs((row, col), U, signs)

Multiply south left singular vectors with gauge signs from the right.
"""
function fix_gauge_south_left_vecs((row, col), U, signs)
    return U[SOUTH, row, col] * signs[SOUTH, row, _prev(col, end)]
end

"""
    fix_gauge_west_left_vecs((row, col), U, signs)

Multiply west left singular vectors with gauge signs from the right.
"""
function fix_gauge_west_left_vecs((row, col), U, signs)
    return U[WEST, row, col] * signs[WEST, _prev(row, end), col]
end

# right singular vectors

"""
    fix_gauge_north_right_vecs((row, col), V, signs)

Multiply north right singular vectors with gauge signs from the left.
"""
function fix_gauge_north_right_vecs((row, col), V, signs)
    return signs[NORTH, row, _next(col, end)]' * V[NORTH, row, col]
end

"""
    fix_gauge_east_right_vecs((row, col), V, signs)

Multiply east right singular vectors with gauge signs from the left.
"""
function fix_gauge_east_right_vecs((row, col), V, signs)
    return signs[EAST, _next(row, end), col]' * V[EAST, row, col]
end

"""
    fix_gauge_south_right_vecs((row, col), V, signs)

Multiply south right singular vectors with gauge signs from the left.
"""
function fix_gauge_south_right_vecs((row, col), V, signs)
    return signs[SOUTH, row, _prev(col, end)]' * V[SOUTH, row, col]
end

"""
    fix_gauge_west((row, col), V, signs)

Multiply west right singular vectors with gauge signs from the left.
"""
function fix_gauge_west_right_vecs((row, col), V, signs)
    return signs[WEST, _prev(row, end), col]' * V[WEST, row, col]
end

# Gauge fixing contractions
# -------------------------

# corners

"""
$(SIGNATURES)

Multiply corner tensor with incoming and outgoing gauge signs.

```
    corner -- σ_in --
      |
     σ_out
      |
```
"""
function fix_gauge_corner(
        corner::CTMRGCornerTensor, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
    )
    return @autoopt @tensor corner_fix[χ_out; χ_in] :=
        σ_in[χ_out; χ1] * corner[χ1; χ2] * conj(σ_out[χ_in; χ2])
end

"""
$(SIGNATURES)

Apply `fix_gauge_corner` to the northwest corner with appropriate row and column indices.
"""
function fix_gauge_northwest_corner((row, col), env::CTMRGEnv, signs)
    return fix_gauge_corner(
        env.corners[NORTHWEST, row, col],
        signs[WEST, row, col],
        signs[NORTH, row, _next(col, end)],
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_corner` to the northeast corner with appropriate row and column indices.
"""
function fix_gauge_northeast_corner((row, col), env::CTMRGEnv, signs)
    return fix_gauge_corner(
        env.corners[NORTHEAST, row, col],
        signs[NORTH, row, col],
        signs[EAST, _next(row, end), col],
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_corner` to the southeast corner with appropriate row and column indices.
"""
function fix_gauge_southeast_corner((row, col), env::CTMRGEnv, signs)
    return fix_gauge_corner(
        env.corners[SOUTHEAST, row, col],
        signs[EAST, row, col],
        signs[SOUTH, row, _prev(col, end)],
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_corner` to the southwest corner with appropriate row and column indices.
"""
function fix_gauge_southwest_corner((row, col), env::CTMRGEnv, signs)
    return fix_gauge_corner(
        env.corners[SOUTHWEST, row, col],
        signs[SOUTH, row, col],
        signs[WEST, _prev(row, end), col],
    )
end

# edges

"""
$(SIGNATURES)

Multiply edge tensor with incoming and outgoing gauge signs.

```
    -- σ_out -- edge -- σ_in --
```
"""
@generated function fix_gauge_edge(
        edge::CTMRGEdgeTensor{T, S, N}, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
    ) where {T, S, N}
    edge_fix_e = tensorexpr(
        :edge_fix,
        (envlabel(:out), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:in),),
    )
    edge_e = tensorexpr(
        :edge, (envlabel(1), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(2),)
    )
    σ_out_e = tensorexpr(:σ_out, (envlabel(:out),), (envlabel(1),))
    σ_in_e = tensorexpr(:σ_in, (envlabel(:in),), (envlabel(2),))
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $edge_fix_e := $σ_out_e * $edge_e * conj($σ_in_e)),
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_edge` to the north edge with appropriate row and column indices.
"""
function fix_gauge_north_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[NORTH, row, col], signs[NORTH, row, col], signs[NORTH, row, _next(col, end)],
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_edge` to the east edge with appropriate row and column indices.
"""
function fix_gauge_east_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[EAST, row, col], signs[EAST, row, col], signs[EAST, _next(row, end), col]
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_edge` to the south edge with appropriate row and column indices.
"""
function fix_gauge_south_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[SOUTH, row, col], signs[SOUTH, row, col], signs[SOUTH, row, _prev(col, end)],
    )
end

"""
$(SIGNATURES)

Apply `fix_gauge_edge` to the west edge with appropriate row and column indices.
"""
function fix_gauge_west_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[WEST, row, col], signs[WEST, row, col], signs[WEST, _prev(row, end), col]
    )
end

# left singular vectors

"""
$(SIGNATURES)

Multiply north left singular vectors with gauge signs from the right.
"""
function fix_gauge_north_left_vecs((row, col), U, signs)
    return U[NORTH, row, col] * signs[NORTH, row, _next(col, end)]'
end

"""
$(SIGNATURES)

Multiply east left singular vectors with gauge signs from the right.
"""
function fix_gauge_east_left_vecs((row, col), U, signs)
    return U[EAST, row, col] * signs[EAST, _next(row, end), col]'
end

"""
$(SIGNATURES)

Multiply south left singular vectors with gauge signs from the right.
"""
function fix_gauge_south_left_vecs((row, col), U, signs)
    return U[SOUTH, row, col] * signs[SOUTH, row, _prev(col, end)]'
end

"""
$(SIGNATURES)

Multiply west left singular vectors with gauge signs from the right.
"""
function fix_gauge_west_left_vecs((row, col), U, signs)
    return U[WEST, row, col] * signs[WEST, _prev(row, end), col]'
end

# right singular vectors

"""
$(SIGNATURES)

Multiply north right singular vectors with gauge signs from the left.
"""
function fix_gauge_north_right_vecs((row, col), V, signs)
    return signs[NORTH, row, _next(col, end)] * V[NORTH, row, col]
end

"""
$(SIGNATURES)

Multiply east right singular vectors with gauge signs from the left.
"""
function fix_gauge_east_right_vecs((row, col), V, signs)
    return signs[EAST, _next(row, end), col] * V[EAST, row, col]
end

"""
$(SIGNATURES)

Multiply south right singular vectors with gauge signs from the left.
"""
function fix_gauge_south_right_vecs((row, col), V, signs)
    return signs[SOUTH, row, _prev(col, end)] * V[SOUTH, row, col]
end

"""
$(SIGNATURES)

Multiply west right singular vectors with gauge signs from the left.
"""
function fix_gauge_west_right_vecs((row, col), V, signs)
    return signs[WEST, _prev(row, end), col] * V[WEST, row, col]
end

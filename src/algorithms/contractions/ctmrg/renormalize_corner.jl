"""
$(SIGNATURES)

Apply projectors to each side of a quadrant.

```
    |~~~~~~~~| -- |~~~~~~|
    |quadrant|    |P_left| --
    |~~~~~~~~| -- |~~~~~~|
     |     |
    [P_right]
        |
```
"""
@generated function renormalize_corner(
        quadrant::AbstractTensorMap{<:Any, S, N, N},
        P_left::AbstractTensorMap{<:Any, S, N, 1},
        P_right::AbstractTensorMap{<:Any, S, 1, N},
    ) where {S, N}
    corner_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    P_right_e = tensorexpr(
        :P_right,
        (envlabel(:out),),
        (envlabel(:L), ntuple(i -> virtuallabel(:L, i), N - 1)...),
    )
    P_left_e = tensorexpr(
        :P_left,
        (envlabel(:R), ntuple(i -> virtuallabel(:R, i), N - 1)...),
        (envlabel(:in),),
    )
    quadrant_e = tensorexpr(
        :quadrant,
        (envlabel(:L), ntuple(i -> virtuallabel(:L, i), N - 1)...),
        (envlabel(:R), ntuple(i -> virtuallabel(:R, i), N - 1)...),
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $corner_e := $P_right_e * $quadrant_e * $P_left_e),
    )
end

# Northwest corner
# ----------------

"""
    renormalize_northwest_corner((row, col), enlarged_env, P_left, P_right)
    renormalize_northwest_corner(quadrant, P_left, P_right)
    renormalize_northwest_corner(E_west, C_northwest, E_north, P_left, P_right, A)

Apply `renormalize_corner` to the enlarged northwest corner.

```
    |~~~~~~~~| -- |~~~~~~|
    |quadrant|    |P_left| --
    |~~~~~~~~| -- |~~~~~~|
     |     |
    [P_right]
        |
```

Alternatively, provide the constituent tensors and perform the complete contraction.

```
    C_northwest -- E_north -- |~~~~~~|
         |           |        |P_left| --
      E_west    --   A     -- |~~~~~~|
         |           |
      [~~~~~P_right~~~~]
               |
```
"""
function renormalize_northwest_corner((row, col), enlarged_env, P_left, P_right)
    return renormalize_northwest_corner(
        enlarged_env[NORTHWEST, row, col],
        P_left[NORTH, row, col],
        P_right[WEST, _next(row, end), col],
    )
end
function renormalize_northwest_corner(
        quadrant::AbstractTensorMap{T, S, N, N}, P_left, P_right
    ) where {T, S, N}
    return renormalize_corner(quadrant, P_left, P_right)
end
function renormalize_northwest_corner(
        E_west, C_northwest, E_north, P_left, P_right, A::PEPSSandwich
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1 D2] *
        E_west[χ1 D3 D4; χ2] * C_northwest[χ2; χ3] * E_north[χ3 D5 D6; χ4] *
        ket(A)[d; D5 D7 D1 D3] * conj(bra(A)[d; D6 D8 D2 D4]) *
        P_left[χ4 D7 D8; χ_in]
end
function renormalize_northwest_corner(
        E_west, C_northwest, E_north, P_left, P_right, A::PFTensor
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1] *
        E_west[χ1 D3; χ2] * C_northwest[χ2; χ3] * E_north[χ3 D5; χ4] *
        A[D3 D1; D5 D7] *
        P_left[χ4 D7; χ_in]
end

@generated function renormalize_northwest_corner(
        E_west, C_northwest, E_north, P_left, P_right, A::PEPOSandwich{H}
    ) where {H}
    C_out_e = _corner_expr(:corner, :out, :in)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :S, :S, H)
    E_west_e = _pepo_edge_expr(:E_west, :S, :WNW, :W, H)
    C_northwest_e = _corner_expr(:C_northwest, :WNW, :NNW)
    E_north_e = _pepo_edge_expr(:E_north, :NNW, :E, :N, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :E, :E, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_west_e, C_northwest_e, E_north_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Northeast corner
# ----------------

"""
    renormalize_northeast_corner((row, col), enlarged_env, P_left, P_right)
    renormalize_northeast_corner(quadrant, P_left, P_right)
    renormalize_northeast_corner(E_north, C_northeast, E_east, P_left, P_right, A)

Apply `renormalize_corner` to the enlarged northeast corner.

```
       |~~~~~~~| -- |~~~~~~~~|
    -- |P_right|    |quadrant|
       |~~~~~~~| -- |~~~~~~~~|
                      |    |
                     [P_left]
                         |
```

Alternatively, provide the constituent tensors and perform the complete contraction.

```
       |~~~~~~~| -- E_north -- C_northeast
    -- |P_right|       |            |
       |~~~~~~~| --    A    --    E_east
                       |            |
                     [~~~~~P_left~~~~~]
                              |
```
"""
function renormalize_northeast_corner((row, col), enlarged_env, P_left, P_right)
    return renormalize_northeast_corner(
        enlarged_env[NORTHEAST, row, col],
        P_left[EAST, row, col],
        P_right[NORTH, row, _prev(col, end)],
    )
end

function renormalize_northeast_corner(
        quadrant::AbstractTensorMap{T, S, N, N}, P_left, P_right
    ) where {T, S, N}
    return renormalize_corner(quadrant, P_left, P_right)
end

function renormalize_northeast_corner(
        E_north, C_northeast, E_east, P_left, P_right, A::PEPSSandwich
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1 D2] *
        E_north[χ1 D3 D4; χ2] * C_northeast[χ2; χ3] * E_east[χ3 D5 D6; χ4] *
        ket(A)[d; D3 D5 D7 D1] * conj(bra(A)[d; D4 D6 D8 D2]) *
        P_left[χ4 D7 D8; χ_in]
end
function renormalize_northeast_corner(
        E_north, C_northeast, E_east, P_left, P_right, A::PFTensor
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1] *
        E_north[χ1 D3; χ2] * C_northeast[χ2; χ3] * E_east[χ3 D5; χ4] *
        A[D1 D7; D3 D5] *
        P_left[χ4 D7; χ_in]
end

@generated function renormalize_northeast_corner(
        E_north, C_northeast, E_east, P_left, P_right, A::PEPOSandwich{H}
    ) where {H}
    C_out_e = _corner_expr(:corner, :out, :in)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :W, :W, H)
    E_north_e = _pepo_edge_expr(:E_north, :W, :NNE, :N, H)
    C_northeast_e = _corner_expr(:C_northeast, :NNE, :ENE)
    E_east_e = _pepo_edge_expr(:E_east, :ENE, :S, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :S, :S, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_north_e, C_northeast_e, E_east_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Southeast corner
# ----------------

"""
    renormalize_southeast_corner((row, col), enlarged_env, P_left, P_right)
    renormalize_southeast_corner(quadrant, P_left, P_right)
    renormalize_southeast_corner(E_east, C_southeast, E_south, P_left, P_right, A)

Apply `renormalize_corner` to the enlarged southeast corner.

```
                        |
                    [P_right]
                      |   |
       |~~~~~~| -- |~~~~~~~~|
    -- |P_left|    |quadrant|
       |~~~~~~| -- |~~~~~~~~|
```

Alternatively, provide the constituent tensors and perform the complete contraction.

```
                            |
                    [~~~~P_right~~~~]
                      |           |
       |~~~~~~| --    A    --   E_east
    -- |P_left|       |           |
       |~~~~~~| -- E_south -- C_southeast
```
"""
function renormalize_southeast_corner((row, col), enlarged_env, P_left, P_right)
    return renormalize_southeast_corner(
        enlarged_env[SOUTHEAST, row, col],
        P_left[SOUTH, row, col],
        P_right[EAST, _prev(row, end), col],
    )
end
function renormalize_southeast_corner(
        quadrant::AbstractTensorMap{T, S, N, N}, P_left, P_right
    ) where {T, S, N}
    return renormalize_corner(quadrant, P_left, P_right)
end
function renormalize_southeast_corner(
        E_east, C_southeast, E_south, P_left, P_right, A::PEPSSandwich
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1 D2] *
        E_east[χ1 D3 D4; χ2] * C_southeast[χ2; χ3] * E_south[χ3 D5 D6; χ4] *
        ket(A)[d; D1 D3 D5 D7] * conj(bra(A)[d; D2 D4 D6 D8]) *
        P_left[χ4 D7 D8; χ_in]
end
function renormalize_southeast_corner(
        E_east, C_southeast, E_south, P_left, P_right, A::PFTensor
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1] *
        E_east[χ1 D3; χ2] * C_southeast[χ2; χ3] * E_south[χ3 D5; χ4] *
        A[D7 D5; D1 D3] *
        P_left[χ4 D7; χ_in]
end

@generated function renormalize_southeast_corner(
        E_east, C_southeast, E_south, P_left, P_right, A::PEPOSandwich{H}
    ) where {H}
    C_out_e = _corner_expr(:corner, :out, :in)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :N, :N, H)
    E_east_e = _pepo_edge_expr(:E_east, :N, :EWE, :E, H)
    C_southeast_e = _corner_expr(:C_southeast, :ESE, :SSE)
    E_south_e = _pepo_edge_expr(:E_south, :SSE, :W, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :W, :W, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_east_e, C_southeast_e, E_south_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Southwest corner
# ----------------

"""
    renormalize_southwest_corner((row, col), enlarged_env, P_left, P_right)
    renormalize_southwest_corner(quadrant, P_left, P_right)
    renormalize_southwest_corner(E_south, C_southwest, E_west, P_left, P_right, A)

Apply `renormalize_corner` to the enlarged southwest corner.

```
         |
     [P_left]
      |    |
    |~~~~~~~~| -- |~~~~~~~|
    |quadrant|    |P_right| --
    |~~~~~~~~| -- |~~~~~~~|
```

Alternatively, provide the constituent tensors and perform the complete contraction.

```
               |
       [~~~~~P_left~~~~~]
         |            |
       E_west   --    A    -- |~~~~~~~|
         |            |       |P_right| --
    C_southwest -- E_south -- |~~~~~~~|
```
"""
function renormalize_southwest_corner((row, col), enlarged_env, P_left, P_right)
    return renormalize_corner(
        enlarged_env[SOUTHWEST, row, col],
        P_left[WEST, row, col],
        P_right[SOUTH, row, _next(col, end)],
    )
end
function renormalize_southwest_corner(
        quadrant::AbstractTensorMap{T, S, N, N}, P_left, P_right
    ) where {T, S, N}
    return renormalize_southwest_corner(quadrant, P_left, P_right)
end
function renormalize_southwest_corner(
        E_south, C_southwest, E_west, P_left, P_right, A::PEPSSandwich
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1 D2] *
        E_south[χ1 D3 D4; χ2] * C_southwest[χ2; χ3] * E_west[χ3 D5 D6; χ4] *
        ket(A)[d; D7 D1 D3 D5] * conj(bra(A)[d; D8 D2 D4 D6]) *
        P_left[χ4 D7 D8; χ_in]
end
function renormalize_southwest_corner(
        E_south, C_southwest, E_west, P_left, P_right, A::PFTensor
    )
    return @autoopt @tensor corner[χ_out; χ_in] :=
        P_right[χ_out; χ1 D1] *
        E_south[χ1 D3; χ2] * C_southwest[χ2; χ3] * E_west[χ3 D5; χ4] *
        A[D5 D3; D7 D1] *
        P_left[χ4 D7; χ_in]
end

@generated function renormalize_southwest_corner(
        E_south, C_southwest, E_west, P_left, P_right, A::PEPOSandwich{H}
    ) where {H}
    C_out_e = _corner_expr(:corner, :out, :in)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :E, :E, H)
    E_south_e = _pepo_edge_expr(:E_south, :E, :SSW, :S, H)
    C_southwest_e = _corner_expr(:C_southwest, :SSW, :WSW)
    E_west_e = _pepo_edge_expr(:E_west, :WSW, :N, :W, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :N, :N, :in, H)

    rhs = Expr(
        :call, :*,
        P_right_e,
        E_south_e, C_southwest_e, E_west_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# For SequentialCTMRG left move only
# ----------------------------------

"""
    renormalize_bottom_corner((r, c), env, projectors)
    renormalize_bottom_corner(C_southwest, E_south, P_left)

Apply left projector to southwest corner and south edge.
```
        |
    [P_left]
     |    |
     C -- E -- in
```
"""
function renormalize_bottom_corner((row, col), env::CTMRGEnv, projectors)
    C_southwest = env.corners[SOUTHWEST, row, _prev(col, end)]
    E_south = env.edges[SOUTH, row, col]
    P_left = projectors[1][row]
    return renormalize_bottom_corner(C_southwest, E_south, P_left)
end
@generated function renormalize_bottom_corner(
        C_southwest::CTMRGCornerTensor{<:Any, S},
        E_south::CTMRGEdgeTensor{<:Any, S, N},
        P_left::AbstractTensorMap{<:Any, S, N, 1},
    ) where {S, N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SSW),), (envlabel(:WSW),))
    E_south_e = tensorexpr(
        :E_south,
        (envlabel(:out), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:SSW),),
    )
    P_left_e = tensorexpr(
        :P_left,
        (envlabel(:WSW), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:in),),
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_south_e * $C_southwest_e * $P_left_e),
    )
end

"""
    renormalize_top_corner((row, col), env, projectors)
    renormalize_top_corner(C_northwest, E_north, P_right)

Apply right projector to northwest corner and north edge.
```
     C --- E --
     |     |
    [P_right]
        |
```
"""
function renormalize_top_corner((row, col), env::CTMRGEnv, projectors)
    C_northwest = env.corners[NORTHWEST, row, _prev(col, end)]
    E_north = env.edges[NORTH, row, col]
    P_right = projectors[2][_next(row, end)]
    return renormalize_top_corner(C_northwest, E_north, P_right)
end
@generated function renormalize_top_corner(
        C_northwest::CTMRGCornerTensor{<:Any, S},
        E_north::CTMRGEdgeTensor{<:Any, S, N},
        P_right::AbstractTensorMap{<:Any, S, 1, N},
    ) where {S, N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:WNW),), (envlabel(:NNW),))
    E_north_e = tensorexpr(
        :E_north, (envlabel(:NNW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:in),)
    )
    P_right_e = tensorexpr(
        :P_right, (envlabel(:out),), (envlabel(:WNW), ntuple(i -> virtuallabel(i), N - 1)...)
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_north_e * $C_northwest_e * $P_right_e),
    )
end

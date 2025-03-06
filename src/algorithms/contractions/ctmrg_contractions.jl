const CTMRGEdgeTensor{T,S,N} = AbstractTensorMap{T,S,N,1}
const CTMRG_PEPS_EdgeTensor{T,S} = CTMRGEdgeTensor{T,S,3}
const CTMRG_PF_EdgeTensor{T,S} = CTMRGEdgeTensor{T,S,2}
const CTMRGCornerTensor{T,S} = AbstractTensorMap{T,S,1,1}

# Enlarged corner contractions
# ----------------------------

"""
    enlarge_northwest_corner((row, col), env, network::InfiniteSquareNetwork{O})
    enlarge_northwest_corner(E_west, C_northwest, E_north, A::O)

Contract the enlarged northwest corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
    C_northwest -- E_north --
         |            |
      E_west    --    A    --
         |            |
```
"""
function enlarge_northwest_corner((row, col), env::CTMRGEnv, network::InfiniteSquareNetwork)
    E_west = env.edges[WEST, row, _prev(col, end)]
    C_northwest = env.corners[NORTHWEST, _prev(row, end), _prev(col, end)]
    E_north = env.edges[NORTH, _prev(row, end), col]
    return enlarge_northwest_corner(E_west, C_northwest, E_north, network[row, col])
end
function enlarge_northwest_corner(
    E_west::CTMRG_PEPS_EdgeTensor,
    C_northwest::CTMRGCornerTensor,
    E_north::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @autoopt @tensor corner[χ_S D_Sabove D_Sbelow; χ_E D_Eabove D_Ebelow] :=
        E_west[χ_S D1 D2; χ1] *
        C_northwest[χ1; χ2] *
        E_north[χ2 D3 D4; χ_E] *
        ket(A)[d; D3 D_Eabove D_Sabove D1] *
        conj(bra(A)[d; D4 D_Ebelow D_Sbelow D2])
end
function enlarge_northwest_corner(
    E_west::CTMRG_PF_EdgeTensor,
    C_northwest::CTMRGCornerTensor,
    E_north::CTMRG_PF_EdgeTensor,
    A::PFTensor,
)
    return @autoopt @tensor corner[χ_S D_S; χ_E D_E] :=
        E_west[χ_S D1; χ1] * C_northwest[χ1; χ2] * E_north[χ2 D2; χ_E] * A[D1 D_S; D2 D_E]
end

"""
    enlarge_northeast_corner((row, col), env, network::InfiniteSquareNetwork{O})
    enlarge_northeast_corner(E_north, C_northeast, E_east, A::O)

Contract the enlarged northeast corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
    -- E_north -- C_northeast
          |             |
    --    A    --    E_east
          |             |
```
"""
function enlarge_northeast_corner((row, col), env::CTMRGEnv, network::InfiniteSquareNetwork)
    E_north = env.edges[NORTH, _prev(row, end), col]
    C_northeast = env.corners[NORTHEAST, _prev(row, end), _next(col, end)]
    E_east = env.edges[EAST, row, _next(col, end)]
    return enlarge_northeast_corner(E_north, C_northeast, E_east, network[row, col])
end
function enlarge_northeast_corner(
    E_north::CTMRG_PEPS_EdgeTensor,
    C_northeast::CTMRGCornerTensor,
    E_east::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @autoopt @tensor corner[χ_W D_Wabove D_Wbelow; χ_S D_Sabove D_Sbelow] :=
        E_north[χ_W D1 D2; χ1] *
        C_northeast[χ1; χ2] *
        E_east[χ2 D3 D4; χ_S] *
        ket(A)[d; D1 D3 D_Sabove D_Wabove] *
        conj(bra(A)[d; D2 D4 D_Sbelow D_Wbelow])
end
function enlarge_northeast_corner(
    E_north::CTMRG_PF_EdgeTensor,
    C_northeast::CTMRGCornerTensor,
    E_east::CTMRG_PF_EdgeTensor,
    A::PFTensor,
)
    return @autoopt @tensor corner[χ_W D_W; χ_S D_S] :=
        E_north[χ_W D1; χ1] * C_northeast[χ1; χ2] * E_east[χ2 D2; χ_S] * A[D_W D_S; D1 D2]
end

"""
    enlarge_southeast_corner((row, col), env, network::InfiniteSquareNetwork{O})
    enlarge_southeast_corner(E_east, C_southeast, E_south, A::O)

Contract the enlarged southeast corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
          |             |
    --    A    --    E_east
          |             |
    -- E_south -- C_southeast
```
"""
function enlarge_southeast_corner((row, col), env::CTMRGEnv, network::InfiniteSquareNetwork)
    E_east = env.edges[EAST, row, _next(col, end)]
    C_southeast = env.corners[SOUTHEAST, _next(row, end), _next(col, end)]
    E_south = env.edges[SOUTH, _next(row, end), col]
    return enlarge_southeast_corner(E_east, C_southeast, E_south, network[row, col])
end
function enlarge_southeast_corner(
    E_east::CTMRG_PEPS_EdgeTensor,
    C_southeast::CTMRGCornerTensor,
    E_south::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @autoopt @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        E_east[χ_N D1 D2; χ1] *
        C_southeast[χ1; χ2] *
        E_south[χ2 D3 D4; χ_W] *
        ket(A)[d; D_Nabove D1 D3 D_Wabove] *
        conj(bra(A)[d; D_Nbelow D2 D4 D_Wbelow])
end
function enlarge_southeast_corner(
    E_east::CTMRG_PF_EdgeTensor,
    C_southeast::CTMRGCornerTensor,
    E_south::CTMRG_PF_EdgeTensor,
    A::PFTensor,
)
    return @autoopt @tensor corner[χ_N D_N; χ_W D_W] :=
        E_east[χ_N D1; χ1] * C_southeast[χ1; χ2] * E_south[χ2 D2; χ_W] * A[D_W D2; D_N D1]
end

"""
    enlarge_southwest_corner((row, col), env, network::InfiniteSquareNetwork{O})
    enlarge_southwest_corner(E_south, C_southwest, E_west, A::O)

Contract the enlarged southwest corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
          |           |       
       E_west   --    A    --
          |           |       
    C_southwest -- E_south -- 
```
"""
function enlarge_southwest_corner((row, col), env::CTMRGEnv, network::InfiniteSquareNetwork)
    E_south = env.edges[SOUTH, _next(row, end), col]
    C_southwest = env.corners[SOUTHWEST, _next(row, end), _prev(col, end)]
    E_west = env.edges[WEST, row, _prev(col, end)]
    return enlarge_southwest_corner(
        E_south, C_southwest, E_west, ket[row, col], bra[row, col]
    )
end
function enlarge_southwest_corner(
    E_south::CTMRG_PEPS_EdgeTensor,
    C_southwest::CTMRGCornerTensor,
    E_west::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @autoopt @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        E_south[χ_E D1 D2; χ1] *
        C_southwest[χ1; χ2] *
        E_west[χ2 D3 D4; χ_N] *
        ket(A)[d; D_Nabove D_Eabove D1 D3] *
        conj(bra(A)[d; D_Nbelow D_Ebelow D2 D4])
end
function enlarge_southwest_corner(
    E_south::CTMRG_PF_EdgeTensor,
    C_southwest::CTMRGCornerTensor,
    E_west::CTMRG_PF_EdgeTensor,
    A::PFTensor,
)
    return @autoopt @tensor corner[χ_E D_E; χ_N D_N] :=
        E_south[χ_E D1; χ1] * C_southwest[χ1; χ2] * E_west[χ2 D2; χ_N] * A[D2 D1; D_N D_E]
end

# Projector contractions
# ----------------------

"""
    left_projector(E_1, C, E_2, V, isqS, A)

Contract the CTMRG left projector with the higher-dimensional subspace facing to the left.

```
     C  --  E_2    -- |~~|
     |       |        |V'| -- isqS --
    E_1 --   A     -- |~~|
     |       |
```
"""
function left_projector(E_1, C, E_2, V, isqS, A::PEPSSandwich)
    return @autoopt @tensor P_left[χ_in D_inabove D_inbelow; χ_out] :=
        E_1[χ_in D1 D2; χ1] *
        C[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket(A)[d; D3 D5 D_inabove D1] *
        conj(bra(A)[d; D4 D6 D_inbelow D2]) *
        conj(V[χ4; χ3 D5 D6]) *
        isqS[χ4; χ_out]
end
function left_projector(E_1, C, E_2, V, isqS, A::PFTensor)
    return @autoopt @tensor P_left[χ_in D_in; χ_out] :=
        E_1[χ_in D1; χ1] *
        C[χ1; χ2] *
        E_2[χ2 D2; χ3] *
        A[D1 D_in; D2 D3] *
        conj(V[χ4; χ3 D3]) *
        isqS[χ4; χ_out]
end

"""
    right_projector(E_1, C, E_2, U, isqS, A)

Contract the CTMRG right projector with the higher-dimensional subspace facing to the right.

```
               |~~| --   E_2   --  C
    -- isqS -- |U'|      |         |
               |~~| --   A     -- E_1
                         |         |
```
"""
function right_projector(E_1, C, E_2, U, isqS, A::PEPSSandwich)
    return @autoopt @tensor P_right[χ_in; χ_out D_outabove D_outbelow] :=
        isqS[χ_in; χ1] *
        conj(U[χ1; χ2 D1 D2]) *
        ket(A)[d; D3 D5 D_outabove D1] *
        conj(bra(A)[d; D4 D6 D_outbelow D2]) *
        E_2[χ2 D3 D4; χ3] *
        C[χ3; χ4] *
        E_1[χ4 D5 D6; χ_out]
end
function right_projector(E_1, C, E_2, U, isqS, A::PFTensor)
    return @autoopt @tensor P_right[χ_in; χ_out D_out] :=
        isqS[χ_in; χ1] *
        conj(U[χ1; χ2 D1]) *
        A[D1 D_out; D2 D3] *
        E_2[χ2 D2; χ3] *
        C[χ3; χ4] *
        E_1[χ4 D3; χ_out]
end

# TODO: clean up infinite_environment?
"""
    half_infinite_environment(quadrant1::AbstractTensorMap{T,S,3,3}, quadrant2::AbstractTensorMap{T,S,N,N})
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4,  A_1::P, A_2::P)
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x, A_1::P, A_2::P)
    half_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P)

Contract two quadrants (enlarged corners) to form a half-infinite environment.

```
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant1|    |quadrant2|
    |~~~~~~~~~| -- |~~~~~~~~~|
      |     |        |     |
```

The environment can also be contracted directly from all its constituent tensors.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      | 
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
```

Alternatively, contract the environment with a vector `x` acting on it

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      | 
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
                  [~~~x~~~~]
```

or contract the adjoint environment with `x`, e.g. as needed for iterative solvers.
"""
function half_infinite_environment(
    quadrant1::AbstractTensorMap{T,S,3,3}, quadrant2::AbstractTensorMap{T,S,3,3}
) where {T,S}
    return @autoopt @tensor env[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
        quadrant1[χ_in D_inabove D_inbelow; χ D1 D2] *
        quadrant2[χ D1 D2; χ_out D_outabove D_outbelow]
end

function half_infinite_environment(
    C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
) where {P<:PEPSSandwich}
    return @autoopt @tensor env[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
        E_1[χ_in D1 D2; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D9 D_inabove D1] *
        conj(bra(A_1)[d1; D4 D10 D_inbelow D2]) *
        ket(A_2)[d2; D5 D7 D_outabove D9] *
        conj(bra(A_2)[d2; D6 D8 D_outbelow D10]) *
        E_3[χ3 D5 D6; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7 D8; χ_out]
end
function half_infinite_environment(
    C_1, C_2, E_1, E_2, E_3, E_4, x::AbstractTensor{T,S,3}, A_1::P, A_2::P
) where {T,S,P<:PEPSSandwich}
    return @autoopt @tensor env_x[χ_in D_inabove D_inbelow] :=
        E_1[χ_in D1 D2; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D9 D_inabove D1] *
        conj(bra(A_1)[d1; D4 D10 D_inbelow D2]) *
        ket(A_2)[d2; D5 D7 D11 D9] *
        conj(bra(A_2)[d2; D6 D8 D12 D10]) *
        E_3[χ3 D5 D6; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7 D8; χ6] *
        x[χ6 D11 D12]
end
function half_infinite_environment(
    x::AbstractTensor{T,S,3}, C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
) where {T,S,P<:PEPSSandwich}
    return @autoopt @tensor x_env[χ_in D_inabove D_inbelow] :=
        x[χ1 D1 D2] *
        conj(E_1[χ1 D3 D4; χ2]) *
        conj(C_1[χ2; χ3]) *
        conj(E_2[χ3 D5 D6; χ4]) *
        conj(ket(A_1)[d1; D5 D11 D1 D3]) *
        bra(A_1)[d1; D6 D12 D2 D4] *
        conj(ket(A_2)[d2; D7 D9 D_inabove D11]) *
        bra(A_2)[d2; D8 D10 D_inbelow D12] *
        conj(E_3[χ4 D7 D8; χ5]) *
        conj(C_2[χ5; χ6]) *
        conj(E_4[χ6 D9 D10; χ_in])
end
function half_infinite_environment(
    quadrant1::AbstractTensorMap{T,S,2,2}, quadrant2::AbstractTensorMap{T,S,2,2}
) where {T,S}
    return @autoopt @tensor env[χ_in D_in; χ_out D_out] :=
        quadrant1[χ_in D_in; χ D1] * quadrant2[χ D1; χ_out D_out]
end
function half_infinite_environment(
    C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
) where {P<:PFTensor}
    return @autoopt @tensor env[χ_in D_in; χ_out D_out] :=
        E_1[χ_in D1; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3; χ3] *
        A_1[D1 D_in; D3 D9] *
        A_2[D9 D_out; D5 D7] *
        E_3[χ3 D5; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7; χ_out]
end
function half_infinite_environment(
    C_1, C_2, E_1, E_2, E_3, E_4, x::AbstractTensor{T,S,2}, A_1::P, A::P
) where {T,S,P<:PFTensor}
    return @autoopt @tensor env_x[χ_in D_in] :=
        E_1[χ_in D1; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3; χ3] *
        A_1[D1 D_in; D3 D9] *
        A_2[D9 D11; D5 D7] *
        E_3[χ3 D5; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7; χ6] *
        x[χ6 D11]
end
function half_infinite_environment(
    x::AbstractTensor{T,S,2}, C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
) where {T,S,P<:PFTensor}
    return @autoopt @tensor env_x[χ_in D_in] :=
        x[χ1 D1 D2] *
        conj(E_1[χ1 D3; χ2]) *
        conj(C_1[χ2; χ3]) *
        conj(E_2[χ3 D5; χ4]) *
        conj(A_1[D3 D1; D5 D11]) *
        conj(A_2[D11 D_in; D7 D9]) *
        conj(E_3[χ4 D7; χ5]) *
        conj(C_2[χ5; χ6]) *
        conj(E_4[χ6 D9; χ_in])
end

"""
    full_infinite_environment(
        quadrant1::T, quadrant2::T, quadrant3::T, quadrant4::T
    ) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,N,N}}
    function full_infinite_environment(
        half1::T, half2::T
    ) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,N,N}}
    full_infinite_environment(C_1, C_2, C_3, C_4, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
                              A_1::P, A_2::P, A_3::P, A_4::P)
    full_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x,
                              A_1::P, A_2::P, A_3::P, A_4::P)
    full_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4,
                              A_1::P, A_2::P, A_3::P, A_4::P)

Contract four quadrants (enlarged corners) to form a full-infinite environment.

```
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant1|    |quadrant2|
    |~~~~~~~~~| -- |~~~~~~~~~|
      |     |        |     |
                     |     |
      |     |        |     |
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant4|    |quadrant3|
    |~~~~~~~~~| -- |~~~~~~~~~|
```

In the same manner two halfs can be used to contract the full-infinite environment.

```
    |~~~~~~~~~~~~~~~~~~~~~~~~|
    |         half1          |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
      |     |        |     |
                     |     |
      |     |        |     |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
    |         half2          |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
```

The environment can also be contracted directly from all its constituent tensors.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      | 
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
                   |      |
     |      |      |      |
    E_8 -- A_4 -- A_3 -- E_5
     |      |      |      |
    C_4 -- E_7 -- E_6 -- C_3
```

Alternatively, contract the environment with a vector `x` acting on it

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      | 
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
                   |      |
    [~~~~x~~~]     |      |
     |      |      |      |
    E_8 -- A_4 -- A_3 -- E_5
     |      |      |      |
    C_4 -- E_7 -- E_6 -- C_3

```

or contract the adjoint environment with `x`, e.g. as needed for iterative solvers.
"""
function full_infinite_environment(
    quadrant1::T, quadrant2::T, quadrant3::T, quadrant4::T
) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,3,3}}
    return @autoopt @tensor env[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
        quadrant1[χ_in D_inabove D_inbelow; χ1 D1 D2] *
        quadrant2[χ1 D1 D2; χ2 D3 D4] *
        quadrant3[χ2 D3 D4; χ3 D5 D6] *
        quadrant4[χ3 D5 D6; χ_out D_outabove D_outbelow]
end
function full_infinite_environment(
    half1::T, half2::T
) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,3,3}}
    return half_infinite_environment(half1, half2)
end
function full_infinite_environment(
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    ket_1::P,
    bra_1::P,
    ket_2::P,
    bra_2::P,
    ket_3::P,
    bra_3::P,
    ket_4::P,
    bra_4::P,
) where {P<:PEPSTensor}
    return @autoopt @tensor env[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
        E_1[χ_in D1 D2; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket_1[d1; D3 D11 D_inabove D1] *
        conj(bra_1[d1; D4 D12 D_inbelow D2]) *
        ket_2[d2; D5 D7 D9 D11] *
        conj(bra_2[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15 D16; χ9] *
        ket_3[d3; D9 D13 D15 D17] *
        conj(bra_3[d3; D10 D14 D16 D18]) *
        ket_4[d4; D_outabove D17 D19 D21] *
        conj(bra_4[d4; D_outbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21 D22; χ_out]
end
function full_infinite_environment(
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    x::AbstractTensor{T,S,3},
    A_1::P,
    A_2::P,
    A_3::P,
    A_4::P,
) where {T,S,P<:PEPSSandwich}
    return @autoopt @tensor env_x[χ_in D_inabove D_inbelow] :=
        E_1[χ_in D1 D2; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D11 D_inabove D1] *
        conj(bra(A_1)[d1; D4 D12 D_inbelow D2]) *
        ket(A_2)[d2; D5 D7 D9 D11] *
        conj(bra(A_2)[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15 D16; χ9] *
        ket(A_3)[d3; D9 D13 D15 D17] *
        conj(bra_3[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_xabove D17 D19 D21] *
        conj(bra(A_4)[d4; D_xbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21 D22; χ_x] *
        x[χ_x D_xabove D_xbelow]
end
function full_infinite_environment(
    x::AbstractTensor{T,S,3},
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    A_1::P,
    A_2::P,
    A_3::P,
    A_4::P,
) where {T,S,P<:PEPSSandwich}
    return @autoopt @tensor x_env[χ_in D_inabove D_inbelow] :=
        x[χ_x D_xabove D_xbelow] *
        E_1[χ_x D1 D2; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D11 D_xabove D1] *
        conj(bra(A_1)[d1; D4 D12 D_xbelow D2]) *
        ket(A_2)[d2; D5 D7 D9 D11] *
        conj(bra(A_2)[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15 D16; χ9] *
        ket(A_3)[d3; D9 D13 D15 D17] *
        conj(bra(A_3)[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_inabove D17 D19 D21] *
        conj(bra(A_4)[d4; D_inbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21 D22; χ_in]
end
function full_infinite_environment(
    quadrant1::T, quadrant2::T, quadrant3::T, quadrant4::T
) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,2,2}}
    return @autoopt @tensor env[χ_in D_inabove; χ_out D_outabove] :=
        quadrant1[χ_in D_inabove; χ1 D1] *
        quadrant2[χ1 D1; χ2 D2] *
        quadrant3[χ2 D2; χ3 D3] *
        quadrant4[χ3 D3; χ_out D_outabove]
end
function full_infinite_environment(
    half1::T, half2::T
) where {T<:AbstractTensorMap{<:Number,<:ElementarySpace,2,2}}
    return half_infinite_environment(half1, half2)
end
function full_infinite_environment(
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    A_1::P,
    A_2::P,
    A_3::P,
    A_4::P,
) where {P<:PFTensor}
    return @autoopt @tensor env[χ_in D_in; χ_out D_out] :=
        E_1[χ_in D1; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3; χ3] *
        A_1[D1 D_in; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_out D17] *
        E_7[χ9 D19; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21; χ_out]
end
function full_infinite_environment(
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    x::AbstractTensor{T,S,2},
    A_1::P,
    A_2::P,
    A_3::P,
    A_4::P,
) where {T,S,P<:PFTensor}
    return @autoopt @tensor env_x[χ_in D_in] :=
        E_1[χ_in D1; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3; χ3] *
        A_1[D1 D_in; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_x D17] *
        E_7[χ9 D19; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21; χ_x] *
        x[χ_x D_x]
end
function full_infinite_environment(
    x::AbstractTensor{T,S,2},
    C_1,
    C_2,
    C_3,
    C_4,
    E_1,
    E_2,
    E_3,
    E_4,
    E_5,
    E_6,
    E_7,
    E_8,
    A_1::P,
    A_2::P,
    A_3::P,
    A_4::P,
) where {T,S,P<:PFTensor}
    return @autoopt @tensor x_env[χ_in D_in] :=
        x[χ_x D_x] *
        E_1[χ_x D1; χ1] *
        C_1[χ1; χ2] *
        E_2[χ2 D3; χ3] *
        A_1[D1 D_x; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] *
        C_2[χ4; χ5] *
        E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] *
        C_3[χ7; χ8] *
        E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_in D17] *
        E_7[χ9 D19; χ10] *
        C_4[χ10; χ11] *
        E_8[χ11 D21; χ_in]
end

# Renormalization contractions
# ----------------------------

# corners

"""
    renormalize_corner(quadrant::AbstractTensorMap{T,S,3,3}, P_left, P_right)
    renormalize_corner(quadrant::AbstractTensorMap{T,S,2,2}, P_left, P_right)

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
function renormalize_corner(
    quadrant::AbstractTensorMap{T,S,3,3}, P_left, P_right
) where {T,S}
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] * quadrant[χ1 D1 D2; χ2 D3 D4] * P_left[χ2 D3 D4; χ_out]
end
function renormalize_corner(
    quadrant::AbstractTensorMap{T,S,2,2}, P_left, P_right
) where {T,S}
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1] * quadrant[χ1 D1; χ2 D3] * P_left[χ2 D3; χ_out]
end

"""
    renormalize_northwest_corner((row, col), enlarged_env::CTMRGEnv, P_left, P_right)
    renormalize_northwest_corner(quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right) where {T,S,N}
    renormalize_northwest_corner(E_west, C_northwest, E_north, P_left, P_right, A::O)

Apply `renormalize_corner` to the enlarged northwest corner.
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
    quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right
) where {T,S,N}
    return renormalize_corner(quadrant, P_left, P_right)
end
function renormalize_northwest_corner(
    E_west, C_northwest, E_north, P_left, P_right, A::PEPSSandwich
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] *
        E_west[χ1 D3 D4; χ2] *
        C_northwest[χ2; χ3] *
        E_north[χ3 D5 D6; χ4] *
        ket(A)[d; D5 D7 D1 D3] *
        conj(bra(A)[d; D6 D8 D2 D4]) *
        P_left[χ4 D7 D8; χ_out]
end
function renormalize_northwest_corner(
    E_west, C_northwest, E_north, P_left, P_right, A::PFTensor
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1] *
        E_west[χ1 D3; χ2] *
        C_northwest[χ2; χ3] *
        E_north[χ3 D5; χ4] *
        A[D3 D1; D5 D7] *
        P_left[χ4 D7; χ_out]
end

"""
    renormalize_northeast_corner((row, col), enlarged_env::CTMRGEnv, P_left, P_right)
    renormalize_northwest_corner(quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right) where {T,S,N}
    renormalize_northeast_corner(E_north, C_northeast, E_east, P_left, P_right, A::O)

Apply `renormalize_corner` to the enlarged northeast corner.
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
    quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right
) where {T,S,N}
    return renormalize_corner(quadrant, P_left, P_right)
end

function renormalize_northeast_corner(
    E_north, C_northeast, E_east, P_left, P_right, A::PEPSSandwich
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] *
        E_north[χ1 D3 D4; χ2] *
        C_northeast[χ2; χ3] *
        E_east[χ3 D5 D6; χ4] *
        ket(A)[d; D3 D5 D7 D1] *
        conj(bra(A)[d; D4 D6 D8 D2]) *
        P_left[χ4 D7 D8; χ_out]
end
function renormalize_northeast_corner(
    E_north, C_northeast, E_east, P_left, P_right, A::PFTensor
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1] *
        E_north[χ1 D3; χ2] *
        C_northeast[χ2; χ3] *
        E_east[χ3 D5; χ4] *
        A[D1 D7; D3 D5] *
        P_left[χ4 D7; χ_out]
end

"""
    renormalize_southeast_corner((row, col), enlarged_env::CTMRGEnv, P_left, P_right)
    renormalize_southeast_corner(quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right) where {T,S,N}
    renormalize_southeast_corner(E_east, C_southeast, E_south, P_left, P_right, A::O)

Apply `renormalize_corner` to the enlarged southeast corner.
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
    quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right
) where {T,S,N}
    return renormalize_corner(quadrant, P_left, P_right)
end
function renormalize_southeast_corner(
    E_east, C_southeast, E_south, P_left, P_right, A::PEPOSandwich
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] *
        E_east[χ1 D3 D4; χ2] *
        C_southeast[χ2; χ3] *
        E_south[χ3 D5 D6; χ4] *
        ket(A)[d; D1 D3 D5 D7] *
        conj(bra(A)[d; D2 D4 D6 D8]) *
        P_left[χ4 D7 D8; χ_out]
end
function renormalize_southeast_corner(
    E_east, C_southeast, E_south, P_left, P_right, A::PFTensor
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1] *
        E_east[χ1 D3; χ2] *
        C_southeast[χ2; χ3] *
        E_south[χ3 D5; χ4] *
        A[D7 D5; D1 D3] *
        P_left[χ4 D7; χ_out]
end

"""
    renormalize_southwest_corner((row, col), enlarged_env::CTMRGEnv, P_left, P_right)
    renormalize_southwest_corner(quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right) where {T,S,N}
    renormalize_southwest_corner(E_south, C_southwest, E_west, P_left, P_right, A::O)

Apply `renormalize_corner` to the enlarged southwest corner.
Alternatively, provide the constituent tensors and perform the complete contraction.

```
               |
       [~~~~P_right~~~~~]
         |            |
       E_west   --    A    -- |~~~~~~|
         |            |       |P_left| --
    C_southwest -- E_south -- |~~~~~~|
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
    quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right
) where {T,S,N}
    return renormalize_southwest_corner(quadrant, P_left, P_right)
end
function renormalize_southwest_corner(
    E_south, C_southwest, E_west, P_left, P_right, A::PEPSSandwich
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] *
        E_south[χ1 D3 D4; χ2] *
        C_southwest[χ2; χ3] *
        E_west[χ3 D5 D6; χ4] *
        ket(A)[d; D7 D1 D3 D5] *
        conj(bra(A)[d; D8 D2 D4 D6]) *
        P_left[χ4 D7 D8; χ_out]
end
function renormalize_southwest_corner(
    E_south, C_southwest, E_west, P_left, P_right, A::PFTensor
)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1] *
        E_south[χ1 D3; χ2] *
        C_southwest[χ2; χ3] *
        E_west[χ3 D5; χ4] *
        A[D5 D3; D7 D1] *
        P_left[χ4 D7; χ_out]
end

"""
    renormalize_bottom_corner((r, c), env, projectors)

Apply bottom projector to southwest corner and south edge.
```
        | 
    [P_bottom]
     |     |
     C --  E -- in
```
"""
function renormalize_bottom_corner(
    (row, col), env::CTMRGEnv{C,<:CTMRG_PEPS_EdgeTensor}, projectors
) where {C}
    C_southwest = env.corners[SOUTHWEST, row, _prev(col, end)]
    E_south = env.edges[SOUTH, row, col]
    P_bottom = projectors[1][row]
    return @autoopt @tensor corner[χ_in; χ_out] :=
        E_south[χ_in D1 D2; χ1] * C_southwest[χ1; χ2] * P_bottom[χ2 D1 D2; χ_out]
end
function renormalize_bottom_corner(
    (row, col), env::CTMRGEnv{C,<:CTMRG_PF_EdgeTensor}, projectors
) where {C}
    C_southwest = env.corners[SOUTHWEST, row, _prev(col, end)]
    E_south = env.edges[SOUTH, row, col]
    P_bottom = projectors[1][row]
    return @autoopt @tensor corner[χ_in; χ_out] :=
        E_south[χ_in D1; χ1] * C_southwest[χ1; χ2] * P_bottom[χ2 D1; χ_out]
end

"""
    renormalize_top_corner((row, col), env::CTMRGEnv, projectors)

Apply top projector to northwest corner and north edge.
```
     C -- E -- 
     |    |
    [~P_top~]
        | 
```
"""
function renormalize_top_corner((row, col), env::CTMRGEnv, projectors)
    C_northwest = env.corners[NORTHWEST, row, _prev(col, end)]
    E_north = env.edges[NORTH, row, col]
    P_top = projectors[2][_next(row, end)]
    return renormalize_top_corner(C_northwest, E_north, P_top)
end
function renormalize_top_corner(C_northwest, E_north::CTMRG_PEPS_EdgeTensor, P_top)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_top[χ_in; χ1 D1 D2] * C_northwest[χ1; χ2] * E_north[χ2 D1 D2; χ_out]
end
function renormalize_top_corner(C_northwest, E_north::CTMRG_PF_EdgeTensor, P_top)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_top[χ_in; χ1 D1] * C_northwest[χ1; χ2] * E_north[χ2 D1; χ_out]
end

# edges

"""
    renormalize_north_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_north_edge(E_north, P_left, P_right, A::P)

Absorb a local effective tensor `A` into the north edge using the given projectors and
environment tensors.

```
       |~~~~~~| -- E_north -- |~~~~~~~| 
    -- |P_left|       |       |P_right| --
       |~~~~~~| --    A    -- |~~~~~~~| 
                      |
```
"""
function renormalize_north_edge(
    (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork
)
    return renormalize_north_edge(
        env.edges[NORTH, _prev(row, end), col],
        P_left[NORTH, row, col],
        P_right[NORTH, row, _prev(col, end)],
        network[row, col], # so here it's fine
    )
end
function renormalize_north_edge(
    E_north::CTMRG_PEPS_EdgeTensor, P_left, P_right, A::PEPSSandwich
)
    out = @autoopt @tensor edge[χ_W D_Sab D_Sbe; χ_E] :=
        E_north[χ1 D1 D2; χ2] *
        ket(A)[d; D1 D3 D_Sab D5] *
        conj(bra(A)[d; D2 D4 D_Sbe D6]) *
        P_left[χ2 D3 D4; χ_E] *
        P_right[χ_W; χ1 D5 D6]
    return out
end
function renormalize_north_edge(E_north::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @autoopt @tensor edge[χ_W D_S; χ_E] :=
        E_north[χ1 D1; χ2] * A[D5 D_S; D1 D3] * P_left[χ2 D3; χ_E] * P_right[χ_W; χ1 D5]
end

"""
    renormalize_east_edge((row, col), env, P_top, P_bottom, network::InfiniteSquareNetwork{P})
    renormalize_east_edge(E_east, P_top, P_bottom, A::P)

Absorb a blocal effective tensor into the east edge using the given projectors and
environment tensors.

```
           |
     [~P_bottom~]
      |        |
    E_east --  A -- 
      |        |
     [~~P_top~~~]
           |
```
"""
function renormalize_east_edge(
    (row, col), env::CTMRGEnv, P_bottom, P_top, network::InfiniteSquareNetwork
)
    return renormalize_east_edge(
        env.edges[EAST, row, _next(col, end)],
        P_bottom[EAST, row, col, end],
        P_top[EAST, _prev(row, end), col],
        network[row, col],
    )
end
function renormalize_east_edge(
    E_east::CTMRG_PEPS_EdgeTensor, P_bottom, P_top, A::PEPSSandwich
)
    return @autoopt @tensor edge[χ_N D_Wab D_Wbe; χ_S] :=
        E_east[χ1 D1 D2; χ2] *
        ket(A)[d; D5 D1 D3 D_Wab] *
        conj(bra(A)[d; D6 D2 D4 D_Wbe]) *
        P_bottom[χ2 D3 D4; χ_S] *
        P_top[χ_N; χ1 D5 D6]
end
function renormalize_east_edge(E_east::CTMRG_PF_EdgeTensor, P_bottom, P_top, A::PFTensor)
    return @autoopt @tensor edge[χ_N D_W; χ_S] :=
        E_east[χ1 D1; χ2] * A[D_W D3; D5 D1] * P_bottom[χ2 D3; χ_S] * P_top[χ_N; χ1 D5]
end

"""
    renormalize_south_edge((row, col), env, P_left, P_right, network::InfiniteSquareNetwork{P})
    renormalize_south_edge(E_south, P_left, P_right, A::P)

Absorb a local effective tensor into the south edge using the given projectors and
environment tensors.

```
                       |
       |~~~~~~~| --    A    -- |~~~~~~| 
    -- |P_right|       |       |P_left| --
       |~~~~~~~| -- E_south -- |~~~~~~| 
                       |
```
"""
function renormalize_south_edge(
    (row, col), env::CTMRGEnv, P_left, P_right, network::InfiniteSquareNetwork
)
    return renormalize_south_edge(
        env.edges[SOUTH, _next(row, end), col],
        P_left[SOUTH, row, col],
        P_right[SOUTH, row, _next(col, end)],
        network[row, col],
    )
end
function renormalize_south_edge(
    E_south::CTMRG_PEPS_EdgeTensor, P_left, P_right, A::PEPSSandwich
)
    return @autoopt @tensor edge[χ_E D_Nab D_Nbe; χ_W] :=
        E_south[χ1 D1 D2; χ2] *
        ket(A)[d; D_Nab D5 D1 D3] *
        conj(bra(A)[d; D_Nbe D6 D2 D4]) *
        P_left[χ2 D3 D4; χ_W] *
        P_right[χ_E; χ1 D5 D6]
end
function renormalize_south_edge(E_south::CTMRG_PF_EdgeTensor, P_left, P_right, A::PFTensor)
    return @autoopt @tensor edge[χ_E D_N; χ_W] :=
        E_south[χ1 D1; χ2] * A[D3 D1; D_N D5] * P_left[χ2 D3; χ_W] * P_right[χ_E; χ1 D5]
end

"""
    renormalize_west_edge((row, col), env, P_top, P_bottom, network::InfiniteSquareNetwork{P})
    renormalize_west_edge(E_west, P_top, P_bottom, A::P)

Absorb a local effective tensor into the west edge using the given projectors and
environment tensors.

```
           |
     [~P_bottom~]
      |        |
   -- A  --  E_west
      |        |
     [~~P_top~~~]
           |
```
"""
function renormalize_west_edge(  # For simultaneous CTMRG scheme
    (row, col),
    env::CTMRGEnv,
    P_bottom::Array{Pb,3},
    P_top::Array{Pt,3},
    network::InfiniteSquareNetwork,
) where {Pt,Pb}
    return renormalize_west_edge(
        env.edges[WEST, row, _prev(col, end)],
        P_bottom[WEST, row, col],
        P_top[WEST, _next(row, end), col],
        network[row, col],
    )
end
function renormalize_west_edge(  # For sequential CTMRG scheme
    (row, col),
    env::CTMRGEnv,
    projectors,
    network::InfiniteSquareNetwork,
)
    return renormalize_west_edge(
        env.edges[WEST, row, _prev(col, end)],
        projectors[1][row],
        projectors[2][_next(row, end)],
        network[row, col],
    )
end
function renormalize_west_edge(
    E_west::CTMRG_PEPS_EdgeTensor, P_bottom, P_top, A::PEPSSandwich
)
    return @autoopt @tensor edge[χ_S D_Eab D_Ebe; χ_N] :=
        E_west[χ1 D1 D2; χ2] *
        ket(A)[d; D3 D_Eab D5 D1] *
        conj(bra(A)[d; D4 D_Ebe D6 D2]) *
        P_bottom[χ2 D3 D4; χ_N] *
        P_top[χ_S; χ1 D5 D6]
end
function renormalize_west_edge(E_west::CTMRG_PF_EdgeTensor, P_bottom, P_top, A::PFTensor)
    return @autoopt @tensor edge[χ_S D_E; χ_N] :=
        E_west[χ1 D1; χ2] * A[D1 D5; D3 D_E] * P_bottom[χ2 D3; χ_N] * P_top[χ_S; χ1 D5]
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
    fix_gauge_northwest_corner((row, col), env, signs)

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
    fix_gauge_northeast_corner((row, col), env, signs)

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
    fix_gauge_southeast_corner((row, col), env, signs)

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
    fix_gauge_southwest_corner((row, col), env, signs)

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
    fix_gauge_edge(edge, σ_in, σ_out)

Multiply edge tensor with incoming and outgoing gauge signs.

```
    -- σ_in -- edge -- σ_out --
```
"""
function fix_gauge_edge(
    edge::CTMRG_PEPS_EdgeTensor, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
)
    @autoopt @tensor edge_fix[χ_in D_above D_below; χ_out] :=
        σ_in[χ_in; χ1] * edge[χ1 D_above D_below; χ2] * conj(σ_out[χ_out; χ2])
end
function fix_gauge_edge(
    edge::CTMRG_PF_EdgeTensor, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
)
    @autoopt @tensor edge_fix[χ_in D_above D_below; χ_out] :=
        σ_in[χ_in; χ1] * edge[χ1 D_above D_below; χ2] * conj(σ_out[χ_out; χ2])
end

"""
    fix_gauge_north_edge((row, col), env, signs)

Apply `fix_gauge_edge` to the north edge with appropriate row and column indices.
"""
function fix_gauge_north_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[NORTH, row, col],
        signs[NORTH, row, col],
        signs[NORTH, row, _next(col, end)],
    )
end

"""
    fix_gauge_east_edge((row, col), env, signs)

Apply `fix_gauge_edge` to the east edge with appropriate row and column indices.
"""
function fix_gauge_east_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[EAST, row, col], signs[EAST, row, col], signs[EAST, _next(row, end), col]
    )
end

"""
    fix_gauge_south_edge((row, col), env, signs)

Apply `fix_gauge_edge` to the south edge with appropriate row and column indices.
"""
function fix_gauge_south_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[SOUTH, row, col],
        signs[SOUTH, row, col],
        signs[SOUTH, row, _prev(col, end)],
    )
end

"""
    fix_gauge_south_edge((row, col), env, signs)

Apply `fix_gauge_edge` to the west edge with appropriate row and column indices.
"""
function fix_gauge_west_edge((row, col), env::CTMRGEnv, signs)
    return fix_gauge_edge(
        env.edges[WEST, row, col], signs[WEST, row, col], signs[WEST, _prev(row, end), col]
    )
end

# left singular vectors

"""
    fix_gauge_north_left_vecs((row, col), U, signs)

Multiply north left singular vectors with gauge signs from the right.
"""
function fix_gauge_north_left_vecs((row, col), U, signs)
    return U[NORTH, row, col] * signs[NORTH, row, _next(col, end)]'
end

"""
    fix_gauge_east_left_vecs((row, col), U, signs)

Multiply east left singular vectors with gauge signs from the right.
"""
function fix_gauge_east_left_vecs((row, col), U, signs)
    return U[EAST, row, col] * signs[EAST, _next(row, end), col]'
end

"""
    fix_gauge_south_left_vecs((row, col), U, signs)

Multiply south left singular vectors with gauge signs from the right.
"""
function fix_gauge_south_left_vecs((row, col), U, signs)
    return U[SOUTH, row, col] * signs[SOUTH, row, _prev(col, end)]'
end

"""
    fix_gauge_west_left_vecs((row, col), U, signs)

Multiply west left singular vectors with gauge signs from the right.
"""
function fix_gauge_west_left_vecs((row, col), U, signs)
    return U[WEST, row, col] * signs[WEST, _prev(row, end), col]'
end

# right singular vectors

"""
    fix_gauge_north_right_vecs((row, col), V, signs)

Multiply north right singular vectors with gauge signs from the left.
"""
function fix_gauge_north_right_vecs((row, col), V, signs)
    return signs[NORTH, row, _next(col, end)] * V[NORTH, row, col]
end

"""
    fix_gauge_east_right_vecs((row, col), V, signs)

Multiply east right singular vectors with gauge signs from the left.
"""
function fix_gauge_east_right_vecs((row, col), V, signs)
    return signs[EAST, _next(row, end), col] * V[EAST, row, col]
end

"""
    fix_gauge_south_right_vecs((row, col), V, signs)

Multiply south right singular vectors with gauge signs from the left.
"""
function fix_gauge_south_right_vecs((row, col), V, signs)
    return signs[SOUTH, row, _prev(col, end)] * V[SOUTH, row, col]
end

"""
    fix_gauge_west((row, col), V, signs)

Multiply west right singular vectors with gauge signs from the left.
"""
function fix_gauge_west_right_vecs((row, col), V, signs)
    return signs[WEST, _prev(row, end), col] * V[WEST, row, col]
end

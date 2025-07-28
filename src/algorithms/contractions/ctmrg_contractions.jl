const CTMRGEdgeTensor{T,S,N} = AbstractTensorMap{T,S,N,1}
const CTMRG_PEPS_EdgeTensor{T,S} = CTMRGEdgeTensor{T,S,3}
const CTMRG_PF_EdgeTensor{T,S} = CTMRGEdgeTensor{T,S,2}
const CTMRGCornerTensor{T,S} = AbstractTensorMap{T,S,1,1}

# Enlarged corner contractions
# ----------------------------

#=
These contractions are hand-optimized by the following heuristics:

1. ensure contraction order gives minimal scaling in χ = D²
2. ensure dominant permutation is as efficient as possible by making large legs contiguous,
    ie moving them to the front

This second part is mostly important for dealing with non-abelian symmetries, where the
    permutations are strongly non-negligable.

For a small benchmark study:
https://gist.github.com/lkdvos/a562c2b09ef461398729ccefdab34745
=#

"""
$(SIGNATURES)

Contract the enlarged northwest corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
    C_northwest -- E_north --
         |            |
      E_west    --    A    --
         |            |
```
"""
function enlarge_northwest_corner(
    E_west::CTMRG_PEPS_EdgeTensor,
    C_northwest::CTMRGCornerTensor,
    E_north::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @tensor begin
        EC[χS DWt DWb; χ2] := E_west[χS DWt DWb; χ1] * C_northwest[χ1; χ2]

        # already putting χE in front here to make next permute cheaper
        ECE[χS χE DWb DNb; DWt DNt] := EC[χS DWt DWb; χ2] * E_north[χ2 DNt DNb; χE]

        ECEket[χS χE DEt DSt; DWb DNb d] :=
            ECE[χS χE DWb DNb; DWt DNt] * ket(A)[d; DNt DEt DSt DWt]

        corner[χS DSt DSb; χE DEt DEb] :=
            ECEket[χS χE DEt DSt; DWb DNb d] * conj(bra(A)[d; DNb DEb DSb DWb])
    end
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
$(SIGNATURES)

Contract the enlarged northeast corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
    -- E_north -- C_northeast
          |             |
    --    A    --    E_east
          |             |
```
"""
function enlarge_northeast_corner(
    E_north::CTMRG_PEPS_EdgeTensor,
    C_northeast::CTMRGCornerTensor,
    E_east::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @tensor begin
        EC[χW DNt DNb; χ2] := E_north[χW DNt DNb; χ1] * C_northeast[χ1; χ2]

        # already putting χE in front here to make next permute cheaper
        ECE[χW χS DNb DEb; DNt DEt] := EC[χW DNt DNb; χ2] * E_east[χ2 DEt DEb; χS]

        ECEket[χW χS DSt DWt; DNb DEb d] :=
            ECE[χW χS DNb DEb; DNt DEt] * ket(A)[d; DNt DEt DSt DWt]

        corner[χW DWt DWb; χS DSt DSb] :=
            ECEket[χW χS DSt DWt; DNb DEb d] * conj(bra(A)[d; DNb DEb DSb DWb])
    end
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
$(SIGNATURES)

Contract the enlarged southeast corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
          |             |
    --    A    --    E_east
          |             |
    -- E_south -- C_southeast
```
"""
function enlarge_southeast_corner(
    E_east::CTMRG_PEPS_EdgeTensor,
    C_southeast::CTMRGCornerTensor,
    E_south::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @tensor begin
        EC[χN DEt DEb; χ2] := E_east[χN DEt DEb; χ1] * C_southeast[χ1; χ2]

        # already putting χE in front here to make next permute cheaper
        ECE[χN χW DEb DSb; DEt DSt] := EC[χN DEt DEb; χ2] * E_south[χ2 DSt DSb; χW]

        ECEket[χN χW DNt DWt; DEb DSb d] :=
            ECE[χN χW DEb DSb; DEt DSt] * ket(A)[d; DNt DEt DSt DWt]

        corner[χN DNt DNb; χW DWt DWb] :=
            ECEket[χN χW DNt DWt; DEb DSb d] * conj(bra(A)[d; DNb DEb DSb DWb])
    end
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
$(SIGNATURES)

Contract the enlarged southwest corner of the CTMRG environment, either by specifying the
coordinates, environments and network, or by directly providing the tensors.

```
          |           |
       E_west   --    A    --
          |           |
    C_southwest -- E_south --
```
"""
function enlarge_southwest_corner(
    E_south::CTMRG_PEPS_EdgeTensor,
    C_southwest::CTMRGCornerTensor,
    E_west::CTMRG_PEPS_EdgeTensor,
    A::PEPSSandwich,
)
    return @tensor begin
        EC[χE DSt DSb; χ2] := E_south[χE DSt DSb; χ1] * C_southwest[χ1; χ2]

        # already putting χE in front here to make next permute cheaper
        ECE[χE χN DSb DWb; DSt DWt] := EC[χE DSt DSb; χ2] * E_west[χ2 DWt DWb; χN]

        ECEket[χE χN DNt DEt; DSb DWb d] :=
            ECE[χE χN DSb DWb; DSt DWt] * ket(A)[d; DNt DEt DSt DWt]

        corner[χE DEt DEb; χN DNt DNb] :=
            ECEket[χE χN DNt DEt; DSb DWb d] * conj(bra(A)[d; DNb DEb DSb DWb])
    end
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
$(SIGNATURES)

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
$(SIGNATURES)

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

"""
$(SIGNATURES)

Compute projectors based on a SVD of `Q * Q_next`, where the inverse square root
`isqS` of the singular values is computed.

Left projector:
```
    -- |~~~~~~| -- |~~|
       |Q_next|    |V'| -- isqS --
    == |~~~~~~| == |~~|
```

Right projector:
```
               |~~| -- |~~~| --
    -- isqS -- |U'|    | Q |
               |~~| == |~~~| ==
```
"""
function contract_projectors(U, S, V, Q, Q_next)
    isqS = sdiag_pow(S, -0.5)
    P_left = Q_next * V' * isqS  # use * to respect fermionic case
    P_right = isqS * U' * Q
    return P_left, P_right
end

"""
    half_infinite_environment(quadrant1, quadrant2)
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2)
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x, A_1, A_2)
    half_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2)

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
    quadrant1::AbstractTensorMap{T,S,N,N}, quadrant2::AbstractTensorMap{T,S,N,N}
) where {T,S,N}
    p = (codomainind(quadrant1), domainind(quadrant1))
    return tensorcontract(quadrant1, p, false, quadrant2, p, false, p)
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
    full_infinite_environment(quadrant1, quadrant2, quadrant3, quadrant4)
    full_infinite_environment(half1, half2)
    full_infinite_environment(C_1, C_2, C_3, C_4, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, A_1, A_2, A_3, A_4)
    full_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x, A_1, A_2, A_3, A_4)
    full_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2, A_3, A_4)

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
@generated function full_infinite_environment(
    quadrant1::AbstractTensorMap{T,S,N,N},
    quadrant2::AbstractTensorMap{T,S,N,N},
    quadrant3::AbstractTensorMap{T,S,N,N},
    quadrant4::AbstractTensorMap{T,S,N,N},
) where {T,S,N}
    env_e = tensorexpr(
        :env,
        (envlabel(:out), ntuple(i -> virtuallabel(:out, i), N - 1)...),
        (envlabel(:in), ntuple(i -> virtuallabel(:in, i), N - 1)...),
    )
    quadrant1_e = tensorexpr(
        :quadrant1,
        (envlabel(:out), ntuple(i -> virtuallabel(:out, i), N - 1)...),
        (envlabel(:NC), ntuple(i -> virtuallabel(:NC, i), N - 1)...),
    )
    quadrant2_e = tensorexpr(
        :quadrant2,
        (envlabel(:NC), ntuple(i -> virtuallabel(:NC, i), N - 1)...),
        (envlabel(:EC), ntuple(i -> virtuallabel(:EC, i), N - 1)...),
    )
    quadrant3_e = tensorexpr(
        :quadrant3,
        (envlabel(:EC), ntuple(i -> virtuallabel(:EC, i), N - 1)...),
        (envlabel(:SC), ntuple(i -> virtuallabel(:SC, i), N - 1)...),
    )
    quadrant4_e = tensorexpr(
        :quadrant4,
        (envlabel(:SC), ntuple(i -> virtuallabel(:SC, i), N - 1)...),
        (envlabel(:in), ntuple(i -> virtuallabel(:in, i), N - 1)...),
    )
    return macroexpand(
        @__MODULE__,
        :(
            return @autoopt @tensor $env_e :=
                $quadrant1_e * $quadrant2_e * $quadrant3_e * $quadrant4_e
        ),
    )
end
function full_infinite_environment(
    half1::AbstractTensorMap{T,S,N}, half2::AbstractTensorMap{T,S,N}
) where {T,S,N}
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
) where {P<:PEPSSandwich}
    return @autoopt @tensor env[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
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
        conj(bra(A_3)[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_outabove D17 D19 D21] *
        conj(bra(A_4)[d4; D_outbelow D18 D20 D22]) *
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
    quadrant::AbstractTensorMap{<:Any,S,N,N},
    P_left::AbstractTensorMap{<:Any,S,N,1},
    P_right::AbstractTensorMap{<:Any,S,1,N},
) where {S,N}
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
    quadrant::AbstractTensorMap{T,S,N,N}, P_left, P_right
) where {T,S,N}
    return renormalize_corner(quadrant, P_left, P_right)
end
function renormalize_southeast_corner(
    E_east, C_southeast, E_south, P_left, P_right, A::PEPSSandwich
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
    renormalize_southwest_corner((row, col), enlarged_env, P_left, P_right)
    renormalize_southwest_corner(quadrant, P_left, P_right)
    renormalize_southwest_corner(E_south, C_southwest, E_west, P_left, P_right, A)

Apply `renormalize_corner` to the enlarged southwest corner.

```
         |
     [P_left]
      |    |
    |~~~~~~~~| -- |~~~~~~|
    |quadrant|    |P_left| --
    |~~~~~~~~| -- |~~~~~~|
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
    renormalize_bottom_corner(C_southwest, E_south, P_bottom)

Apply bottom projector to southwest corner and south edge.
```
        |
    [P_bottom]
     |     |
     C --  E -- in
```
"""
function renormalize_bottom_corner((row, col), env::CTMRGEnv, projectors)
    C_southwest = env.corners[SOUTHWEST, row, _prev(col, end)]
    E_south = env.edges[SOUTH, row, col]
    P_bottom = projectors[1][row]
    return renormalize_bottom_corner(C_southwest, E_south, P_bottom)
end
@generated function renormalize_bottom_corner(
    C_southwest::CTMRGCornerTensor{<:Any,S},
    E_south::CTMRGEdgeTensor{<:Any,S,N},
    P_bottom::AbstractTensorMap{<:Any,S,N,1},
) where {S,N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SSW),), (envlabel(:WSW),))
    E_south_e = tensorexpr(
        :E_south,
        (envlabel(:out), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:SSW),),
    )
    P_bottom_e = tensorexpr(
        :P_bottom,
        (envlabel(:WSW), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:in),),
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_south_e * $C_southwest_e * $P_bottom_e),
    )
end

"""
    renormalize_top_corner((row, col), env, projectors)
    renormalize_top_corner(C_northwest, E_north, P_top)

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
@generated function renormalize_top_corner(
    C_northwest::CTMRGCornerTensor{<:Any,S},
    E_north::CTMRGEdgeTensor{<:Any,S,N},
    P_top::AbstractTensorMap{<:Any,S,1,N},
) where {S,N}
    C_out_e = tensorexpr(:corner, (envlabel(:out),), (envlabel(:in),))
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:WNW),), (envlabel(:NNW),))
    E_north_e = tensorexpr(
        :E_north, (envlabel(:NNW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:in),)
    )
    P_top_e = tensorexpr(
        :P_top, (envlabel(:out),), (envlabel(:WNW), ntuple(i -> virtuallabel(i), N - 1)...)
    )
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $C_out_e := $E_north_e * $C_northwest_e * $P_top_e),
    )
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
function renormalize_north_edge(E_north, P_left, P_right, A)
    A_west = _rotl90_localsandwich(A)
    return renormalize_west_edge(E_north, P_left, P_right, A_west)
end

"""
    renormalize_east_edge((row, col), env, P_top, P_bottom, network::InfiniteSquareNetwork{P})
    renormalize_east_edge(E_east, P_top, P_bottom, A::P)

Absorb a blocal effective tensor into the east edge using the given projectors and
environment tensors.

```
           |
     [~~P_top~~~]
      |        |
   -- A  --  E_east
      |        |
     [~P_bottom~]
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
function renormalize_east_edge(E_east, P_bottom, P_top, A)
    A_west = _rot180_localsandwich(A)
    return renormalize_west_edge(E_east, P_bottom, P_top, A_west)
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

function renormalize_south_edge(E_south, P_left, P_right, A)
    A_west = _rotr90_localsandwich(A)
    return renormalize_west_edge(E_south, P_left, P_right, A_west)
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
    E_west --  A --
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
    # starting with P_top to save one permute in the end
    return @tensor begin
        # already putting χE in front here to make next permute cheaper
        PE[χS χNW DSb DWb; DSt DWt] := P_top[χS; χSW DSt DSb] * E_west[χSW DWt DWb; χNW]

        PEket[χS χNW DNt DEt; DSb DWb d] :=
            PE[χS χNW DSb DWb; DSt DWt] * ket(A)[d; DNt DEt DSt DWt]

        corner[χS DEt DEb; χNW DNt DNb] :=
            PEket[χS χNW DNt DEt; DSb DWb d] * conj(bra(A)[d; DNb DEb DSb DWb])

        edge[χS DEt DEb; χN] := corner[χS DEt DEb; χNW DNt DNb] * P_bottom[χNW DNt DNb; χN]
    end
end
function renormalize_west_edge(E_west::CTMRG_PF_EdgeTensor, P_bottom, P_top, A::PFTensor)
    return @autoopt @tensor edge[χ_S D_E; χ_N] :=
        E_west[χ1 D1; χ2] * A[D1 D5; D3 D_E] * P_bottom[χ2 D3; χ_N] * P_top[χ_S; χ1 D5]
end

# Gauge fixing contractions
# -------------------------

# corners

"""
$(SIGNATURES)

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
    -- σ_in -- edge -- σ_out --
```
"""
@generated function fix_gauge_edge(
    edge::CTMRGEdgeTensor{T,S,N}, σ_in::CTMRGCornerTensor, σ_out::CTMRGCornerTensor
) where {T,S,N}
    edge_fix_e = tensorexpr(
        :edge_fix,
        (envlabel(:in), ntuple(i -> virtuallabel(i), N - 1)...),
        (envlabel(:out),),
    )
    edge_e = tensorexpr(
        :edge, (envlabel(1), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(2),)
    )
    σ_in_e = tensorexpr(:σ_in, (envlabel(:in),), (envlabel(1),))
    σ_out_e = tensorexpr(:σ_out, (envlabel(:out),), (envlabel(2),))
    return macroexpand(
        @__MODULE__,
        :(return @autoopt @tensor $edge_fix_e := $σ_in_e * $edge_e * conj($σ_out_e)),
    )
end

"""
$(SIGNATURES)

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
        env.edges[SOUTH, row, col],
        signs[SOUTH, row, col],
        signs[SOUTH, row, _prev(col, end)],
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

#
# Expressions
#

## PEPS tensor expressions

function _virtual_labels(dir, layer, args...; contract=nothing)
    return isnothing(contract) ? (dir, layer, args...) : (contract, layer)
end
_north_labels(args...; kwargs...) = _virtual_labels(:N, args...; kwargs...)
_east_labels(args...; kwargs...) = _virtual_labels(:E, args...; kwargs...)
_south_labels(args...; kwargs...) = _virtual_labels(:S, args...; kwargs...)
_west_labels(args...; kwargs...) = _virtual_labels(:W, args...; kwargs...)

# layer=:top for ket PEPS, layer=:bot for bra PEPS, connects to PEPO slice h
function _pepo_pepstensor_expr(
    tensorname,
    layer::Symbol,
    h::Int,
    args...;
    contract_north=nothing,
    contract_east=nothing,
    contract_south=nothing,
    contract_west=nothing,
)
    return tensorexpr(
        tensorname,
        (physicallabel(h, args...),),
        (
            virtuallabel(_north_labels(layer, args...; contract=contract_north)...),
            virtuallabel(_east_labels(layer, args...; contract=contract_east)...),
            virtuallabel(_south_labels(layer, args...; contract=contract_south)...),
            virtuallabel(_west_labels(layer, args...; contract=contract_west)...),
        ),
    )
end

# PEPO slice h
function _pepo_pepotensor_expr(
    tensorname,
    h::Int,
    args...;
    contract_north=nothing,
    contract_east=nothing,
    contract_south=nothing,
    contract_west=nothing,
)
    layer = Symbol(:mid, :_, h)
    return tensorexpr(
        tensorname,
        (physicallabel(h + 1, args...), physicallabel(h, args...)),
        (
            virtuallabel(_north_labels(layer, args...; contract=contract_north)...),
            virtuallabel(_east_labels(layer, args...; contract=contract_east)...),
            virtuallabel(_south_labels(layer, args...; contract=contract_south)...),
            virtuallabel(_west_labels(layer, args...; contract=contract_west)...),
        ),
    )
end

# PEPOSandwich
function _pepo_sandwich_expr(sandwichname, H::Int, args...; kwargs...)
    ket_e = _pepo_pepstensor_expr(:(ket($sandwichname)), :top, 1, args...; kwargs...)
    bra_e = _pepo_pepstensor_expr(:(bra($sandwichname)), :bot, H + 1, args...; kwargs...)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo($sandwichname, $h)), h, args...; kwargs...)
    end

    return ket_e, bra_e, pepo_es
end

## Corner expressions

function _corner_expr(cornername, codom_label, dom_label, args...)
    return tensorexpr(
        cornername, (envlabel(codom_label, args...),), (envlabel(dom_label, args...),)
    )
end

## Edge expressions

function _pepo_edge_expr(edgename, codom_label, dom_label, dir, H::Int, args...)
    return tensorexpr(
        edgename,
        (
            envlabel(codom_label, args...),
            virtuallabel(dir, :top, args...),
            ntuple(i -> virtuallabel(dir, :mid, i, args...), H)...,
            virtuallabel(dir, :bot, args...),
        ),
        (envlabel(dom_label, args...),),
    )
end

## Enlarged corner (quadrant) expressions

function _pepo_enlarged_corner_expr(
    cornername, codom_label, dom_label, codom_dir, dom_dir, H::Int, args...
)
    return tensorexpr(
        cornername,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, args...), H)...,
            virtuallabel(codom_dir, :bot, args...),
        ),
        (
            envlabel(dom_label, args...),
            virtuallabel(dom_dir, :top, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, args...), H)...,
            virtuallabel(dom_dir, :bot, args...),
        ),
    )
end

## Environment expressions

function _pepo_env_expr(
    envname,
    codom_label,
    dom_label,
    codom_dir,
    dom_dir,
    codom_site,
    dom_site,
    H::Int,
    args...,
)
    return tensorexpr(
        envname,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, codom_site, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, codom_site, args...), H)...,
            virtuallabel(codom_dir, :bot, codom_site, args...),
        ),
        (
            envlabel(dom_label, args...),
            virtuallabel(dom_dir, :top, dom_site, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, dom_site, args...), H)...,
            virtuallabel(dom_dir, :bot, dom_site, args...),
        ),
    )
end

function _pepo_env_arg_expr(argname, codom_label, codom_dir, codom_site, H::Int, args...)
    return tensorexpr(
        argname,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, codom_site, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, codom_site, args...), H)...,
            virtuallabel(codom_dir, :bot, codom_site, args...),
        ),
    )
end

## Projector expressions

function _pepo_codomain_projector_expr(
    projname, codom_label, dom_label, dom_dir, H::Int, args...
)
    return tensorexpr(
        projname,
        (envlabel(codom_label, args...),),
        (
            envlabel(dom_label, args...),
            virtuallabel(dom_dir, :top, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, args...), H)...,
            virtuallabel(dom_dir, :bot, args...),
        ),
    )
end

function _pepo_domain_projector_expr(
    projname, codom_label, codom_dir, dom_label, H::Int, args...
)
    return tensorexpr(
        projname,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, args...), H)...,
            virtuallabel(codom_dir, :bot, args...),
        ),
        (envlabel(dom_label, args...),),
    )
end

#
# PEPO Contractions
#

## Site contraction
@generated function _contract_site(
    C_northwest,
    C_northeast,
    C_southeast,
    C_southwest,
    E_north::CTMRGEdgeTensor{T,S,N},
    E_east::CTMRGEdgeTensor{T,S,N},
    E_south::CTMRGEdgeTensor{T,S,N},
    E_west::CTMRGEdgeTensor{T,S,N},
    O::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    C_northwest_e = _corner_expr(:C_northwest, :WNW, :NNW)
    C_northeast_e = _corner_expr(:C_northeast, :NNE, :ENE)
    C_southeast_e = _corner_expr(:C_southeast, :ESE, :SSE)
    C_southwest_e = _corner_expr(:C_southwest, :SSW, :WSW)

    E_north_e = _pepo_edge_expr(:E_north, :NNW, :NNE, :N, H)
    E_east_e = _pepo_edge_expr(:E_east, :ENE, :ESE, :E, H)
    E_south_e = _pepo_edge_expr(:E_south, :SSE, :SSW, :S, H)
    E_west_e = _pepo_edge_expr(:E_west, :WSW, :WNW, :W, H)

    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    rhs = Expr(
        :call,
        :*,
        C_northwest_e,
        C_northeast_e,
        C_southeast_e,
        C_southwest_e,
        E_north_e,
        E_east_e,
        E_south_e,
        E_west_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

## Enlarged corner contractions

@generated function enlarge_northwest_corner(
    E_west::CTMRGEdgeTensor{T,S,N},
    C_northwest::CTMRGCornerTensor,
    E_north::CTMRGEdgeTensor{T,S,N},
    O::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    E_west_e = _pepo_edge_expr(:E_west, :SW, :WNW, :W, H)
    C_northwest_e = _corner_expr(:C_northwest, :WNW, :NNW)
    E_north_e = _pepo_edge_expr(:E_north, :NNW, :NE, :N, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_northwest´, :SW, :NE, :S, :E, H)

    rhs = Expr(
        :call,
        :*,
        E_west_e,
        C_northwest_e,
        E_north_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

@generated function enlarge_northeast_corner(
    E_north::CTMRGEdgeTensor{T,S,N},
    C_northeast::CTMRGCornerTensor,
    E_east::CTMRGEdgeTensor{T,S,N},
    O::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    E_north_e = _pepo_edge_expr(:E_north, :NW, :NNE, :N, H)
    C_northeast = _corner_expr(:C_northeast, :NNE, :ENE)
    E_east_e = _pepo_edge_expr(:E_east, :ENE, :SE, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_northeast´, :NW, :SE, :W, :S, H)

    rhs = Expr(
        :call,
        :*,
        E_north_e,
        C_northeast,
        E_east_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

@generated function enlarge_southeast_corner(
    E_east::CTMRGEdgeTensor{T,S,N},
    C_southeast::CTMRGCornerTensor,
    E_south::CTMRGEdgeTensor{T,S,N},
    O::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    E_east_e = _pepo_edge_expr(:E_east, :NE, :ESE, :E, H)
    C_southeast_e = _corner_expr(:C_southeast, :ESE, :SSE)
    E_south_e = _pepo_edge_expr(:E_south, :SSE, :SW, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_southeast´, :NE, :SW, :N, :W, H)

    rhs = Expr(
        :call,
        :*,
        E_east_e,
        C_southeast_e,
        E_south_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

@generated function enlarge_southwest_corner(
    E_south::CTMRGEdgeTensor{T,S,N},
    C_southwest::CTMRGCornerTensor,
    E_west::CTMRGEdgeTensor{T,S,N},
    O::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    E_south_e = _pepo_edge_expr(:E_south, :SE, :SSW, :S, H)
    C_southwest_e = _corner_expr(:C_southwest, :SSW, :WSW)
    E_west_e = _pepo_edge_expr(:E_west, :WSW, :NW, :W, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_southwest´, :SE, :NW, :E, :N, H)

    rhs = Expr(
        :call,
        :*,
        E_south_e,
        C_southwest_e,
        E_west_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

## Projector contractions: skip, since these are somehow never used

## HalfInfiniteEnvironment contractions

# reuse partial multiplication expression; TODO: return quadrants separately?
function _half_infinite_environnment_expr(H)
    # site 1 (codomain)
    C1_e = _corner_expr(:C_1, :WNW, :NNW)
    E1_e = _pepo_edge_expr(:E_1, :SW, :WNW, :W, H, 1)
    E2_e = _pepo_edge_expr(:E_2, :NNW, :NC, :N, H, 1)
    ket1_e, bra1_e, pepo1_es = _pepo_sandwich_expr(:A_1, H, 1; contract_east=:NC)

    # site 2 (domain)
    C2_e = _corner_expr(:C_2, :NNE, :ENE)
    E3_e = _pepo_edge_expr(:E_3, :NC, :NNE, :N, H, 2)
    E4_e = _pepo_edge_expr(:E_4, :ENE, :SE, :E, H, 2)
    ket2_e, bra2_e, pepo2_es = _pepo_sandwich_expr(:A_2, H, 2; contract_west=:NC)

    partial_expr = Expr(
        :call,
        :*,
        E1_e,
        C1_e,
        E2_e,
        ket1_e,
        Expr(:call, :conj, bra1_e),
        pepo1_es...,
        E3_e,
        C2_e,
        E4_e,
        ket2_e,
        Expr(:call, :conj, bra2_e),
        pepo2_es...,
    )

    return partial_expr
end

@generated function half_infinite_environment(
    C_1, C_2, E_1, E_2, E_3, E_4, A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}
) where {H}
    # return projector expression
    env_e = _pepo_env_expr(:env, :SW, :SE, :S, :S, 1, 2, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_e := $rhs))
end
@generated function half_infinite_environment(
    C_1,
    C_2,
    E_1,
    E_2,
    E_3,
    E_4,
    x::AbstractTensor{T,S,N},
    A_1::PEPOSandwich{H},
    A_2::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    # codomain vector (output)
    env_x_e = _pepo_env_arg_expr(:env_x, :SW, :S, 1, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    # domain vector (input)
    x_e = _pepo_env_arg_expr(:x, :SE, :S, 2, H)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $env_x_e := $proj_expr * $x_e)
    )
end
@generated function half_infinite_environment(
    x::AbstractTensor{T,S,N},
    C_1,
    C_2,
    E_1,
    E_2,
    E_3,
    E_4,
    A_1::PEPOSandwich{H},
    A_2::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3
    # codomain vector (input)
    x_e = _pepo_env_arg_expr(:x, :SW, :S, 1, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    # domain vector (output)
    x_env_e = _pepo_env_arg_expr(:env_x, :SE, :S, 2, H)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $x_env_e := $x_e * $proj_expr)
    )
end

## FullInfiniteEnvironment contractions

# reuse partial multiplication expression; TODO: return quadrants separately?
function _full_infinite_environment_expr(H)
    # site 1 (codomain)
    C1_e = _corner_expr(:C_1, :WNW, :NNW)
    E1_e = _pepo_edge_expr(:E_1, :SW, :WNW, :W, H, 1)
    E2_e = _pepo_edge_expr(:E_2, :NNW, :NC, :N, H, 1)
    ket1_e, bra1_e, pepo1_es = _pepo_sandwich_expr(:A_1, H, 1; contract_east=:NC)

    # site 2
    C2_e = _corner_expr(:C_2, :NNE, :ENE)
    E3_e = _pepo_edge_expr(:E_3, :NC, :NNE, :N, H, 2)
    E4_e = _pepo_edge_expr(:E_4, :ENE, :EC, :E, H, 2)
    ket2_e, bra2_e, pepo2_es = _pepo_sandwich_expr(
        :A_2, H, 2; contract_west=:NC, contract_south=:EC
    )

    # site 3
    C3_e = _corner_expr(:C_3, :WSW, :SSW)
    E5_e = _pepo_edge_expr(:E_5, :EC, :WSW, :E, H, 3)
    E6_e = _pepo_edge_expr(:E_6, :SSW, :SC, :S, H, 3)
    ket3_e, bra3_e, pepo3_es = _pepo_sandwich_expr(
        :A_3, H, 3; contract_north=:EC, contract_west=:SC
    )

    # site 4 (domain)
    C4_e = _corner_expr(:C_4, :SSW, :WSW)
    E7_e = _pepo_edge_expr(:E_7, :SC, :SSW, :S, H, 4)
    E8_e = _pepo_edge_expr(:E_8, :WSW, :NW, :W, H, 4)
    ket4_e, bra4_e, pepo4_es = _pepo_sandwich_expr(:A_4, H, 4; contract_east=:SC)

    partial_expr = Expr(
        :call,
        :*,
        E1_e,
        C1_e,
        E2_e,
        ket1_e,
        Expr(:call, :conj, bra1_e),
        pepo1_es...,
        E3_e,
        C2_e,
        E4_e,
        ket2_e,
        Expr(:call, :conj, bra2_e),
        pepo2_es...,
        E5_e,
        C3_e,
        E6_e,
        ket3_e,
        Expr(:call, :conj, bra3_e),
        pepo3_es...,
        E7_e,
        C4_e,
        E8_e,
        ket4_e,
        Expr(:call, :conj, bra4_e),
        pepo4_es...,
    )

    return partial_expr
end

@generated function full_infinite_environment(
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
    A_1::PEPOSandwich{H},
    A_2::PEPOSandwich{H},
    A_3::PEPOSandwich{H},
    A_4::PEPOSandwich{H},
) where {H}
    # return projector expression
    env_e = _pepo_env_expr(:env, :SW, :NW, :S, :N, 1, 4, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_e := $proj_expr))
end
@generated function full_infinite_environment(
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
    x::AbstractTensor{T,S,N},
    A_1::PEPOSandwich{H},
    A_2::PEPOSandwich{H},
    A_3::PEPOSandwich{H},
    A_4::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    # codomain vecor (output)
    env_x_e = _pepo_env_arg_expr(:env, :SW, :S, 1, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    # domain vector (input)
    x_e = _pepo_env_arg_expr(:env, :NW, :N, 4, N - 1)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_x_e := $proj_expr * x_e))
end
@generated function full_infinite_environment(
    x::AbstractTensor{T,S,N},
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
    A_1::PEPOSandwich{H},
    A_2::PEPOSandwich{H},
    A_3::PEPOSandwich{H},
    A_4::PEPOSandwich{H},
) where {T,S,N,H}
    @assert N == H + 3

    # codomain vecor (input)
    x_e = _pepo_env_arg_expr(:env, :SW, :S, 1, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    # domain vector (output)
    x_env_e = _pepo_env_arg_expr(:env, :NW, :N, 4, N - 1)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $x_env_e := $x_e * $proj_expr)
    )
end

## Corner renormalization contractions

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
        :call,
        :*,
        P_right_e,
        E_west_e,
        C_northwest_e,
        E_north_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
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
        :call,
        :*,
        P_right_e,
        E_north_e,
        C_northeast_e,
        E_east_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
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
        :call,
        :*,
        P_right_e,
        E_east_e,
        C_southeast_e,
        E_south_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
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
        :call,
        :*,
        P_right_e,
        E_south_e,
        C_southwest_e,
        E_west_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

## Edge renormalization contractions

@generated function renormalize_north_edge(
    E_north::CTMRGEdgeTensor{T,S,N}, P_left, P_right, A::PEPOSandwich{H}
) where {T,S,N,H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :S, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :W, :W, H)
    E_north_e = _pepo_edge_expr(:E_north, :W, :E, :N, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :E, :E, :in, H)

    rhs = Expr(
        :call,
        :*,
        P_right_e,
        E_north_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

@generated function renormalize_east_edge(
    E_east::CTMRGEdgeTensor{T,S,N}, P_bottom, P_top, A::PEPOSandwich{H}
) where {T,S,N,H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :W, H)

    P_top_e = _pepo_codomain_projector_expr(:P_top, :out, :N, :N, H)
    E_east_e = _pepo_edge_expr(:E_east, :N, :S, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_bottom_e = _pepo_domain_projector_expr(:P_bottom, :S, :S, :in, H)

    rhs = Expr(
        :call,
        :*,
        P_top_e,
        E_east_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_bottom_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

@generated function renormalize_south_edge(
    E_south::CTMRGEdgeTensor{T,S,N}, P_left, P_right, A::PEPOSandwich{H}
) where {T,S,N,H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :N, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :E, :E, H)
    E_south_e = _pepo_edge_expr(:E_south, :E, :W, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :W, :W, :in, H)

    rhs = Expr(
        :call,
        :*,
        P_right_e,
        E_south_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_left_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

@generated function renormalize_west_edge(
    E_west::CTMRGEdgeTensor{T,S,N}, P_bottom, P_top, A::PEPOSandwich{H}
) where {T,S,N,H}
    @assert N == H + 3

    E_out_e = _pepo_edge_expr(:edge, :out, :in, :E, H)

    P_top_e = _pepo_codomain_projector_expr(:P_top, :out, :S, :S, H)
    E_west_e = _pepo_edge_expr(:E_west, :S, :N, :W, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)
    P_bottom_e = _pepo_domain_projector_expr(:P_bottom, :N, :N, :in, H)

    rhs = Expr(
        :call,
        :*,
        P_top_e,
        E_west_e,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
        P_bottom_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $E_out_e := $rhs))
end

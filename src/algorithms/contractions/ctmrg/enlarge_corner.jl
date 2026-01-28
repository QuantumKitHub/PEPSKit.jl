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

# Northwest corner
# ----------------

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
        E_west::CTMRG_PEPS_EdgeTensor, C_northwest::CTMRGCornerTensor,
        E_north::CTMRG_PEPS_EdgeTensor, A::PEPSSandwich,
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
        E_west::CTMRG_PF_EdgeTensor, C_northwest::CTMRGCornerTensor,
        E_north::CTMRG_PF_EdgeTensor, A::PFTensor,
    )
    return @tensor begin
        EC[χ_S DW; χ2] := E_west[χ_S DW; χ1] * C_northwest[χ1; χ2]
        ECE[χ_S χ_E; DW DN] := EC[χ_S DW; χ2] * E_north[χ2 DN; χ_E]
        corner[χ_S D_S; χ_E D_E] := ECE[χ_S χ_E; DW DN] * A[DW D_S; DN D_E]
    end
end

@generated function enlarge_northwest_corner(
        E_west::CTMRGEdgeTensor{T, S, N},
        C_northwest::CTMRGCornerTensor,
        E_north::CTMRGEdgeTensor{T, S, N},
        O::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    E_west_e = _pepo_edge_expr(:E_west, :SW, :WNW, :W, H)
    C_northwest_e = _corner_expr(:C_northwest, :WNW, :NNW)
    E_north_e = _pepo_edge_expr(:E_north, :NNW, :NE, :N, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_northwest´, :SW, :NE, :S, :E, H)

    rhs = Expr(
        :call, :*,
        E_west_e, C_northwest_e, E_north_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Northeast corner
# ----------------

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
        E_north::CTMRG_PEPS_EdgeTensor, C_northeast::CTMRGCornerTensor,
        E_east::CTMRG_PEPS_EdgeTensor, A::PEPSSandwich,
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
        E_north::CTMRG_PF_EdgeTensor, C_northeast::CTMRGCornerTensor,
        E_east::CTMRG_PF_EdgeTensor, A::PFTensor,
    )
    return @tensor begin
        EC[DN χ_W; χ2] := E_north[χ_W DN; χ1] * C_northeast[χ1; χ2]
        ECE[DN DE; χ_S χ_W] := EC[DN χ_W; χ2] * E_east[χ2 DE; χ_S]
        corner[χ_W D_W; χ_S D_S] := A[D_W D_S; DN DE] * ECE[DN DE; χ_S χ_W]
    end
end

@generated function enlarge_northeast_corner(
        E_north::CTMRGEdgeTensor{T, S, N},
        C_northeast::CTMRGCornerTensor,
        E_east::CTMRGEdgeTensor{T, S, N},
        O::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    E_north_e = _pepo_edge_expr(:E_north, :NW, :NNE, :N, H)
    C_northeast = _corner_expr(:C_northeast, :NNE, :ENE)
    E_east_e = _pepo_edge_expr(:E_east, :ENE, :SE, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_northeast´, :NW, :SE, :W, :S, H)

    rhs = Expr(
        :call, :*,
        E_north_e, C_northeast, E_east_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Southeast corner
# ----------------

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
        E_east::CTMRG_PEPS_EdgeTensor, C_southeast::CTMRGCornerTensor,
        E_south::CTMRG_PEPS_EdgeTensor, A::PEPSSandwich,
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
        E_east::CTMRG_PF_EdgeTensor, C_southeast::CTMRGCornerTensor,
        E_south::CTMRG_PF_EdgeTensor, A::PFTensor,
    )
    return @tensor begin
        EC[χ_N D1; χ2] := E_east[χ_N D1; χ1] * C_southeast[χ1; χ2]
        ECE[χ_N χ_W; D1 D2] := EC[χ_N D1; χ2] * E_south[χ2 D2; χ_W]
        corner[χ_N D_N; χ_W D_W] := ECE[χ_N χ_W; D1 D2] * A[D_W D2; D_N D1]
    end
end

@generated function enlarge_southeast_corner(
        E_east::CTMRGEdgeTensor{T, S, N},
        C_southeast::CTMRGCornerTensor,
        E_south::CTMRGEdgeTensor{T, S, N},
        O::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    E_east_e = _pepo_edge_expr(:E_east, :NE, :ESE, :E, H)
    C_southeast_e = _corner_expr(:C_southeast, :ESE, :SSE)
    E_south_e = _pepo_edge_expr(:E_south, :SSE, :SW, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_southeast´, :NE, :SW, :N, :W, H)

    rhs = Expr(
        :call, :*,
        E_east_e, C_southeast_e, E_south_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

# Southwest corner
# ----------------

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
        E_south::CTMRG_PEPS_EdgeTensor, C_southwest::CTMRGCornerTensor,
        E_west::CTMRG_PEPS_EdgeTensor, A::PEPSSandwich,
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
        E_south::CTMRG_PF_EdgeTensor, C_southwest::CTMRGCornerTensor,
        E_west::CTMRG_PF_EdgeTensor, A::PFTensor,
    )
    return @tensor begin
        EC[χ_E D1; χ2] := E_south[χ_E D1; χ1] * C_southwest[χ1; χ2]
        ECE[χ_E χ_N; D2 D1] := EC[χ_E D1; χ2] * E_west[χ2 D2; χ_N]
        corner[χ_E D_E; χ_N D_N] := ECE[χ_E χ_N; D2 D1] * A[D2 D1; D_N D_E]
    end
end

@generated function enlarge_southwest_corner(
        E_south::CTMRGEdgeTensor{T, S, N},
        C_southwest::CTMRGCornerTensor,
        E_west::CTMRGEdgeTensor{T, S, N},
        O::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    E_south_e = _pepo_edge_expr(:E_south, :SE, :SSW, :S, H)
    C_southwest_e = _corner_expr(:C_southwest, :SSW, :WSW)
    E_west_e = _pepo_edge_expr(:E_west, :WSW, :NW, :W, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    C_out_e = _pepo_enlarged_corner_expr(:C_southwest´, :SE, :NW, :E, :N, H)

    rhs = Expr(
        :call, :*,
        E_south_e, C_southwest_e, E_west_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $C_out_e := $rhs))
end

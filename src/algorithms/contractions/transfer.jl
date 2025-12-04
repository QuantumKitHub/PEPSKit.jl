#
# Transfer function for (CTMRG) edges
#

edge_transfer_left(v, ::Nothing, A, B) = edge_transfer_left(v, A, B)
edge_transfer_right(v, ::Nothing, A, B) = edge_transfer_right(v, A, B)

"""
    edge_transfer_left(v, Et, Eb)

Apply an edge transfer matrix to the left.

```
 ┌─Et─
-v │
 └─qƎ─
```
"""
@generated function edge_transfer_left(
        v::AbstractTensorMap{<:Any, S, 1, N₁},
        Etop::CTMRGEdgeTensor{<:Any, S, N₂},
        Ebot::CTMRGEdgeTensor{<:Any, S, N₂}
    ) where {S, N₁, N₂}
    t_out = tensorexpr(:v, -1, -(2:(N₁ + 1)))
    t_top = tensorexpr(:Etop, 2:(N₂ + 1), -(N₁ + 1))
    t_bot = tensorexpr(:Ebot, (-1, (3:(N₂ + 1))...), 1)
    t_in = tensorexpr(:v, 1, (-(2:N₁)..., 2))
    return macroexpand(
        @__MODULE__, :(return @tensor $t_out := $t_in * $t_top * $t_bot)
    )
end


"""
    edge_transfer_right(v, Et, Eb)

Apply an edge transfer matrix to the right.

```
─Et─┐
 │  v-
─qƎ─┘
```
"""
@generated function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 1, N₁},
        Etop::CTMRGEdgeTensor{<:Any, S, N₂},
        Ebot::CTMRGEdgeTensor{<:Any, S, N₂}
    ) where {S, N₁, N₂}
    t_out = tensorexpr(:v, -1, -(2:(N₁ + 1)))
    t_top = tensorexpr(:Etop, (-1, (3:(N₂ + 1))...), 1)
    t_bot = tensorexpr(:Ebot, (2, (3:(N₂ + 1))...), -(N₁ + 1))
    t_in = tensorexpr(:v, 1, (-(2:N₁)..., 2))
    return macroexpand(
        @__MODULE__, :(return @tensor $t_out := $t_top * $t_bot * $t_in)
    )
end

"""
    edge_transfer_left(v, O, Et, Eb)

Apply an edge transfer matrix to the left.

```
 ┌──Et─
 │  │
 v──O──
 │  │
 └──qƎ─
```
"""
function edge_transfer_left(
        v::CTMRGEdgeTensor{<:Any, S, 3},
        O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3},
        Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    @autoopt @tensor v´[χ_SE D_E_above D_E_below; χ_NE] :=
        v[χ_SW D_W_above D_W_below; χ_NW] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])

    return v´
end
function edge_transfer_left(
        v::CTMRGEdgeTensor{<:Any, S, 2},
        O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2},
        Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    @autoopt @tensor v´[χ_SE D_E; χ_NE] :=
        v[χ_SW D_W; χ_NW] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]

    return v´
end

"""
    transfer_right(v, Et, Eb)
    
Apply an edge transfer matrix to the right.

```
──Et─┐
  │  │
──O──v
  │  │
──qƎ─┘
```
"""
function edge_transfer_right(
        v::CTMRGEdgeTensor{<:Any, S, 3},
        O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3},
        Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    @autoopt @tensor v′[χ_NW D_W_above D_W_below; χ_SW] :=
        v[χ_NE D_E_above D_E_below; χ_SE] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])

    return v′
end
function edge_transfer_right(
        v::CTMRGEdgeTensor{<:Any, S, 2},
        O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2},
        Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_NW D_W; χ_SW] :=
        v[χ_NE D_E; χ_SE] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]

    return v′
end

"""
    edge_transfer_left(v, O, Et, Eb)

Apply an edge transfer matrix to the left on an excited vector.

```
 ┌──Et─
 │  │
-v──O──
 │  │
 └──qƎ─
```
"""
function edge_transfer_left(
        v::CTMRGEdgeTensor{<:Any, S, 4}, O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    return @autoopt @tensor v′[χ_SE D_E_above d_string D_E_below; χ_NE] :=
        v[χ_SW D_W_above d_string D_W_below; χ_NW] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_left(
        v::CTMRGEdgeTensor{<:Any, S, 3}, O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2}, Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_SE D_E d_string; χ_NE] :=
        v[χ_SW D_W d_string; χ_NW] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]
end

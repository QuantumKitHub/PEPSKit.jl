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
    t_out = tensorexpr(:v′, -1, -(2:(N₁ + 1)))
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
    t_out = tensorexpr(:v′, -1, -(2:(N₁ + 1)))
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
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    return @autoopt @tensor v′[χ_SE D_E_above D_E_below; χ_NE] :=
        v[χ_SW D_W_above D_W_below; χ_NW] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_left(
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PEPOPurifiedSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    ket_tensor = twistdual(ket(O), (1, 2))
    bra_tensor = bra(O)
    return @autoopt @tensor v′[χ_SE D_E_above D_E_below; χ_NE] :=
        v[χ_SW D_W_above D_W_below; χ_NW] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket_tensor[d a; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra_tensor[d a; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_left(
        v::AbstractTensorMap{<:Any, S, 2, 1}, O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2}, Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_SE D_E; χ_NE] :=
        v[χ_SW D_W; χ_NW] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]
end

"""
    edge_transfer_right(v, Et, Eb)
    
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
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    return @autoopt @tensor v′[χ_NW D_W_above D_W_below; χ_SW] :=
        v[χ_NE D_E_above D_E_below; χ_SE] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PEPOPurifiedSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    ket_tensor = twistdual(ket(O), (1, 2))
    bra_tensor = bra(O)
    return @autoopt @tensor v′[χ_NW D_W_above D_W_below; χ_SW] :=
        v[χ_NE D_E_above D_E_below; χ_SE] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket_tensor[d a; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra_tensor[d a; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 2, 1}, O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2}, Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_NW D_W; χ_SW] :=
        v[χ_NE D_E; χ_SE] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]
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
        v::AbstractTensorMap{<:Any, S, 4, 1}, O::PEPSSandwich,
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
        v::AbstractTensorMap{<:Any, S, 4, 1}, O::PEPOPurifiedSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    ket_tensor = twistdual(ket(O), (1, 2))
    bra_tensor = bra(O)
    return @autoopt @tensor v′[χ_SE D_E_above d_string D_E_below; χ_NE] :=
        v[χ_SW D_W_above d_string D_W_below; χ_NW] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket_tensor[d a; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra_tensor[d a; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_left(
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2}, Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_SE D_E d_string; χ_NE] :=
        v[χ_SW D_W d_string; χ_NW] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]
end

"""
    edge_transfer_right(v, O, Et, Eb)
    
Apply an edge transfer matrix to the right on an excited vector.

```
──Et─┐
  │  │
──O──v-
  │  │
──qƎ─┘
```
"""
function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 4, 1}, O::PEPSSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    return @autoopt @tensor v′[χ_NW D_W_above d_string D_W_below; χ_SW] :=
        v[χ_NE D_E_above d_string D_E_below; χ_SE] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 4, 1}, O::PEPOPurifiedSandwich,
        Etop::CTMRGEdgeTensor{<:Any, S, 3}, Ebot::CTMRGEdgeTensor{<:Any, S, 3},
    ) where {S}
    ket_tensor = twistdual(ket(O), (1, 2))
    bra_tensor = bra(O)
    return @autoopt @tensor v′[χ_NW D_W_above d_string D_W_below; χ_SW] :=
        v[χ_NE D_E_above d_string D_E_below; χ_SE] *
        Etop[χ_NW D_N_above D_N_below; χ_NE] *
        Ebot[χ_SE D_S_above D_S_below; χ_SW] *
        ket_tensor[d a; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra_tensor[d a; D_N_below D_E_below D_S_below D_W_below])
end
function edge_transfer_right(
        v::AbstractTensorMap{<:Any, S, 3, 1}, O::PFTensor,
        Etop::CTMRGEdgeTensor{<:Any, S, 2}, Ebot::CTMRGEdgeTensor{<:Any, S, 2},
    ) where {S}
    return @autoopt @tensor v′[χ_NW D_W d_string; χ_SW] :=
        v[χ_NE D_E d_string; χ_SE] *
        Etop[χ_NW D_N; χ_NE] *
        Ebot[χ_SE D_S; χ_SW] *
        O[D_W D_S; D_N D_E]
end

"""
Map north-boundary site `k` to the corresponding site in the east-to-west south boundary.
`N` is the finite window length, including the two boundary sites.
"""
south_site(k::Int, N::Int) = N + 1 - k

"""
Construct the endpoint identities for a finite north-W-south sandwich.
```
    ┌-←-- north --←-┐
    |       |       |
    L-←---- W ----←-R
    |       |       |
    └-→-- south --→-┘
```
"""
function _window_edge_boundaries(south::FiniteMPS, W::FiniteMPO, north::FiniteMPS)
    N = length(north)
    length(south) == length(W) == N || throw(DimensionMismatch("row and boundary lengths must match"))
    left = isomorphism(
        storagetype(north.AL[1]), domain(south.AR[N])[1] ⊗ space(W[1], 1)', space(north.AL[1], 1)
    )
    right = isomorphism(
        storagetype(north.AR[N]), domain(north.AR[N])[1] ⊗ domain(W[N])[2], space(south.AL[1], 1)
    )
    return left, right
end

"""
Build left and right environments outside `site` in the north-W-south sandwich.
`lefts[k]` contains columns before `k`; `rights[k - site + 1]` contains columns after `k`.
Only valid entries are stored, with lengths `site` and `length(north) - site + 1`, respectively.

Note that sites in `south` are ordered from right (east) to left (west).
"""
function _window_edge_environments(south::FiniteMPS, W::FiniteMPO, north::FiniteMPS, site::Int)
    left, right = _window_edge_boundaries(south, W, north)
    N = length(north)
    lefts, rights = [left], [right]
    for k in 1:(site - 1)
        push!(lefts, last(lefts) * edge_transfermatrix(north.AL[k], W[k], south.AR[south_site(k, N)]))
    end
    for k in N:-1:(site + 1)
        push!(rights, edge_transfermatrix(north.AR[k], W[k], south.AL[south_site(k, N)]) * last(rights))
    end
    reverse!(rights)
    return (; lefts, rights)
end

"""
Contract opposite-oriented boundary MPSs without conjugation.
"""
function dot_noconj(south::FiniteMPS, north::FiniteMPS)
    N = length(north)
    length(south) == N || throw(DimensionMismatch("boundary lengths must match"))
    right = isomorphism(storagetype(north.AR[N]), domain(north.AR[N]), space(south.AL[1], 1))
    for k in N:-1:1
        top = k == 1 ? north.AC[k] : north.AR[k]
        bottom = k == 1 ? south.AC[N] : south.AL[south_site(k, N)]
        right = edge_transfermatrix(top, bottom) * right
    end
    left = isomorphism(storagetype(right), domain(south.AC[N]), space(north.AC[1], 1))
    return tr(left * right)
end

"""
Contract a row MPO between opposite-oriented boundary MPSs without conjugation.
"""
function dot_noconj(south::FiniteMPS, W::FiniteMPO, north::FiniteMPS)
    left, right = _window_edge_boundaries(south, W, north)
    N = length(north)
    for k in N:-1:1
        top = k == 1 ? north.AC[k] : north.AR[k]
        bottom = k == 1 ? south.AC[N] : south.AL[south_site(k, N)]
        right = edge_transfermatrix(top, W[k], bottom) * right
    end
    return _contract_transfer_boundaries(left, right)
end

"""
Contract the left and right transfer-matrix environments to a scalar.
```
    (north)
    ┌-←-- 3 --←-┐
    |           |
    L-←-- 2 --←-R
    |           |
    └-→-- 1 --→-┘
    (south)
```
"""
function _contract_transfer_boundaries(left::MPSTensor, right::MPSTensor)
    # The three bonds close around the window without crossing
    return @tensor left[1 2; 3] * right[3 2; 1]
end

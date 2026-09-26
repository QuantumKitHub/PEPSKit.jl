#= 
The construction of bond environment for Neighborhood Tensor Update (NTU) 
is adapted from YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

"""
Algorithms to construct bond environment for Neighborhood Tensor Update (NTU).
"""
abstract type NeighbourEnv end

"""
For a rank-(2,2) tensor `T_{ik;jl}`, approximately decompose it to the
product of two rank-2 tensors `A_{ik;} * B_{;jl}` using truncated SVD
that keeps only the largest singular value in the charge-neutral sector.

The kept singular value is the largest in the entire SVD spectrum if:
- `permute(T, ((1, 3), (2, 4)))` is positive semi-definite, and
- `space(T, 1) == space(T, 2)'` and `space(T, 3) == space(T, 4)`,
"""
function _svd_cut!(t::AbstractTensorMap{<:Any, <:Any, 2, 2})
    A, B = left_orth!(t; trunc = truncspace(oneunit(spacetype(t))))
    return removeunit(A, numind(A)), removeunit(B, 1)
end

"""
Algorithm struct for "NTU-NN" bond environment. 
"""
struct NNEnv <: NeighbourEnv end
"""
Calculate the bond environment within "NTU-NN" approximation.
```
    -1      ●=======●
            ║       ║
    0   ●===X==   ==Y===●
            ║       ║
    1       ●=======●
        -1  0       1   2
```
"""
function bondenv_ntu(
        row::Int, col::Int, X, Y, state::InfiniteState, alg::NNEnv
    )
    neighbors = [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]
    m = collect_neighbors(state, row, col, neighbors)
    X, Y = _prepare_site_tensor(X), _prepare_site_tensor(Y)
    return _contract_ntu_NNEnv(
        X, Y, hair_w(m[0, -1]), hair_e(m[0, 2]),
        cor_nw(m[-1, 0]), cor_ne(m[-1, 1]), cor_sw(m[1, 0]), cor_se(m[1, 1])
    )
end

"""
Algorithm struct for "NTU-NN+" bond environment. 
"""
struct NNpEnv <: NeighbourEnv end
"""
Calculate the bond environment within "NTU-NN+" approximation.
```
    -2          ●.......●
                ║       ║
    -1      ○===●=======●===○
            ║   ║       ║   ║
    0   ●===●===X==   ==Y===●===●
            ║   ║       ║   ║
    1       ○===●=======●===○
                ║       ║
    2           ●.......●
        -2  -1  0       1   2   3
```
Dotted lines and ○ are splitted using SVD with `truncrank(1)`.
"""
function bondenv_ntu(
        row::Int, col::Int, X, Y, state::InfiniteState, alg::NNpEnv
    )
    neighbors = [
        (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (1, 2), (0, 2), (-1, 2),
        (-1, 1), (-1, 0), (0, -2), (2, 0), (2, 1), (0, 3), (-2, 1), (-2, 0),
    ]
    ms = collect_neighbors(state, row, col, neighbors)
    X, Y = _prepare_site_tensor(X), _prepare_site_tensor(Y)

    # ---- hairs (size D^2) with a 1D auxiliary leg ----

    @tensor top[-1 -2; -3 -4] := cor_nw(ms[-2, 0])[1 2 -1 -2] * cor_ne(ms[-2, 1])[-3 -4 1 2]
    tl, tr = _svd_cut!(top)

    @tensor bot[-1 -2; -3 -4] := cor_sw(ms[2, 0])[-1 -2 1 2] * cor_se(ms[2, 1])[-3 -4 1 2]
    bl, br = _svd_cut!(bot)

    nw = permute(cor_nw(ms[-1, -1]), ((3, 4), (1, 2)))
    nw1, nw2 = _svd_cut!(nw)

    ne = permute(cor_ne(ms[-1, 2]), ((3, 4), (1, 2)))
    ne1, ne2 = _svd_cut!(ne)

    sw = permute(cor_sw(ms[1, -1]), ((1, 2), (3, 4)))
    sw1, sw2 = _svd_cut!(sw)

    se = permute(cor_se(ms[1, 2]), ((3, 4), (1, 2)))
    se1, se2 = _svd_cut!(se)

    @tensoropt hW[DXw1 DXw0] :=
        hair_w(ms[0, -2])[Dw21 Dw20] *
        nw1[Dnw11 Dnw10] * sw1[Dsw11 Dsw10] *
        twistdual(ms[0, -1], 1)[phW Dnw10 DXw0 Dsw10 Dw20] *
        conj(ms[0, -1][phW Dnw11 DXw1 Dsw11 Dw21])
    @tensoropt hE[DYe1 DYe0] :=
        hair_e(ms[0, 3])[De21 De20] *
        ne2[Dne21 Dne20] * se2[Dse21 Dse20] *
        twistdual(ms[0, 2], 1)[phE Dne20 De20 Dse20 DYe0] *
        conj(ms[0, 2][phE Dne21 De21 Dse21 DYe1])
    @tensoropt NW[Dn1 Dn0 DXn1 DXn0] :=
        tl[Dtl1 Dtl0] * nw2[Dnw21 Dnw20] *
        twistdual(ms[-1, 0], 1)[pNW Dtl0 Dn0 DXn0 Dnw20] *
        conj(ms[-1, 0][pNW Dtl1 Dn1 DXn1 Dnw21])
    @tensoropt NE[DYn1 DYn0 Dn1 Dn0] :=
        tr[Dtr1 Dtr0] * ne1[Dne11 Dne10] *
        twistdual(ms[-1, 1], 1)[pNE Dtr0 Dne10 DYn0 Dn0] *
        conj(ms[-1, 1][pNE Dtr1 Dne11 DYn1 Dn1])
    @tensoropt SW[DXs1 DXs0 Ds1 Ds0] :=
        bl[Dbl1 Dbl0] * sw2[Dsw21 Dsw20] *
        twistdual(ms[1, 0], 1)[pSW DXs0 Ds0 Dbl0 Dsw20] *
        conj(ms[1, 0][pSW DXs1 Ds1 Dbl1 Dsw21])
    @tensoropt SE[DYs1 DYs0 Ds1 Ds0] :=
        br[Dbr1 Dbr0] * se1[Dse11 Dse10] *
        twistdual(ms[1, 1], 1)[pSE DYs0 Dse10 Dbr0 Ds0] *
        conj(ms[1, 1][pSE DYs1 Dse11 Dbr1 Ds1])
    return _contract_ntu_NNEnv(X, Y, hW, hE, NW, NE, SW, SE)
end

# Common NN/NN+ network; each virtual bond has a bra and a ket index.
#             NW -- NE
#              |     |
#       hW --- X     Y --- hE
#              |     |
#             SW -- SE
# X/Y identify the central site; n/e/s/w identify its virtual leg.
# N/S join the two upper/lower corners; pX/pY are physical indices.
# Suffix 0 denotes ket and 1 denotes bra.
# The four open QR indices are Xq1, Yq1, Xq0, Yq0.
const _NTU_ENV_NETWORK = [
    [:Xw1, :Xw0],
    [:Ye1, :Ye0],
    [:N1, :N0, :Xn1, :Xn0],
    [:Yn1, :Yn0, :N1, :N0],
    [:Xs1, :Xs0, :S1, :S0],
    [:Ys1, :Ys0, :S1, :S0],
    [:pX, :Xn1, :Xq1, :Xs1, :Xw1],
    [:pX, :Xn0, :Xq0, :Xs0, :Xw0],
    [:pY, :Yn1, :Ye1, :Ys1, :Yq1],
    [:pY, :Yn0, :Ye0, :Ys0, :Yq0],
]

"""
Choose the common environment's contraction order using actual leg dimensions.
The fixed ten-tensor network is independent of the MPO path and its length.
Floating-point costs avoid integer overflow; dense dimensions approximate symmetry-block costs.
"""
function _ntu_contraction_order(tensors::NamedTuple)
    costs = Dict(
        label => Float64(dim(space(t, axis)))
            for (t, labels) in zip(tensors, _NTU_ENV_NETWORK)
            for (axis, label) in enumerate(labels)
    )
    tree, _ = TensorOperations.optimaltree(_NTU_ENV_NETWORK, costs)
    return first(TensorOperations.tree2indexorder(tree, _NTU_ENV_NETWORK))
end

"""
Compile the selected order to ordinary `@tensor` contractions with statically known ranks.
Specializations depend on the contraction order and tensor types, not on leg dimensions.
"""
@generated function _contract_ntu_kernel(::Val{Order}, tensors::NamedTuple) where {Order}
    order = Expr(:tuple, Order...)
    return macroexpand(
        @__MODULE__, :(
            @tensor order = $order benv[Xq1 Yq1; Xq0 Yq0] :=
                tensors.hW[Xw1 Xw0] * tensors.hE[Ye1 Ye0] *
                tensors.NW[N1 N0 Xn1 Xn0] * tensors.NE[Yn1 Yn0 N1 N0] *
                tensors.SW[Xs1 Xs0 S1 S0] * tensors.SE[Ys1 Ys0 S1 S0] *
                conj(tensors.Xbra[pX Xn1 Xq1 Xs1 Xw1]) *
                tensors.Xket[pX Xn0 Xq0 Xs0 Xw0] *
                conj(tensors.Ybra[pY Yn1 Ye1 Ys1 Yq1]) *
                tensors.Yket[pY Yn0 Ye0 Ys0 Yq0]
        )
    )
end

"Contract an NN or NN+ boundary while keeping the central bra and ket layers separate."
function _contract_ntu_NNEnv(X::PEPSTensor, Y::PEPSTensor, hW, hE, NW, NE, SW, SE)
    # Bra factors are conjugated in the kernel; ket factors carry the physical-leg twist.
    tensors = (;
        hW, hE, NW, NE, SW, SE, Xbra = X, Xket = twistdual(X, 1),
        Ybra = Y, Yket = twistdual(Y, 1),
    )
    order = _ntu_contraction_order(tensors)
    # The runtime-selected kernel always returns the same concrete rank-(2,2) type.
    T = tensormaptype(spacetype(X), 2, 2, TensorKit.promote_storagetype(tensors...))
    benv = _contract_ntu_kernel(Val(Tuple(order)), tensors)::T
    return normalize!(benv, Inf)
end

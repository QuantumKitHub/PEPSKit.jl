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
SVD which truncion only keeping the leading singular value in the trivial sector.
"""
function _svd_cut!(t::AbstractTensorMap)
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
    @tensoropt benv[Dw1 De1; Dw0 De0] :=
        ( # southwest half
        cor_se(m[1, 1])[Dse1 Dse0 Ds1 Ds0] *
            cor_sw(m[1, 0])[Dsw1 Dsw0 Ds1 Ds0] *
            edge_w(X, hair_w(m[0, -1]))[Dnw1 Dnw0 Dw1 Dw0 Dsw1 Dsw0]
    ) * ( # northeast half
        cor_nw(m[-1, 0])[Dn1 Dn0 Dnw1 Dnw0] *
            cor_ne(m[-1, 1])[Dne1 Dne0 Dn1 Dn0] *
            edge_e(Y, hair_e(m[0, 2]))[Dne1 Dne0 Dse1 Dse0 De1 De0]
    )
    normalize!(benv, Inf)
    return benv
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

    @tensoropt benv[Dw1, De1; Dw0, De0] :=
        ( # hW
        hair_w(ms[0, -2])[Dw21 Dw20] *
            nw1[Dnw11 Dnw10] * sw1[Dsw11 Dsw10] *
            twistdual(ms[0, -1], 1)[phW Dnw10 DXw0 Dsw10 Dw20] *
            conj(ms[0, -1][phW Dnw11 DXw1 Dsw11 Dw21])
    ) * ( # hE
        hair_e(ms[0, 3])[De21 De20] *
            ne2[Dne21 Dne20] * se2[Dse21 Dse20] *
            twistdual(ms[0, 2], 1)[phE Dne20 De20 Dse20 DYe0] *
            conj(ms[0, 2][phE Dne21 De21 Dse21 DYe1])
    ) * ( # NW
        tl[Dtl1 Dtl0] * nw2[Dnw21 Dnw20] *
            twistdual(ms[-1, 0], 1)[pNW Dtl0 Dn0 DXn0 Dnw20] *
            conj(ms[-1, 0][pNW Dtl1 Dn1 DXn1 Dnw21])
    ) * ( # NE
        tr[Dtr1 Dtr0] * ne1[Dne11 Dne10] *
            twistdual(ms[-1, 1], 1)[pNE Dtr0 Dne10 DYn0 Dn0] *
            conj(ms[-1, 1][pNE Dtr1 Dne11 DYn1 Dn1])
    ) * ( # SW
        bl[Dbl1 Dbl0] * sw2[Dsw21 Dsw20] *
            twistdual(ms[1, 0], 1)[pSW DXs0 Ds0 Dbl0 Dsw20] *
            conj(ms[1, 0][pSW DXs1 Ds1 Dbl1 Dsw21])
    ) * ( # SE
        br[Dbr1 Dbr0] * se1[Dse11 Dse10] *
            twistdual(ms[1, 1], 1)[pSE DYs0 Dse10 Dbr0 Ds0] *
            conj(ms[1, 1][pSE DYs1 Dse11 Dbr1 Ds1])
    ) * conj(X[pX DXn1 Dw1 DXs1 DXw1]) *
        twistdual(X, 1)[pX DXn0 Dw0 DXs0 DXw0] *
        conj(Y[pY DYn1 DYe1 DYs1 De1]) *
        twistdual(Y, 1)[pY DYn0 DYe0 DYs0 De0]
    normalize!(benv, Inf)
    return benv
end

"""
Algorithm struct for "NTU-NNN" bond environment. 
"""
struct NNNEnv <: NeighbourEnv end
"""
Calculates the bond environment within "NTU-NNN" approximation.
```
    -1  ●===●=======●===●
        ║   ║       ║   ║
    0   ●===X==   ==Y===●
        ║   ║       ║   ║
    1   ●===●=======●===●
        -1  0       1   2
```
"""
function bondenv_ntu(
        row::Int, col::Int, X, Y, state::InfiniteState, alg::NNNEnv
    )
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (1, 0), (1, 1), (1, 2), (0, 2),
        (-1, 2), (-1, 1), (-1, 0),
    ]
    m = collect_neighbors(state, row, col, neighbors)
    X, Y = _prepare_site_tensor(X), _prepare_site_tensor(Y)
    @tensoropt benv[Dw1 De1; Dw0 De0] :=
        ( # NW corner
        cor_nw(m[-1, -1])[χnw1 χnw0 χwn1 χwn0] *
            edge_n(m[-1, 0])[χn1 χn0 Dnw1 Dnw0 χnw1 χnw0] *
            edge_w(m[0, -1])[χwn1 χwn0 Dww1 Dww0 χws1 χws0] *
            conj(X[dX Dnw1 Dw1 Dsw1 Dww1]) *
            twistdual(X, 1)[dX Dnw0 Dw0 Dsw0 Dww0]
    ) * ( # SE corner
        cor_se(m[1, 2])[χes1 χes0 χse1 χse0] *
            edge_s(m[1, 1])[Dse1 Dse0 χse1 χse0 χs1 χs0] *
            edge_e(m[0, 2])[χen1 χen0 χes1 χes0 Dee1 Dee0] *
            conj(Y[dY Dne1 Dee1 Dse1 De1]) *
            twistdual(Y, 1)[dY Dne0 Dee0 Dse0 De0]
    ) * # NE edge
        edge_n(m[-1, 1])[χne1 χne0 Dne1 Dne0 χn1 χn0] *
        cor_ne(m[-1, 2])[χen1 χen0 χne1 χne0] *
        # SW edge
        cor_sw(m[1, -1])[χws1 χws0 χsw1 χsw0] *
        edge_s(m[1, 0])[Dsw1 Dsw0 χs1 χs0 χsw1 χsw0]
    normalize!(benv, Inf)
    return benv
end

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
SVD with `truncrank(1)`.
"""
function _svd_cut!(t::AbstractTensorMap)
    t1, s, t2 = svd_trunc!(t; trunc = truncrank(1))
    return absorb_s(t1, s, t2)
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
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNEnv
    ) where {T, S <: InfiniteState}
    neighbors = [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]
    m = collect_neighbors(state, row, col, neighbors)
    X, Y = _prepare_site_tensor(X), _prepare_site_tensor(Y)
    # southwest half
    @autoopt @tensor benv_sw[Dse1 Dse0 Dw1 Dw0 Dnw1 Dnw0] :=
        cor_se(m[1, 1])[Dse1 Dse0 Ds1 Ds0] *
        cor_sw(m[1, 0])[Dsw1 Dsw0 Ds1 Ds0] *
        edge_w(X, hair_w(m[0, -1]))[Dnw1 Dnw0 Dw1 Dw0 Dsw1 Dsw0]
    normalize!(benv_sw, Inf)
    # northeast half
    @autoopt @tensor benv_ne[Dnw1 Dnw0 De1 De0 Dse1 Dse0] :=
        cor_nw(m[-1, 0])[Dn1 Dn0 Dnw1 Dnw0] *
        cor_ne(m[-1, 1])[Dne1 Dne0 Dn1 Dn0] *
        edge_e(Y, hair_e(m[0, 2]))[Dne1 Dne0 Dse1 Dse0 De1 De0]
    normalize!(benv_ne, Inf)
    @tensor benv[Dw1 De1; Dw0 De0] :=
        benv_sw[Dse1 Dse0 Dw1 Dw0 Dnw1 Dnw0] * benv_ne[Dnw1 Dnw0 De1 De0 Dse1 Dse0]
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
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNpEnv
    ) where {T, S <: InfiniteState}
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

    m = ms[0, -1]
    @tensoropt hW[anw DXw1 DXw0 asw] :=
        hair_w(ms[0, -2])[Dw21 Dw20] *
        nw1[Dnw11 Dnw10 anw] * sw1[Dsw11 Dsw10 asw] *
        twistdual(m, 1)[p Dnw10 DXw0 Dsw10 Dw20] *
        conj(m[p Dnw11 DXw1 Dsw11 Dw21])

    m = ms[0, 2]
    @tensoropt hE[ane DYe1 DYe0 ase] :=
        hair_e(ms[0, 3])[De21 De20] *
        ne2[ane Dne21 Dne20] * se2[ase Dse21 Dse20] *
        twistdual(m, 1)[p Dne20 De20 Dse20 DYe0] *
        conj(m[p Dne21 De21 Dse21 DYe1])

    # ---- corners (size D^4) with a 1D auxiliary leg ----

    m = ms[-1, 0]
    @tensoropt NW[at Dn1 Dn0 DXn1 DXn0 anw] :=
        tl[Dtl1 Dtl0 at] * nw2[anw Dnw21 Dnw20] *
        twistdual(m, 1)[p Dtl0 Dn0 DXn0 Dnw20] *
        conj(m[p Dtl1 Dn1 DXn1 Dnw21])

    m = ms[-1, 1]
    @tensoropt NE[at Dn1 Dn0 DYn1 DYn0 ane] :=
        tr[at Dtr1 Dtr0] * ne1[Dne11 Dne10 ane] *
        twistdual(m, 1)[p Dtr0 Dne10 DYn0 Dn0] *
        conj(m[p Dtr1 Dne11 DYn1 Dn1])

    m = ms[1, 0]
    @tensoropt SW[asw DXs1 DXs0 Ds1 Ds0 ab] :=
        bl[Dbl1 Dbl0 ab] * sw2[asw Dsw21 Dsw20] *
        twistdual(m, 1)[p DXs0 Ds0 Dbl0 Dsw20] *
        conj(m[p DXs1 Ds1 Dbl1 Dsw21])

    m = ms[1, 1]
    @tensoropt SE[ase DYs1 DYs0 Ds1 Ds0 ab] :=
        br[ab Dbr1 Dbr0] * se1[Dse11 Dse10 ase] *
        twistdual(m, 1)[p DYs0 Dse10 Dbr0 Ds0] *
        conj(m[p DYs1 Dse11 Dbr1 Ds1])

    # ---- left half ----
    @tensoropt benvL[at Dn1 Dn0 Dw1 Dw0 Ds1 Ds0 ab] :=
        hW[anw DXw1 DXw0 asw] *
        NW[at Dn1 Dn0 DXn1 DXn0 anw] *
        SW[asw DXs1 DXs0 Ds1 Ds0 ab] *
        twistdual(X, 1)[p DXn0 Dw0 DXs0 DXw0] *
        conj(X[p DXn1 Dw1 DXs1 DXw1])
    normalize!(benvL, Inf)

    # ---- right half ----

    @tensoropt benvR[at Dn1 Dn0 De1 De0 Ds1 Ds0 ab] :=
        hE[ane DYe1 DYe0 ase] *
        NE[at Dn1 Dn0 DYn1 DYn0 ane] *
        SE[ase DYs1 DYs0 Ds1 Ds0 ab] *
        twistdual(Y, 1)[p DYn0 DYe0 DYs0 De0] *
        conj(Y[p DYn1 DYe1 DYs1 De1])
    normalize!(benvR, Inf)

    # ---- the full NN+ environment ----

    @tensor benv[Dw1, De1; Dw0, De0] :=
        benvL[at Dn1 Dn0 Dw1 Dw0 Ds1 Ds0 ab] *
        benvR[at Dn1 Dn0 De1 De0 Ds1 Ds0 ab]
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
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNNEnv
    ) where {T, S <: InfiniteState}
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (1, 0), (1, 1), (1, 2), (0, 2),
        (-1, 2), (-1, 1), (-1, 0),
    ]
    m = collect_neighbors(state, row, col, neighbors)
    X, Y = _prepare_site_tensor(X), _prepare_site_tensor(Y)
    #= left half
    -1  ●======●=== -1/-2
        ║      ║
    0   ●======X=== -3/-4
        ║      ║
    ....D1.....D2...
        ║      ║
    1   ●==D3==●=== -5/-6
        -1     0
    =#
    vecl = enlarge_corner_nw(cor_nw(m[-1, -1]), edge_n(m[-1, 0]), edge_w(m[0, -1]), X)
    @tensor vecl[:] :=
        cor_sw(m[1, -1])[D11 D10 D31 D30] *
        edge_s(m[1, 0])[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    normalize!(vecl, Inf)
    #= right half
    -1  -1/-2===●==D1==●
                ║      ║
        ........D2.....D3...
                ║      ║
    0   -3/-4===Y======●
                ║      ║     
    1   -5/-6===●======●
                0      1
    =#
    vecr = enlarge_corner_se(cor_se(m[1, 2]), edge_s(m[1, 1]), edge_e(m[0, 2]), Y)
    @tensor vecr[:] :=
        edge_n(m[-1, 1])[D11 D10 D21 D20 -1 -2] *
        cor_ne(m[-1, 2])[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    normalize!(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    normalize!(benv, Inf)
    return benv
end

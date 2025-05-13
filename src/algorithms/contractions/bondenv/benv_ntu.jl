#= 
The construction of bond environment for Neighborhood Tensor Update (NTU) 
is adapted from YASTN (https://github.com/yastn/yastn).
Copyright 2024 The YASTN Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0
=#

"""
Algorithms to construct bond environment for Neighborhood Tensor Update (NTU).
"""
abstract type NTUEnvAlgorithm end

"""
Construct the "NTU-NN" bond environment. 
```
            (-1 +0)══(-1 +1)
                ║        ║
    (+0 -1)═════X══   ═══Y═══(+0 +2)
                ║        ║
            (+1 +0)══(+1 +1)
```
"""
@kwdef struct NTUEnvNN <: NTUEnvAlgorithm
    add_bwt::Bool = true
end
"""
Calculate the bond environment within "NTU-NN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, alg::NTUEnvNN
) where {T<:Union{PEPSTensor,PEPSOrth}}
    neighbors = [
        (-1, 0, [NORTH, WEST]),
        (0, -1, [NORTH, SOUTH, WEST]),
        (1, 0, [SOUTH, WEST]),
        (1, 1, [EAST, SOUTH]),
        (0, 2, [NORTH, EAST, SOUTH]),
        (-1, 1, [NORTH, EAST]),
    ]
    m = collect_neighbors(peps, row, col, neighbors, alg.add_bwt)
    #= contraction indices

                (-1 +0) ══ Dn ══ (-1 +1)
                    ║               ║
            ........Dnw......       Dne
                    ║       :       ║
        (+0 -1) ═══ X ══ Dw : De ══ Y ═══ (+0 +2)
                    ║       :       ║
                    Dsw     :.......Dse........
                    ║               ║
                (+1 +0) ══ Ds ══ (+1 +1)    
    =#
    # bottom-left half
    @autoopt @tensor benv_sw[Dse1 Dse0 Dw1 Dw0 Dnw1 Dnw0] :=
        cor_se(m[1, 1])[Dse1 Dse0 Ds1 Ds0] *
        cor_sw(m[1, 0])[Dsw1 Dsw0 Ds1 Ds0] *
        edge_w(X, hair_w(m[0, -1]))[Dnw1 Dnw0 Dw1 Dw0 Dsw1 Dsw0]
    benv_sw /= norm(benv_sw, Inf)
    # top-right half
    @autoopt @tensor benv_ne[Dnw1 Dnw0 De1 De0 Dse1 Dse0] :=
        cor_nw(m[-1, 0])[Dn1 Dn0 Dnw1 Dnw0] *
        cor_ne(m[-1, 1])[Dne1 Dne0 Dn1 Dn0] *
        edge_e(Y, hair_e(m[0, 2]))[Dne1 Dne0 Dse1 Dse0 De1 De0]
    benv_ne /= norm(benv_ne, Inf)
    @tensor benv[Dw1 De1; Dw0 De0] :=
        benv_sw[Dse1 Dse0 Dw1 Dw0 Dnw1 Dnw0] * benv_ne[Dnw1 Dnw0 De1 De0 Dse1 Dse0]
    return benv / norm(benv, Inf)
end

"""
Construct the "NTU-NNN" bond environment. 
```
    (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
        ║       ║        ║       ║
    (+0 -1)═════X══   ═══Y═══(+0 +2)
        ║       ║        ║       ║
    (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)
```
"""
@kwdef struct NTUEnvNNN <: NTUEnvAlgorithm
    add_bwt::Bool = true
end
"""
Calculates the bond environment within "NTU-NNN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, alg::NTUEnvNNN
) where {T<:Union{PEPSTensor,PEPSOrth}}
    neighbors = [
        (-1, -1, [NORTH, WEST]),
        (0, -1, [WEST]),
        (1, -1, [SOUTH, WEST]),
        (1, 0, [SOUTH]),
        (1, 1, [SOUTH]),
        (1, 2, [EAST, SOUTH]),
        (0, 2, [EAST]),
        (-1, 2, [NORTH, EAST]),
        (-1, 1, [NORTH]),
        (-1, 0, [NORTH]),
    ]
    m = collect_neighbors(peps, row, col, neighbors, alg.add_bwt)
    #= left half
        (-1 -1)══════(-1 +0)═ -1/-2
            ║           ║
        (+0 -1)════════ X ═══ -3/-4
            ║           ║
        ....D1..........D2.........
            ║           ║
        (+1 -1)═ D3 ═(+1 +0)═ -5/-6
    =#
    vecl = enlarge_corner_nw(cor_nw(m[-1, -1]), edge_n(m[-1, 0]), edge_w(m[0, -1]), X)
    @tensor vecl[:] :=
        cor_sw(m[1, -1])[D11 D10 D31 D30] *
        edge_s(m[1, 0])[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    vecl /= norm(vecl, Inf)
    #= right half
        -1/-2 ══ (-1 +1)═ D1 ═(-1 +2)
                    ║           ║
        ............D2..........D3...
                    ║           ║
        -3/-4 ═════ Y ═══════(+0 +2)
                    ║           ║     
        -5/-6 ══ (+1 +1)═════(+1 +2)
    =#
    vecr = enlarge_corner_se(cor_se(m[1, 2]), edge_s(m[1, 1]), edge_e(m[0, 2]), Y)
    @tensor vecr[:] :=
        edge_n(m[-1, 1])[D11 D10 D21 D20 -1 -2] *
        cor_ne(m[-1, 2])[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    vecr /= norm(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    return benv / norm(benv, Inf)
end

"""
Construct the "NTU-NNN+" bond environment. 
```
            (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                ║       ║        ║       ║
    (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)═(-1 +3)
                ║       ║        ║       ║
    (+0 -2)=(+0 -1)═════X══   ═══Y═══(+0 +2)═(+0 +3)
                ║       ║        ║       ║
    (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                ║       ║        ║       ║
            (+2 -1) (+2 +0)  (+2 +1) (+2 +2)
```
"""
@kwdef struct NTUEnvNNNp <: NTUEnvAlgorithm
    add_bwt::Bool = true
end
"""
Calculates the bond environment within "NTU-NNN+" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::T, Y::T, peps::InfiniteWeightPEPS, alg::NTUEnvNNNp
) where {T<:Union{PEPSTensor,PEPSOrth}}
    EMPTY = Vector{Int}()
    neighbors = [
        (-2, -1, [NORTH, EAST, WEST]),
        (-2, 0, [NORTH, EAST, WEST]),
        (-2, 1, [NORTH, EAST, WEST]),
        (-2, 2, [NORTH, EAST, WEST]),
        (-1, -2, [NORTH, SOUTH, WEST]),
        (-1, -1, EMPTY),
        (-1, 0, EMPTY),
        (-1, 1, EMPTY),
        (-1, 2, EMPTY),
        (-1, 3, [NORTH, EAST, SOUTH]),
        (0, -2, [NORTH, SOUTH, WEST]),
        (0, -1, EMPTY),
        (0, 2, EMPTY),
        (0, 3, [NORTH, EAST, SOUTH]),
        (1, -2, [NORTH, SOUTH, WEST]),
        (1, -1, EMPTY),
        (1, 0, EMPTY),
        (1, 1, EMPTY),
        (1, 2, EMPTY),
        (1, 3, [NORTH, EAST, SOUTH]),
        (2, -1, [EAST, SOUTH, WEST]),
        (2, 0, [EAST, SOUTH, WEST]),
        (2, 1, [EAST, SOUTH, WEST]),
        (2, 2, [EAST, SOUTH, WEST]),
    ]
    m = collect_neighbors(peps, row, col, neighbors, alg.add_bwt)
    #= left half
                (-2 -1)      (-2 +0)
                    ║           ║   
        (-1 -2)=(-1 -1)======(-1 +0)═ -1/-2
                    ║           ║
        (+0 -2)=(+0 -1)════════ X ═══ -3/-4
                    ║           ║
        ............D1..........D2.........
                    ║           ║
        (+1 -2)=(+1 -1)= D3 =(+1 +0)═ -5/-6
                    ║           ║
                (+2 -1)       (+2 +0)
    =#
    vecl = enlarge_corner_nw(
        cor_nw(m[-1, -1], hair_n(m[-2, -1]), hair_w(m[-1, -2])),
        edge_n(m[-1, 0], hair_n(m[-2, 0])),
        edge_w(m[0, -1], hair_w(m[0, -2])),
        X,
    )
    @tensor vecl[:] :=
        cor_sw(m[1, -1], hair_s(m[2, -1]), hair_w(m[1, -2]))[D11 D10 D31 D30] *
        edge_s(m[1, 0], hair_s(m[2, 0]))[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    vecl /= norm(vecl, Inf)
    #= 
                (-2 +1)      (-2 +2)
                    ║           ║
        -1/-2 ══(-1 +1)═ D1 ═(-1 +2)═(-1 +3)
                    ║           ║
        ............D2..........D3..........
                    ║           ║
        -3/-4 ═════ Y ═══════(+0 +2)═(+0 +3)
                    ║           ║
        -5/-6 ══(+1 +1)══════(+1 +2)═(+1 +3)
                    ║           ║
                (+2 +1)      (+2 +2)
    =#
    vecr = enlarge_corner_se(
        cor_se(m[1, 2], hair_e(m[1, 3]), hair_s(m[2, 2])),
        edge_s(m[1, 1], hair_s(m[2, 1])),
        edge_e(m[0, 2], hair_e(m[0, 3])),
        Y,
    )
    @tensor vecr[:] :=
        edge_n(m[-1, 1], hair_n(m[-2, 1]))[D11 D10 D21 D20 -1 -2] *
        cor_ne(m[-1, 2], hair_n(m[-2, 2]), hair_e(m[-1, 3]))[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    vecr /= norm(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    return benv / norm(benv, Inf)
end

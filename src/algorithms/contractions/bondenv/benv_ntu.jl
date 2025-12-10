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
Construct the "NTU-NN" bond environment. 
```
            (-1 +0)══(-1 +1)
                ║        ║
    (+0 -1)═════X══   ═══Y═══(+0 +2)
                ║        ║
            (+1 +0)══(+1 +1)
```
"""
struct NNEnv <: NeighbourEnv end
"""
Calculate the bond environment within "NTU-NN" approximation.
"""
function bondenv_ntu(
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNEnv
    ) where {T <: StateTensor, S <: InfinitePEPS}
    neighbors = [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]
    m = collect_neighbors(state, row, col, neighbors)
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
    normalize!(benv_sw, Inf)
    # top-right half
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
Construct the "NTU-NN+" bond environment. 
```
                    (-2 +0)┈┈(-2 +1)
                        ║        ║
            (-1 -1)┈(-1 +0)══(-1 +1)┈(-1 +2)
                ┊       ║        ║       ┊
    (+0 -2)=(+0 -1)═════X══   ═══Y═══(+0 +2)=(+0 +3)
                ┊       ║        ║       ┊
            (+1 -1)┈(+1 +0)══(+1 +1)┈(+1 +2)
                        ║        ║
                    (+2 +0)┈┈(+2 +1)
```
The tensors connected with dotted lines (e.g. (-1 +2)) 
are splitted into two hair tensors.
"""
struct NNpEnv <: NeighbourEnv end
"""
Calculate the bond environment within "NTU-NN+" approximation.
"""
function bondenv_ntu(
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNpEnv
    ) where {T <: StateTensor, S <: InfinitePEPS}
    neighbors = [
        (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (1, 2), (0, 2), (-1, 2),
        (-1, 1), (-1, 0), (0, -2), (2, 0), (2, 1), (0, 3), (-2, 1), (-2, 0),
    ]
    m = collect_neighbors(state, row, col, neighbors)
    error("To be implemented.")
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
struct NNNEnv <: NeighbourEnv end
"""
Calculates the bond environment within "NTU-NNN" approximation.
"""
function bondenv_ntu(
        row::Int, col::Int, X::T, Y::T, state::S, alg::NNNEnv
    ) where {T <: StateTensor, S <: InfinitePEPS}
    neighbors = [
        (-1, -1), (0, -1), (1, -1),
        (1, 0), (1, 1), (1, 2), (0, 2),
        (-1, 2), (-1, 1), (-1, 0),
    ]
    m = collect_neighbors(state, row, col, neighbors)
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
    normalize!(vecl, Inf)
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
    normalize!(vecr, Inf)
    # combine left and right part
    @tensor benv[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
    normalize!(benv, Inf)
    return benv
end

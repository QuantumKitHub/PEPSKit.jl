"""
Neighborhood tensor update (NTU) algorithms to construct bond environment.
"""
abstract type NTUEnvAlgorithm <: BondEnvAlgorithm end

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
struct NTUEnvNN <: NTUEnvAlgorithm end
"""
Calculate the bond environment within "NTU-NN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::PEPSOrth, Y::PEPSOrth, peps::InfinitePEPS, ::NTUEnvNN
)
    neighbors = [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]
    m = collect_neighbors(peps, row, col, neighbors)
    env_l = edge_l(X, hair_l(m[0, -1]))
    env_r = edge_r(Y, hair_r(m[0, 2]))
    ctl = cor_tl(m[-1, 0])
    ctr = cor_tr(m[-1, 1])
    cbr = cor_br(m[1, 1])
    cbl = cor_bl(m[1, 0])
    #= contraction indices

        ctl ═════ Dt ═════ ctr
        ║                   ║
    ....Dtl........         Dtr
        ║          :        ║
        env_l ═ Dl :Dr ══ env_r
        ║          :        ║
        Dbl        :........Dbr....
        ║                   ║
        cbl ═════ Db ═════ cbr
    =#
    @autoopt @tensor env_l[Dbr1 Dbr0 Dl1 Dl0 Dtl1 Dtl0] :=
        cbr[Dbr1 Dbr0 Db1 Db0] * cbl[Dbl1 Dbl0 Db1 Db0] * env_l[Dtl1 Dtl0 Dl1 Dl0 Dbl1 Dbl0]
    @autoopt @tensor env_r[Dtl1 Dtl0 Dr1 Dr0 Dbr1 Dbr0] :=
        ctl[Dt1 Dt0 Dtl1 Dtl0] * ctr[Dtr1 Dtr0 Dt1 Dt0] * env_r[Dtr1 Dtr0 Dbr1 Dbr0 Dr1 Dr0]
    @tensor env[Dl1 Dr1; Dl0 Dr0] :=
        env_l[Dbr1 Dbr0 Dl1 Dl0 Dtl1 Dtl0] * env_r[Dtl1 Dtl0 Dr1 Dr0 Dbr1 Dbr0]
    # normalize `env`
    return env / norm(env, Inf)
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
struct NTUEnvNNN <: NTUEnvAlgorithm end
"""
Calculates the bond environment within "NTU-NNN" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::PEPSOrth, Y::PEPSOrth, peps::InfinitePEPS, ::NTUEnvNNN
)
    neighbors = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (1, 2),
        (0, 2),
        (-1, 2),
        (-1, 1),
        (-1, 0),
    ]
    m = collect_neighbors(peps, row, col, neighbors)
    #= left half
        (-1 -1)══════(-1 +0)═ -1/-2
            ║           ║
            ║           ║
        (+0 -1)════════ X ═══ -3/-4
            ║           ║
            D1          D2
            ║           ║
        (+1 -1)═ D3 ═(+1 +0)═ -5/-6
    =#
    vecl = enlarge_corner_tl(cor_tl(m[-1, -1]), edge_t(m[-1, 0]), edge_l(m[0, -1]), X)
    @tensor vecl[:] :=
        cor_bl(m[1, -1])[D11 D10 D31 D30] *
        edge_b(m[1, 0])[D21 D20 -5 -6 D31 D30] *
        vecl[D11 D10 D21 D20 -1 -2 -3 -4]
    #= right half
        -1/-2 ══ (-1 +1)═ D1 ═(-1 +2)
                    ║           ║
                    D2          D3
                    ║           ║
        -3/-4 ═════ Y ═══════(+0 +2)
                    ║           ║
                    ║           ║     
        -5/-6 ══ (+1 +1)═════(+1 +2)
    =#
    vecr = enlarge_corner_br(cor_br(m[1, 2]), edge_b(m[1, 1]), edge_r(m[0, 2]), Y)
    @tensor vecr[:] :=
        edge_t(m[-1, 1])[D11 D10 D21 D20 -1 -2] *
        cor_tr(m[-1, 2])[D31 D30 D11 D10] *
        vecr[D21 D20 D31 D30 -3 -4 -5 -6]
    # combine left and right part
    return @tensor g[-1 -2; -3 -4] := vecl[1 2 -1 -3 3 4] * vecr[1 2 -2 -4 3 4]
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
struct NTUEnvNNNp <: NTUEnvAlgorithm end
"""
Calculates the bond environment within "NTU-NNN+" approximation.
"""
function bondenv_ntu(
    row::Int, col::Int, X::PEPSOrth, Y::PEPSOrth, peps::InfinitePEPS, ::NTUEnvNNNp
)
    return error("Not implemented")
end

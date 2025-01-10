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
    @autoopt @tensor env_l[Dbr1 Dl1 Dtl1 Dbr0 Dl0 Dtl0] :=
        cbr[Dbr1 Db1 Dbr0 Db0] * cbl[Dbl1 Db1 Dbl0 Db0] * env_l[Dtl1 Dl1 Dbl1 Dtl0 Dl0 Dbl0]
    @autoopt @tensor env_r[Dtl1 Dr1 Dbr1 Dtl0 Dr0 Dbr0] :=
        ctl[Dt1 Dtl1 Dt0 Dtl0] * ctr[Dtr1 Dt1 Dtr0 Dt0] * env_r[Dtr1 Dbr1 Dr1 Dtr0 Dbr0 Dr0]
    @autoopt @tensor env[Dl1 Dr1; Dl0 Dr0] :=
        env_l[Dbr1 Dl1 Dtl1 Dbr0 Dl0 Dtl0] * env_r[Dtl1 Dr1 Dbr1 Dtl0 Dr0 Dbr0]
    # normalize `env`
    return env / (norm(env, Inf) / 5.0)
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
    return error("Not implemented")
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

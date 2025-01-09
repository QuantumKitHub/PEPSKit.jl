"""
    bondenv_NN(peps::InfinitePEPS, row::Int, col::Int, Q0::PEPSTensor, Q1::PEPSTensor)

Calculate the bond environment within "NTU-NN" approximation.
```
            (-1 +0)══(-1 +1)
                ║        ║
    (+0 -1)════Q0══   ══Q1═══(+0 +2)
                ║        ║
            (+1 +0)══(+1 +1)
```
"""
function bondenv_NN(peps::InfinitePEPS, row::Int, col::Int, Q0::PEPSTensor, Q1::PEPSTensor)
    neighbors = [(-1, 0), (0, -1), (1, 0), (1, 1), (0, 2), (-1, 1)]
    m = collect_neighbors(peps, row, col, neighbors)
    env_l = edge_l(Q0, hair_l(m[0, -1]))
    env_r = edge_r(Q1, hair_r(m[0, 2]))
    ctl = cor_tl(m[-1, 0])
    ctr = cor_tr(m[-1, 1])
    cbr = cor_br(m[1, 1])
    cbl = cor_bl(m[1, 0])
    #= contraction indices

        ctl ═════ Dt ═════ ctr
        ║                   ║
        Dtl                 Dtr
        ║                   ║
        env_l ═ Dl  Dr ══ env_r
        ║                   ║
        Dbl                 Dbr
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
    bondenv_NNN(peps::InfinitePEPS, row::Int, col::Int)

Calculates the bond environment within "NTU-NNN" approximation.

```
    (-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)
        ║       ║        ║       ║
    (+0 -1)════Q0══   ══Q1═══(+0 +2)
        ║       ║        ║       ║
    (+1 -1)=(+1 +0)══(+1 +1)=(+1 +2)
```
"""
function bondenv_NNN(peps::InfinitePEPS, row::Int, col::Int)
    return error("Not implemented")
end

"""
    bondenv_NNNp(peps::InfinitePEPS, row::Int, col::Int)

Calculates the bond environment within "NTU-NNN+" approximation.
```
            (-2 -1) (-2 +0)  (-2 +1) (-2 +2)
                ║       ║        ║       ║
    (-1 -2)=(-1 -1)=(-1 +0)══(-1 +1)=(-1 +2)═(-1 +3)
                ║       ║        ║       ║
    (+0 -2)=(+0 -1)════Q0══   ══Q1═══(+0 +2)═(+0 +3)
                ║       ║        ║       ║
    (+1 -2)=(+1 -1)=(+1 +0)══(+1 +1)═(+1 +2)═(+1 +3)
                ║       ║        ║       ║
            (+2 -1) (+2 +0)  (+2 +1) (+2 +2)
```
"""
function bondenv_NNNp(peps::InfinitePEPS, row::Int, col::Int)
    return error("Not implemented")
end

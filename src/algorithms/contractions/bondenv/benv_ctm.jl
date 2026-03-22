"""
Construct the environment (norm) tensor
```
    C1---T1---------T1---C2   r-1
    |    ‖          ‖    |
    T4===XX==     ==YY===T2    r
    |    ‖          ‖    |
    C4---T3---------T3---C3   r+1
    c-1  c         c+1  c+2
```
where `XX = X' X` and `YY = Y' Y` (stacked together).

Axis order: `[DX1 DY1; DX0 DY0]`, as in
```
    ┌---------------------┐
    | ┌----┐              |
    └-|    |---DX0  DY0---┘
      |benv|
    ┌-|    |---DX1  DY1---┐
    | └----┘              |
    └---------------------┘
```
"""
function bondenv_ctm(row::Int, col::Int, X::PEPSOrth, Y::PEPSOrth, env::CTMRGEnv)
    Nr, Nc = size(env.corners)[[2, 3]]
    cm1 = _prev(col, Nc)
    cp1 = _next(col, Nc)
    cp2 = _next(cp1, Nc)
    rm1 = _prev(row, Nr)
    rp1 = _next(row, Nr)
    c1 = env.corners[1, rm1, cm1]
    c2 = env.corners[2, rm1, cp2]
    c3 = env.corners[3, rp1, cp2]
    c4 = env.corners[4, rp1, cm1]
    t1X, t1Y = env.edges[1, rm1, col], env.edges[1, rm1, cp1]
    t2 = env.edges[2, row, cp2]
    t3X, t3Y = env.edges[3, rp1, col], env.edges[3, rp1, cp1]
    t4 = env.edges[4, row, cm1]
    #= index labels

    C1--χ4--T1X---------χ6---------T1Y--χ8---C2     r-1
    |        ‖                      ‖        |
    χ2      DNX                    DNY      χ10
    |        ‖                      ‖        |
    T4==DWX==XX===DX==       ==DY===YY==DEY==T2     r
    |        ‖                      ‖        |
    χ1      DSX                    DSY       χ9
    |        ‖                      ‖        |
    C4--χ3--T3X---------χ5---------T3Y--χ7---C3     r+1
    c-1      c                      c+1     c+2
    =#
    @autoopt @tensor benv[DX1 DY1; DX0 DY0] :=
        c4[χ3 χ1] * t4[χ1 DWX0 DWX1 χ2] * c1[χ2 χ4] * t3X[χ5 DSX0 DSX1 χ3] *
        X[DNX0 DX0 DSX0 DWX0] * conj(X[DNX1 DX1 DSX1 DWX1]) * t1X[χ4 DNX0 DNX1 χ6] *
        c3[χ9 χ7] * t2[χ10 DEY0 DEY1 χ9] * c2[χ8 χ10] * t3Y[χ7 DSY0 DSY1 χ5] *
        Y[DNY0 DEY0 DSY0 DY0] * conj(Y[DNY1 DEY1 DSY1 DY1]) * t1Y[χ6 DNY0 DNY1 χ8]

    normalize!(benv, Inf)
    return benv
end

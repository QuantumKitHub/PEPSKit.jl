"""
Construct the environment (norm) tensor
```
    left half                       right half
    C1 -χ4 - T1 ------- χ6 ------- T1 - χ8 - C2     r-1
    |        ‖                      ‖        |
    χ2      DNX                    DNY      χ10
    |        ‖                      ‖        |
    T4 =DWX= XX = DX =       = DY = YY =DEY= T2     r
    |        ‖                      ‖        |
    χ1      DSX                    DSY       χ9
    |        ‖                      ‖        |
    C4 -χ3 - T3 ------- χ5 ------- T3 - χ7 - C3     r+1
    c-1      c                      c+1     c+2
```
which can be more simply denoted as
```
    |------------|
    |→ DX1  DY1 ←|   axis order
    |← DX0  DX1 →|   (DX1, DY1, DX0, DY0)
    |------------|
```
The axes 1, 2 (or 3, 4) come from X†, Y† (or X, Y)
"""
function bondenv_fu(row::Int, col::Int, X::PEPSOrth, Y::PEPSOrth, envs::CTMRGEnv)
    Nr, Nc = size(envs.corners)[[2, 3]]
    cm1 = _prev(col, Nc)
    cp1 = _next(col, Nc)
    cp2 = _next(cp1, Nc)
    rm1 = _prev(row, Nr)
    rp1 = _next(row, Nr)
    c1 = envs.corners[1, rm1, cm1]
    c2 = envs.corners[2, rm1, cp2]
    c3 = envs.corners[3, rp1, cp2]
    c4 = envs.corners[4, rp1, cm1]
    t1X, t1Y = envs.edges[1, rm1, col], envs.edges[1, rm1, cp1]
    t2 = envs.edges[2, row, cp2]
    t3X, t3Y = envs.edges[3, rp1, col], envs.edges[3, rp1, cp1]
    t4 = envs.edges[4, row, cm1]
    # left half
    @autoopt @tensor lhalf[DX1, DX0, χ5, χ6] := (
        c4[χ3, χ1] *
        t4[χ1, DWX0, DWX1, χ2] *
        c1[χ2, χ4] *
        t3X[χ5, DSX0, DSX1, χ3] *
        X[DNX0, DX0, DSX0, DWX0] *
        conj(X[DNX1, DX1, DSX1, DWX1]) *
        t1X[χ4, DNX0, DNX1, χ6]
    )
    # right half
    @autoopt @tensor rhalf[DY1, DY0, χ5, χ6] := (
        c3[χ9, χ7] *
        t2[χ10, DEY0, DEY1, χ9] *
        c2[χ8, χ10] *
        t3Y[χ7, DSY0, DSY1, χ5] *
        Y[DNY0, DEY0, DSY0, DY0] *
        conj(Y[DNY1, DEY1, DSY1, DY1]) *
        t1Y[χ6, DNY0, DNY1, χ8]
    )
    # combine
    @autoopt @tensor env[DX1, DY1; DX0, DY0] := (
        lhalf[DX1, DX0, χ5, χ6] * rhalf[DY1, DY0, χ5, χ6]
    )
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    return env / norm(env, Inf)
end

"""
Replace `env` by its positive/negative approximant `± Z Z†`
(returns the sign and Z†)
```
                        |-→ 1   2 ←-|
                        |           |
    |----env----|       |←--- Z ---→|
    |→ 1     2 ←|   =         ↑
    |← 3     4 →|       |---→ Z† ←--|
    |-----------|       |           |
                        |←- 3   4 -→|
```
"""
function positive_approx(env::BondEnv)
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    # hermitize env, and perform eigen-decomposition
    # env = U D U'
    D, U = eigh((env + env') / 2)
    # determine env is (mostly) positive or negative
    sgn = sign(mean(vcat((diag(b) for (k, b) in blocks(D))...)))
    if sgn == -1
        D *= -1
    end
    # set negative eigenvalues to 0
    for (k, b) in blocks(D)
        for i in diagind(b)
            if b[i] < 0
                b[i] = 0.0
            end
        end
    end
    Zdg = sdiag_pow(D, 0.5) * U'
    return sgn, Zdg
end

"""
Fix local gauge of the env tensor around a bond
"""
function fu_fixgauge(
    Zdg::AbstractTensorMap{S,1,2}, X::PEPSOrth, Y::PEPSOrth, aR::BondPhys, bL::BondPhys
) where {S<:ElementarySpace}
    #= 
            1               1
            ↑               ↑
        2 → Z† ← 3  =   2 → QR ← 3  1 ← R ← 2

                                        1
                                        ↑
                    =   2 → L → 1   3 → QL ← 2
    =#
    QR, R = leftorth(Zdg, ((1, 2), (3,)); alg=QRpos())
    QL, L = leftorth(Zdg, ((1, 3), (2,)); alg=QRpos())
    @assert !isdual(codomain(R)[1]) && !isdual(domain(R)[1])
    @assert !isdual(codomain(L)[1]) && !isdual(domain(L)[1])
    Rinv, Linv = inv(R), inv(L)
    #= fix gauge of aR, bL, Z†

                    ↑
        |→-(Linv -→ Z† ← Rinv)←-|
        |                       |
        ↑                       ↑
        |        ↑     ↑        |
        |← (L ← aR) ← (bL → R) →|
        |-----------------------|

                     -2              -2
                      ↑               ↑        
        -1 ← L ← 1 ← aR2 ← -3   -1 ← bL2 → 1 → R → -3

                        -1
                        ↑
        -2 → Linv → 1 → Z† ← 2 ← Rinv ← -3
    =#
    aR = ncon([L, aR], [[-1, 1], [1, -2, -3]])
    bL = ncon([bL, R], [[-1, -2, 1], [-3, 1]])
    Zdg = permute(ncon([Zdg, Linv, Rinv], [[-1, 1, 2], [1, -2], [2, -3]]), (1,), (2, 3))
    #= fix gauge of X, Y

            -1                                      -1
             |                                      |
        -4 - X ← 1 ← Linv ← -2      -4 → Rinv → 1 → Y - -2
             |                                      |
            -3                                      -3
    =#
    X = ncon([X, Linv], [[-1, 1, -3, -4], [1, -2]])
    Y = ncon([Y, Rinv], [[-1, -2, -3, 1], [1, -4]])
    return Zdg, X, Y, aR, bL
end

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
function positive_approx(env::AbstractTensorMap)
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
    Zdg::AbstractTensorMap,
    X::AbstractTensorMap,
    Y::AbstractTensorMap,
    aR::AbstractTensorMap,
    bL::AbstractTensorMap,
)
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

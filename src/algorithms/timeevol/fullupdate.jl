include("fu_gaugefix.jl")
include("fu_optimize.jl")

"""
CTMRG left-move to update CTMRGEnv in the c-th column 
```
    ---> absorb
    C1 ← T1 ←   r-1
    ↓    ‖
    T4 = M' =   r
    ↓    ‖
    C4 → T3 →   r+1
    c-1  c 
```
"""
function ctmrg_leftmove!(
    col::Int, peps::InfinitePEPS, envs::CTMRGEnv, chi::Int, svderr::Float64=1e-9
)
    trscheme = truncerr(svderr) & truncdim(chi)
    alg = CTMRG(;
        verbosity=0, miniter=1, maxiter=10, trscheme=trscheme, ctmrgscheme=:sequential
    )
    envs2, info = ctmrg_leftmove(col, peps, envs, alg)
    envs.corners[:, :, col] = envs2.corners[:, :, col]
    envs.edges[:, :, col] = envs2.edges[:, :, col]
    return info
end

"""
CTMRG right-move to update CTMRGEnv in the c-th column
```
    absorb <---
    ←-- T1 ← C2     r-1
        ‖    ↑
    === M' = T2     r
        ‖    ↑
    --→ T3 → C3     r+1
        c   c+1
```
"""
function ctmrg_rightmove!(
    col::Int, peps::InfinitePEPS, envs::CTMRGEnv, chi::Int, svderr::Float64=1e-9
)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    PEPSKit.rot180!(envs)
    ctmrg_leftmove!(Nc + 1 - col, rot180(peps), envs, chi, svderr)
    PEPSKit.rot180!(envs)
    return nothing
end

"""
Update all horizontal bonds in the c-th column
(i.e. `(r,c) (r,c+1)` for all `r = 1, ..., Nr`).
To update rows, rotate the network clockwise by 90 degrees.
"""
function update_column!(
    col::Int,
    gate::AbstractTensorMap,
    peps::InfinitePEPS,
    envs::CTMRGEnv,
    Dcut::Int,
    chi::Int;
    svderr::Float64=1e-9,
    maxiter::Int=50,
    maxdiff::Float64=1e-15,
    gaugefix::Bool=true,
)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    localfid = 0.0
    costs = zeros(Nr)
    truncscheme = truncerr(svderr) & truncdim(Dcut)
    #= Axis order of X, aR, Y, bL

            1             2            2         1
            |            ↗           ↗           |
        4 - X ← 2   1 ← aR ← 3  1 ← bL → 3   4 → Y - 2
            |                                    |
            3                                    3
    =#
    for row in 1:Nr
        cp1 = _next(col, Nc)
        A, B = peps[row, col], peps[row, cp1]
        # TODO: relax dual requirement on the bonds
        @assert !isdual(domain(A)[2])
        #= QR and LQ decomposition

            2   1               1             2
            | ↗                 |            ↗
        5 - A ← 3   ====>   4 - X ← 2   1 ← aR ← 3
            |                   |
            4                   3
        =#
        X, aR0 = leftorth(A, ((2, 4, 5), (1, 3)); alg=QRpos())
        X = permute(X, (1, 4, 2, 3))
        #=
            2   1                 2         2
            | ↗                 ↗           |
        5 → B - 3   ====>  1 ← bL → 3   1 → Y - 3
            |                               |
            4                               4
        =#
        Y, bL0 = leftorth(B, ((2, 3, 4), (1, 5)); alg=QRpos())
        bL0 = permute(bL0, (3, 2, 1))
        env = tensor_env(row, col, X, Y, envs)
        # positive/negative-definite approximant: env = ± Z Z†
        sgn, Zdg = positive_approx(env)
        # fix gauge
        if gaugefix
            Zdg, X, Y, aR0, bL0 = fu_fixgauge(Zdg, X, Y, aR0, bL0)
        end
        env = sgn * Zdg' * Zdg
        #= apply gate

                -2          -3
                ↑           ↑
                |----gate---|
                ↑           ↑
                1           2
                ↑           ↑
            -1← aR -← 3 -← bL → -4
        =#
        tmp = ncon((gate, aR0, bL0), ([-2, -3, 1, 2], [-1, 1, 3], [3, 2, -4]))
        # initialize truncated tensors using simple SVD truncation
        # TODO: return truncated and untruncated SVD result at once, without repeated calculation
        aR2, s, bL2, ϵ = tsvd(tmp, ((1, 2), (3, 4)); trunc=truncerr(1e-15))
        aR, s_cut, bL, ϵ = tsvd(tmp, ((1, 2), (3, 4)); trunc=truncscheme)
        aR2, bL2 = absorb_s(aR2, s, bL2)
        aR, bL = absorb_s(aR, s_cut, bL)
        # optimize aR, bL
        aR, bL, cost = fu_optimize(
            aR, bL, aR2, bL2, env; maxiter=maxiter, maxdiff=maxdiff, verbose=false
        )
        costs[row] = cost
        aR /= maxabs(aR)
        bL /= maxabs(bL)
        localfid += local_fidelity(aR, bL, aR0, bL0)
        #= update and normalize peps, ms

                -2        -1               -1     -2
                |        ↗                ↗       |
            -5- X ← 1 ← aR ← -3     -5 ← bL → 1 → Y - -3
                |                                 |
                -4                                -4
        =#
        peps.A[row, col] = permute(
            ncon([X, aR], [[-2, 1, -4, -5], [1, -1, -3]]), (1,), Tuple(2:5)
        )
        peps.A[row, cp1] = permute(
            ncon([bL, Y], [[-5, -1, 1], [-2, -3, -4, 1]]), (1,), Tuple(2:5)
        )
        # normalize
        for c_ in [col, cp1]
            peps.A[row, c_] /= maxabs(peps.A[row, c_])
        end
    end
    # update CTMRGEnv
    ctmrg_leftmove!(col, peps, envs, chi, svderr)
    ctmrg_rightmove!(_next(col, Nc), peps, envs, chi, svderr)
    return localfid, costs
end

"""
One round of full update on the input InfinitePEPS `peps` and its CTMRGEnv `envs`

When `cheap === true`, use half-infinite environment to construct CTMRG projectors.
Otherwise, use full-infinite environment instead.

Reference: Physical Review B 92, 035142 (2015)
"""
function fullupdate!(
    gate::AbstractTensorMap,
    peps::InfinitePEPS,
    envs::CTMRGEnv,
    Dcut::Int,
    chi::Int,
    svderr::Float64=1e-9,
)
    Nr, Nc = size(peps)
    fid, maxcost = 0.0, 0.0
    for col in 1:Nc
        tmpfid, costs = update_column!(col, gate, peps, envs, Dcut, chi; svderr=svderr)
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    rotr90!(peps)
    rotr90!(envs)
    for row in 1:Nr
        tmpfid, costs = update_column!(row, gate, peps, envs, Dcut, chi; svderr=svderr)
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    rotl90!(peps)
    rotl90!(envs)
    fid /= (2 * Nr * Nc)
    return fid, maxcost
end

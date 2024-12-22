include("fu_gaugefix.jl")
include("fu_optimize.jl")

"""
Algorithm struct for full update (FU) of infinite PEPS with bond weights/
Each FU run stops when the energy starts to increase.
"""
@kwdef struct FullUpdate
    dt::Float64
    maxiter::Int
    fixgauge::Bool = true
    # truncation scheme after applying gate
    trscheme::TensorKit.TruncationScheme
    # alternating least square optimization
    opt_alg::FUALSOptimize = FUALSOptimize()
    # SequentialCTMRG column move after updating a column of bonds
    colmove_alg::SequentialCTMRG
    # interval to reconverge environments
    reconv_int::Int = 10
    # CTMRG for reconverging environment
    reconv_alg::CTMRGAlgorithm
end

function truncation_scheme(alg::FullUpdate, v::ElementarySpace)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(v)
    else
        return alg.trscheme
    end
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
function ctmrg_rightmove(col::Int, peps::InfinitePEPS, envs::CTMRGEnv, alg::SequentialCTMRG)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    envs, info = ctmrg_leftmove(Nc + 1 - col, rot180(peps), rot180(envs), alg)
    return rot180(envs), info
end

"""
Update all horizontal bonds in the c-th column
(i.e. `(r,c) (r,c+1)` for all `r = 1, ..., Nr`).
To update rows, rotate the network clockwise by 90 degrees.
"""
function update_column!(
    col::Int, gate::LocalOperator, peps::InfinitePEPS, envs::CTMRGEnv, alg::FullUpdate
)
    Nr, Nc = size(peps)
    @assert 1 <= col <= Nc
    localfid = 0.0
    costs = zeros(Nr)
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
        if alg.fixgauge
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
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        aR2bL2 = ncon((term, aR0, bL0), ([-2, -3, 1, 2], [-1, 1, 3], [3, 2, -4]))
        # initialize truncated tensors using SVD truncation
        aR, s_cut, bL, ϵ = tsvd(
            aR2bL2, ((1, 2), (3, 4)); trunc=truncation_scheme(alg, space(aR0, 3))
        )
        aR, bL = absorb_s(aR, s_cut, bL)
        # optimize aR, bL
        aR, bL, cost = fu_optimize(aR, bL, aR2bL2, env, alg.opt_alg)
        costs[row] = cost
        aR /= norm(aR, Inf)
        bL /= norm(bL, Inf)
        localfid += local_fidelity(aR, bL, _combine_aRbL(aR0, bL0))
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
            peps.A[row, c_] /= norm(peps.A[row, c_], Inf)
        end
    end
    # update CTMRGEnv
    envs2, info = ctmrg_leftmove(col, peps, envs, alg.colmove_alg)
    envs2, info = ctmrg_rightmove(_next(col, Nc), peps, envs2, alg.colmove_alg)
    for c in [col, _next(col, Nc)]
        envs.corners[:, :, c] = envs2.corners[:, :, c]
        envs.edges[:, :, c] = envs2.edges[:, :, c]
    end
    return localfid, costs
end

"""
One round of full update on the input InfinitePEPS `peps` and its CTMRGEnv `envs`

Reference: Physical Review B 92, 035142 (2015)
"""
function fu_iter(gate::LocalOperator, peps::InfinitePEPS, envs::CTMRGEnv, alg::FullUpdate)
    Nr, Nc = size(peps)
    fid, maxcost = 0.0, 0.0
    peps2, envs2 = deepcopy(peps), deepcopy(envs)
    for col in 1:Nc
        tmpfid, costs = update_column!(col, gate, peps2, envs2, alg)
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    peps2, envs2 = rotr90(peps2), rotr90(envs2)
    gate_rotated = rotr90(gate)
    for row in 1:Nr
        tmpfid, costs = update_column!(row, gate_rotated, peps2, envs2, alg)
        fid += tmpfid
        maxcost = max(maxcost, maximum(costs))
    end
    peps2, envs2 = rotl90(peps2), rotl90(envs2)
    fid /= (2 * Nr * Nc)
    return peps2, envs2, (fid, maxcost)
end

"""
Perform full update with nearest neighbor Hamiltonian `ham`.
After FU stops, the final environment is calculated with CTMRG algorithm `ctm_alg`.
"""
function fullupdate(
    peps::InfinitePEPS,
    envs::CTMRGEnv,
    ham::LocalOperator,
    fu_alg::FullUpdate,
    ctm_alg::CTMRGAlgorithm,
)
    time_start = time()
    N1, N2 = size(peps)
    @printf(
        "%-4s %7s%10s%12s%11s  %s/%s\n",
        "step",
        "dt",
        "energy",
        "Δe",
        "svd_diff",
        "speed",
        "meas(s)"
    )
    gate = get_gate(fu_alg.dt, ham)
    esite0, peps0, envs0 = Inf, deepcopy(peps), deepcopy(envs)
    diff_energy = 0.0
    for count in 1:(fu_alg.maxiter)
        time0 = time()
        peps, envs, (fid, cost) = fu_iter(gate, peps, envs, fu_alg)
        time1 = time()
        if count == 1 || count % fu_alg.reconv_int == 0
            meast0 = time()
            # reconverge `env` (in place)
            println(stderr, "---- FU step $count: reconverging envs ----")
            envs = leading_boundary(envs, peps, fu_alg.reconv_alg)
            esite = costfun(peps, envs, ham) / (N1 * N2)
            meast1 = time()
            # monitor change of CTMRGEnv by its singular values
            diff_energy = esite - esite0
            diff_ctm = calc_convergence(envs, envs0)[1]
            @printf(
                "%-4d %7.0e%10.5f%12.3e%11.3e  %.3f/%.3f\n",
                count,
                fu_alg.dt,
                esite,
                diff_energy,
                diff_ctm,
                time1 - time0,
                meast1 - meast0
            )
            if diff_energy > 0
                @printf("Energy starts to increase. Abort evolution.\n")
                # restore peps and envs at last checking
                peps, envs = peps0, envs0
                break
            end
            esite0, peps0, envs0 = esite, deepcopy(peps), deepcopy(envs)
        end
    end
    # reconverge the environment tensors
    for io in (stdout, stderr)
        @printf(io, "Reconverging final envs ... \n")
    end
    envs = leading_boundary(envs, peps, ctm_alg)
    time_end = time()
    @printf("Evolution time: %.3f s\n\n", time_end - time_start)
    print(stderr, "\n----------\n\n")
    return peps, envs, (esite0, diff_energy)
end

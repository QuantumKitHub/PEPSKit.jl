"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct NTUpdate <: TimeEvolAlgorithm
    dt::Float64
    maxiter::Int
    # algorithm to construct bond environment (metric)
    bondenv_alg::BondEnvAlgorithm
    # alternating least square optimization
    # opt_alg::ALSOptimize = ALSOptimize()
    opt_alg::FullEnvTruncation
    # monitor energy every `ctm_int` steps
    ctm_int::Int = 10
    # CTMRG algorithm to monitor energy
    ctm_alg::CTMRGAlgorithm
end

"""
Neighborhood tensor update for the bond between sites `[row, col]` and `[row, col+1]`.
"""
function _ntu_bondx!(
    row::Int, col::Int, gate::AbstractTensorMap{S,2,2}, peps::InfinitePEPS, alg::NTUpdate
) where {S<:ElementarySpace}
    Nr, Nc = size(peps)
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
    aR0 = permute(aR0, (1, 2, 3))
    #=
        2   1                 2         2
        | ↗                 ↗           |
    5 ← B - 3   ====>  1 ← bL → 3   1 → Y - 3
        |                               |
        4                               4
    =#
    Y, bL0 = leftorth(B, ((2, 3, 4), (1, 5)); alg=QRpos())
    Y = permute(Y, (1, 2, 3, 4))
    bL0 = permute(bL0, (3, 2, 1))
    env = bondenv_ntu(row, col, X, Y, peps, alg.bondenv_alg)
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    #= apply gate

            -2          -3
            ↑           ↑
            |----gate---|
            ↑           ↑
            1           2
            ↑           ↑
        -1← aR -← 3 -← bL → -4
    =#
    aR2bL2 = ncon((gate, aR0, bL0), ([-2, -3, 1, 2], [-1, 1, 3], [3, 2, -4]))
    # initialize aR, bL using un-truncated SVD
    aR, s_cut, bL, ϵ = tsvd(
        aR2bL2,
        ((1, 2), (3, 4));
        trunc=truncerr(1e-15),
        # trunc=truncation_scheme(alg, space(aR0, 3))
    )
    aR, bL = absorb_s(aR, s_cut, bL)
    aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
    # optimize aR, bL
    # aR, bL, cost = als_optimize(aR, bL, aR2bL2, env, alg.opt_alg)
    aR, bL, (cost, fid) = bond_optimize(env, aR, bL, alg.opt_alg)
    aR /= norm(aR, Inf)
    bL /= norm(bL, Inf)
    # fid = local_fidelity(_combine_aRbL(aR, bL), _combine_aRbL(aR0, bL0))
    #= update and normalize peps, ms

            -2        -1               -1     -2
            |        ↗                ↗       |
        -5- X ← 1 ← aR ← -3     -5 ← bL → 1 → Y - -3
            |                                 |
            -4                                -4
    =#
    @tensor A[-1; -2 -3 -4 -5] := X[-2, 1, -4, -5] * aR[1, -1, -3]
    @tensor B[-1; -2 -3 -4 -5] := bL[-5, -1, 1] * Y[-2, -3, -4, 1]
    peps.A[row, col] = A / norm(A, Inf)
    peps.A[row, cp1] = B / norm(B, Inf)
    return cost, fid
end

"""
One round of update of all nearest neighbor bonds in InfinitePEPS

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function ntu_iter(gate::LocalOperator, peps::InfinitePEPS, alg::NTUpdate)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.A[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
    end
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
        for site in CartesianIndices(peps2.A)
            r, c = Tuple(site)
            term = get_gateterm(
                direction == 1 ? gate : gate_mirrored,
                (CartesianIndex(r, c), CartesianIndex(r, c + 1)),
            )
            ϵ = _ntu_bondx!(r, c, term, peps2, alg)
        end
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
    end
    return peps2
end

"""
Perform NTU on InfinitePEPS with nearest neighbor Hamiltonian `ham`. 
"""
function ntupdate(
    peps::InfinitePEPS,
    envs::CTMRGEnv,
    ham::LocalOperator,
    alg::NTUpdate,
    ctm_alg::CTMRGAlgorithm,
)
    time_start = time()
    Nr, Nc = size(peps)
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
    gate = get_gate(alg.dt, ham)
    peps0, envs0 = deepcopy(peps), deepcopy(envs)
    esite0, diff_energy = Inf, 0.0
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = ntu_iter(gate, peps, alg)
        time1 = time()
        if count == 1 || count % alg.ctm_int == 0
            # monitor energy
            meast0 = time()
            envs = leading_boundary(envs, peps, alg.ctm_alg)
            esite = costfun(peps, envs, ham) / (Nr * Nc)
            meast1 = time()
            # monitor change of CTMRGEnv by its singular values
            diff_energy = esite - esite0
            diff_ctm = calc_convergence(envs, envs0)[1]
            @printf(
                "%-4d %7.0e%10.5f%12.3e%11.3e  %.3f/%.3f\n",
                count,
                alg.dt,
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

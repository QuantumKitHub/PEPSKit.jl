"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct NTUpdate
    dt::Float64
    maxiter::Int
    # maximum weight difference for convergence
    tol::Float64
    # algorithm to construct bond environment (metric)
    bondenv_alg::BondEnvAlgorithm
    # bond truncation after applying time evolution gate
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
    row::Int,
    col::Int,
    gate::AbstractTensorMap{S,2,2},
    peps::InfiniteWeightPEPS,
    alg::NTUpdate,
) where {S<:ElementarySpace}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    cp1 = _next(col, Nc)
    # TODO: relax dual requirement on the bonds
    A, B = peps.vertices[row, col], peps.vertices[row, cp1]
    @assert !isdual(domain(A)[2])
    A = _absorb_weight(A, row, col, "", peps.weights)
    B = _absorb_weight(B, row, cp1, "", peps.weights)
    #= QR and LQ decomposition

        2   1               1             2
        | ↗                 |            ↗
    5 - A ← 3   ====>   4 - X ← 2   1 ← aR ← 3
        |                   |
        4                   3
    =#
    X, aR = leftorth(A, ((2, 4, 5), (1, 3)); alg=QRpos())
    X, aR = permute(X, (1, 4, 2, 3)), permute(aR, (1, 2, 3))
    #=
        2   1                 2         2
        | ↗                 ↗           |
    5 ← B - 3   ====>  1 ← bL → 3   1 → Y - 3
        |                               |
        4                               4
    =#
    Y, bL = leftorth(B, ((2, 3, 4), (1, 5)); alg=QRpos())
    bL, Y = permute(bL, (3, 2, 1)), permute(Y, (1, 2, 3, 4))
    env = bondenv_ntu(row, col, X, Y, peps, alg.bondenv_alg)
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    #= apply gate

            -2         -3
            ↑           ↑
            |----gate---|
            ↑           ↑
            1           2
            ↑           ↑
        -1← aR -← 3 -← bL ← -4
    =#
    @tensor aR2bL2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * aR[-1 1 3] * bL[3 2 -4]
    # initialize aR, bL using un-truncated SVD
    aR, s, bL, ϵ = tsvd(aR2bL2; trunc=truncerr(1e-15))
    aR, bL = absorb_s(aR, s, bL)
    aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
    # optimize aR, bL
    aR, s, bL, (cost, fid) = bond_optimize(env, aR, bL, alg.opt_alg)
    #= update and normalize peps, ms

            -2        -1               -1     -2
            |        ↗                ↗       |
        -5- X ← 1 ← aR ← -3     -5 ← bL → 1 → Y - -3
            |                                 |
            -4                                -4
    =#
    @tensor A[-1; -2 -3 -4 -5] := X[-2, 1, -4, -5] * aR[1, -1, -3]
    @tensor B[-1; -2 -3 -4 -5] := bL[-5, -1, 1] * Y[-2, -3, -4, 1]
    # remove bond weights
    for ax in (2, 4, 5)
        A = absorb_weight(A, row, col, ax, peps.weights; sqrtwt=true, invwt=true)
    end
    for ax in (2, 3, 4)
        B = absorb_weight(B, row, cp1, ax, peps.weights; sqrtwt=true, invwt=true)
    end
    peps.vertices[row, col] = A / norm(A, Inf)
    peps.vertices[row, cp1] = B / norm(B, Inf)
    peps.weights[1, row, col] = s / norm(s, Inf)
    return cost, fid
end

"""
    ntu_iter(gate::LocalOperator, peps::InfiniteWeightPEPS, alg::NTUpdate; bipartite::Bool=false)

One round of neighborhood tensor update on `peps` applying the nearest neighbor `gate`.

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function ntu_iter(
    gate::LocalOperator, peps::InfiniteWeightPEPS, alg::NTUpdate; bipartite::Bool=false
)
    @assert size(gate.lattice) == size(peps)
    Nr, Nc = size(peps)
    # TODO: make algorithm independent on the choice of dual in the network
    for (r, c) in Iterators.product(1:Nr, 1:Nc)
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights[1, r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights[2, r, c], ax)) for ax in 1:2] == [0, 1]
    end
    peps2 = deepcopy(peps)
    gate_mirrored = mirror_antidiag(gate)
    for direction in 1:2
        if direction == 2
            peps2 = mirror_antidiag(peps2)
        end
        for site in CartesianIndices(peps2.vertices)
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
    for s in peps2.weights.data
        @assert norm(s, Inf) ≈ 1.0
    end
    return peps2
end

"""
Perform NTU on InfiniteWeightPEPS with nearest neighbor Hamiltonian `ham`. 

If `bipartite == true` (for square lattice), a unit cell size of `(2, 2)` is assumed, 
as well as tensors and x/y weights which are the same across the diagonals, i.e. at
`(row, col)` and `(row+1, col+1)`.
"""
function ntupdate(
    peps::InfiniteWeightPEPS,
    envs::CTMRGEnv,
    ham::LocalOperator,
    alg::NTUpdate,
    ctm_alg::CTMRGAlgorithm;
    bipartite::Bool=false,
)
    time_start = time()
    Nr, Nc = size(peps)
    if bipartite
        error("Not implemented")
    end
    @info @sprintf(
        "%-4s %7s%10s%12s%11s  %s/%s\n",
        "step",
        "dt",
        "energy",
        "Δe",
        "|Δλ|",
        "speed",
        "meas(s)"
    )
    gate = get_gate(alg.dt, ham)
    wts0, peps0, envs0 = deepcopy(peps.weights), deepcopy(envs), deepcopy(envs)
    esite0, ediff, wtdiff = Inf, 0.0, 1.0
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = ntu_iter(gate, peps, alg)
        wtdiff = compare_weights(peps.weights, wts0)
        converge = (wtdiff < alg.tol)
        cancel = (count == alg.maxiter)
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if count == 1 || count % alg.ctm_int == 0 || converge || cancel
            # monitor energy
            meast0 = time()
            peps_ = InfinitePEPS(peps)
            envs = leading_boundary(envs, peps_, alg.ctm_alg)
            esite = costfun(peps_, envs, ham) / (Nr * Nc)
            meast1 = time()
            # monitor change of energy
            ediff = esite - esite0
            message = @sprintf(
                "%-4d %7.0e%10.5f%12.3e%11.3e  %.3f/%.3f\n",
                count,
                alg.dt,
                esite,
                ediff,
                wtdiff,
                time1 - time0,
                meast1 - meast0
            )
            cancel ? (@warn message) : (@info message)
            if ediff > 0
                @info "Energy starts to increase. Abort evolution.\n"
                # restore peps and envs at last checking
                peps, envs = peps0, envs0
                break
            end
            esite0, peps0, envs0 = esite, deepcopy(peps), deepcopy(envs)
            converge && break
        end
    end
    # reconverge the environment tensors
    for io in (stdout, stderr)
        @printf(io, "Reconverging final envs ... \n")
    end
    peps_ = InfinitePEPS(peps)
    envs = leading_boundary(envs, peps_, ctm_alg)
    time_end = time()
    @printf("Evolution time: %.3f s\n\n", time_end - time_start)
    print(stderr, "\n----------\n\n")
    return peps, envs, (esite0, ediff)
end

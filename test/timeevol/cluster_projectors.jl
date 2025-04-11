using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using Random
using PEPSKit: sdiag_pow, _cluster_truncate!
include("cluster_tools.jl")

nrm = 20
Vspaces = [
    (ℂ^2, ℂ^4, (ℂ^12)'),
    (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 2, 1 => 2),
        Vect[FermionParity](0 => 6, 1 => 6)',
    ),
]

@testset "Cluster bond truncation with projectors" begin
    Random.seed!(0)
    N, n = 5, 2
    for (Vphy, Vns, V) in Vspaces
        Vvirs = fill(Vns, N + 1)
        Vvirs[n + 1] = V
        Ms1 = Vector{AbstractTensorMap}(undef, N)
        for i in 1:N
            Vw, Ve = Vvirs[i], Vvirs[i + 1]
            Ms1[i] = rand(Vw ← Vphy' ⊗ Vns ⊗ Vns' ⊗ Ve) / nrm
        end
        revs = [isdual(space(M, 1)) for M in Ms1[2:end]]
        # no truncation
        Ms2 = deepcopy(Ms1)
        wts2, ϵs = _cluster_truncate!(Ms2, FixedSpaceTruncation(), revs)
        @test all((ϵ == 0) for ϵ in ϵs)
        absorb_wts_cluster!(Ms2, wts2)
        for (i, M) in enumerate(Ms2)
            Ms2[i] *= 0.05 / norm(M, Inf)
        end
        @test fidelity_cluster(Ms1, Ms2) ≈ 1.0
        # truncation on one bond
        Ms3 = deepcopy(Ms1)
        wts3, ϵs = _cluster_truncate!(Ms3, truncspace(Vns), revs)
        @test all((i == n) || (ϵ == 0) for (i, ϵ) in enumerate(ϵs))
        absorb_wts_cluster!(Ms3, wts3)
        for (i, M) in enumerate(Ms3)
            Ms3[i] *= 0.05 / norm(M, Inf)
        end
        ϵ = ϵs[n]
        wt2, wt3 = wts2[n], wts3[n]
        fid3, fid3_ = fidelity_cluster(Ms1, Ms3), fidelity_cluster(Ms2, Ms3)
        @info "Fidelity of truncated cluster = $(fid3)"
        @test fid3 ≈ fid3_
        @test fid3 ≈ (norm(wt3) / norm(wt2))^2
        @test fid3 ≈ 1.0 - (ϵ / norm(wt2))^2
    end
end

@testset "Identity gate on 3-site cluster" begin
    N, n = 3, 1
    for (Vphy, Vns, V) in Vspaces
        Vvirs = fill(Vns, N + 1)
        Vvirs[n + 1] = V
        Ms1 = Vector{AbstractTensorMap}(undef, N)
        for i in 1:N
            Vw, Ve = Vvirs[i], Vvirs[i + 1]
            Ms1[i] = rand(Vw ← Vphy' ⊗ Vns ⊗ Vns' ⊗ Ve) / nrm
        end
        unit = id(Vphy)
        gate = reduce(⊗, fill(unit, 3))
        gs = PEPSKit.gate_to_mpo3(gate)
        @test mpo_to_gate3(gs) ≈ gate
        Ms2 = deepcopy(Ms1)
        PEPSKit._apply_gatempo!(Ms2, gs)
        fid = fidelity_cluster(Ms1, Ms2)
        @test fid ≈ 1.0
    end
end

@testset "Heisenberg model with usual SU and 3-site SU" begin
    Nr, Nc = 2, 2
    ctmrg_tol = 1e-9
    Random.seed!(100)
    Pspace = U1Space(1//2 => 1, -1//2 => 1)
    Vspace = U1Space(0 => 2, 1//2 => 1, -1//2 => 1)
    Espace = U1Space(0 => 8, 1//2 => 4, -1//2 => 4)
    ham = j1_j2(ComplexF64, U1Irrep, InfiniteSquare(Nr, Nc); J1=1.0, J2=0.0, sublattice=false)
    wpeps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(Nr, Nc))
    # convert to real tensors
    ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)
    # usual 2-site simple update, and measure energy
    dts = [1e-2, 1e-3, 1e-4]
    tols = [1e-8, 1e-8, 1e-8]
    maxiter = 6000
    trscheme = truncerr(1e-10) & truncdim(4)
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        result = simpleupdate(wpeps, ham, alg; bipartite=false, check_interval=1000)
        wpeps = result[1]
    end
    peps = InfinitePEPS(wpeps)
    normalize!(peps)
    env, = leading_boundary(CTMRGEnv(rand, Float64, peps, Espace), peps; tol=ctmrg_tol)
    e_site = cost_function(peps, env, ham) / (Nr * Nc)
    @info "2-site simple update energy = $e_site"
    # continue with 3-site simple update; energy should not change much
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        result = simpleupdate3site(wpeps, ham, alg; check_interval=1000)
        wpeps = result[1]
    end
    peps = InfinitePEPS(wpeps)
    normalize!(peps)
    env, = leading_boundary(env, peps; tol=ctmrg_tol)
    e_site2 = cost_function(peps, env, ham) / (Nr * Nc)
    @info "3-site simple update energy = $e_site2"
    @test e_site ≈ e_site2 atol = 1e-4
end

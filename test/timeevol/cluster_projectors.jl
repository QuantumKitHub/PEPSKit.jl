using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using Random
import MPSKitModels: hubbard_space
using PEPSKit: sdiag_pow, _cluster_truncate!
using MPSKit: GenericMPSTensor, MPSBondTensor
include("cluster_tools.jl")

Vspaces = [
    (
        U1Space(0 => 1, 1 => 1, -1 => 1),
        U1Space(0 => 1, 1 => 2, -1 => 1)',
        U1Space(0 => 4, 1 => 5, -1 => 6)',
    ),
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
        Ms1 = map(1:N) do i
            Vw, Ve = Vvirs[i], Vvirs[i + 1]
            return rand(Vw ⊗ Vphy ⊗ Vns' ⊗ Vns ← Ve)
        end
        normalize!.(Ms1, Inf)
        revs = [isdual(space(M, 1)) for M in Ms1[2:end]]
        # no truncation
        Ms2 = deepcopy(Ms1)
        wts2, ϵs, = _cluster_truncate!(Ms2, fill(FixedSpaceTruncation(), N - 1), revs)
        @test all((ϵ == 0) for ϵ in ϵs)
        normalize!.(Ms2, Inf)
        @test fidelity_cluster(Ms1, Ms2) ≈ 1.0
        lorths, rorths = verify_cluster_orth(Ms2, wts2)
        @test all(lorths) && all(rorths)
        # truncation on one bond
        Ms3 = deepcopy(Ms1)
        tspace = isdual(Vns) ? flip(Vns) : Vns
        wts3, ϵs, = _cluster_truncate!(Ms3, fill(truncspace(tspace), N - 1), revs)
        @test all((i == n) || (ϵ == 0) for (i, ϵ) in enumerate(ϵs))
        normalize!.(Ms3, Inf)
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
        Ms1 = map(1:N) do i
            Vw, Ve = Vvirs[i], Vvirs[i + 1]
            return normalize(rand(Vw ⊗ Vphy ⊗ Vns' ⊗ Vns ← Ve), Inf)
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
    for (Vphy, Vns, V) in Vspaces
        Vvirs = fill(Vns, N + 1)
        Vvirs[n + 1] = V
        Ms1 = map(1:N) do i
            Vw, Ve = Vvirs[i], Vvirs[i + 1]
            return normalize(rand(Vw ⊗ Vphy ⊗ Vphy' ⊗ Vns' ⊗ Vns ← Ve), Inf)
        end
        unit = id(Vphy)
        gate = reduce(⊗, fill(unit, 3))
        gs = PEPSKit.gate_to_mpo3(gate)
        @test mpo_to_gate3(gs) ≈ gate
        for gate_ax in 1:2
            Ms2 = deepcopy(Ms1)
            PEPSKit._apply_gatempo!(Ms2, gs)
            fid = fidelity_cluster(
                [first(PEPSKit._fuse_physicalspaces(M)) for M in Ms1],
                [first(PEPSKit._fuse_physicalspaces(M)) for M in Ms2]
            )
            @test fid ≈ 1.0
        end
    end
end

@testset "Hubbard model with 2-site and 3-site SU" begin
    Nr, Nc = 2, 2
    ctmrg_tol = 1.0e-9
    Random.seed!(100)
    # with U(1) spin rotation symmetry
    Pspace = hubbard_space(Trivial, U1Irrep)
    Vspace = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    Espace = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 8, (1, 1 // 2) => 4, (1, -1 // 2) => 4)
    trunc_env = truncerror(; atol = 1.0e-12) & truncrank(16)
    peps = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc))
    wts = SUWeight(peps)
    ham = real(
        hubbard_model(
            ComplexF64, Trivial, U1Irrep, InfiniteSquare(Nr, Nc); t = 1.0, U = 8.0, mu = 0.0
        ),
    )
    # usual 2-site simple update, and measure energy
    dts = [1.0e-2, 1.0e-2, 5.0e-3]
    tols = [1.0e-8, 1.0e-8, 1.0e-8]
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        trunc = truncerror(; atol = 1.0e-10) & truncrank(n == 1 ? 4 : 2)
        alg = SimpleUpdate(; trunc, bipartite = true)
        peps, wts, = time_evolve(peps, ham, dt, 10000, alg, wts; tol, check_interval = 1000)
    end
    normalize!.(peps.A, Inf)
    env = CTMRGEnv(wts, peps)
    env, = leading_boundary(env, peps; tol = ctmrg_tol, trunc = trunc_env)
    e_site = cost_function(peps, env, ham) / (Nr * Nc)
    @info "2-site simple update energy = $e_site"
    # continue with 3-site simple update; energy should not change much
    dts = [1.0e-2, 5.0e-3]
    tols = [1.0e-8, 1.0e-8]
    trunc = truncerror(; atol = 1.0e-10) & truncrank(2)
    alg = SimpleUpdate(; trunc, force_3site = true)
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        peps, wts, = time_evolve(peps, ham, dt, 5000, alg, wts; tol, check_interval = 1000)
    end
    normalize!.(peps.A, Inf)
    env, = leading_boundary(env, peps; tol = ctmrg_tol, trunc = trunc_env)
    e_site2 = cost_function(peps, env, ham) / (Nr * Nc)
    @info "3-site simple update energy = $e_site2"
    @test e_site ≈ e_site2 atol = 5.0e-4
end

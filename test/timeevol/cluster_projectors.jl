using Test
using TensorKit
using PEPSKit
using LinearAlgebra
using Random
import MPSKitModels: hubbard_space
using PEPSKit: sdiag_pow, _cluster_truncate!, _flip_virtuals!, _next
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
        flips = [isdual(space(M, 1)) for M in Ms1[2:end]]
        # no truncation
        Ms2 = _flip_virtuals!(deepcopy(Ms1), flips)
        wts2, ϵs, = _cluster_truncate!(Ms2, fill(FixedSpaceTruncation(), N - 1))
        @test all((ϵ == 0) for ϵ in ϵs)
        normalize!.(Ms2, Inf)
        @test fidelity_cluster(Ms1, Ms2) ≈ 1.0
        lorths, rorths = verify_cluster_orth(Ms2, wts2)
        @test all(lorths) && all(rorths)
        # truncation on one bond
        Ms3 = _flip_virtuals!(deepcopy(Ms1), flips)
        tspace = isdual(Vns) ? flip(Vns) : Vns
        wts3, ϵs, = _cluster_truncate!(Ms3, fill(truncspace(tspace), N - 1))
        @test all((i == n) || (ϵ == 0) for (i, ϵ) in enumerate(ϵs))
        normalize!.(Ms3, Inf)
        ϵ = ϵs[n]
        wt2, wt3 = wts2[n], wts3[n]
        _flip_virtuals!(Ms3, flips)
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
        flips = [isdual(space(M, 1)) for M in Ms1[2:end]]
        unit = id(Vphy)
        gate = reduce(⊗, fill(unit, 3))
        gs = PEPSKit.gate_to_mpo(gate)
        @test mpo_to_gate3(gs) ≈ gate
        Ms2 = _flip_virtuals!(deepcopy(Ms1), flips)
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
        flips = [isdual(space(M, 1)) for M in Ms1[2:end]]
        unit = id(Vphy)
        gate = reduce(⊗, fill(unit, 3))
        gs = PEPSKit.gate_to_mpo(gate)
        @test mpo_to_gate3(gs) ≈ gate
        for gate_ax in 1:2
            Ms2 = _flip_virtuals!(deepcopy(Ms1), flips)
            PEPSKit._apply_gatempo!(Ms2, gs)
            fid = fidelity_cluster(
                [first(PEPSKit._fuse_physicalspaces(M)) for M in Ms1],
                [first(PEPSKit._fuse_physicalspaces(M)) for M in Ms2]
            )
            @test fid ≈ 1.0
        end
    end
end

@testset "Hubbard model SU (MPO gate)" begin
    Nr, Nc = 2, 2
    ctmrg_tol = 1.0e-9
    Random.seed!(1459)
    # with U(1) spin rotation symmetry
    Pspace = hubbard_space(Trivial, U1Irrep)
    Vspace = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1 // 2) => 1, (1, -1 // 2) => 1)
    Espace = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 8, (1, 1 // 2) => 4, (1, -1 // 2) => 4)
    truncs_env = collect(truncerror(; atol = 1.0e-12) & truncrank(χ) for χ in [8, 16])
    peps0 = InfinitePEPS(rand, Float64, Pspace, Vspace, Vspace'; unitcell = (Nr, Nc))
    # make initial state bipartite
    for r in 1:2
        peps0.A[_next(r, 2), 2] = copy(peps0.A[r, 1])
    end
    wts0 = SUWeight(peps0)
    ham = hubbard_model(Float64, Trivial, U1Irrep, InfiniteSquare(Nr, Nc); t = 1.0, U = 6.0, mu = 3.0)
    # applying 2-site gates decomposed to MPO or not,
    # resulting energy should be almost the same
    e_sites = map((true, false)) do force_mpo
        dts = [1.0e-2, 1.0e-2]
        tols = [1.0e-6, 1.0e-8]
        peps, wts = deepcopy(peps0), deepcopy(wts0)
        for (n, (dt, tol)) in enumerate(zip(dts, tols))
            trunc = truncerror(; atol = 1.0e-10) & truncrank(n == 1 ? 4 : 2)
            alg = SimpleUpdate(; trunc, force_mpo)
            peps, wts, = time_evolve(
                peps, ham, dt, 10000, alg, wts;
                tol, symmetrize_gates = true, check_interval = 1000
            )
        end
        normalize!.(peps.A, Inf)
        env = CTMRGEnv(wts)
        for trunc in truncs_env
            env, = leading_boundary(env, peps; alg = :sequential, tol = ctmrg_tol, trunc)
        end
        e_site = cost_function(peps, env, ham) / (Nr * Nc)
        @info "Energy (force_mpo = $(force_mpo)): $e_site"
        return e_site
    end
    @test e_sites[1] ≈ e_sites[2] atol = 1.0e-4
end

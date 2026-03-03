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
        gs = PEPSKit.gate_to_mpo3(gate)
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
        gs = PEPSKit.gate_to_mpo3(gate)
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

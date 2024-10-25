using Test
using Random
using PEPSKit
using MPSKit
using PEPSKit: leading_boundary
using TensorKit
using LinearAlgebra

begin "test utility"
    ds = [ℂ^2]
    Ds = [ℂ^2]
    χs = [ℂ^20]
end

@testset "initialize environments for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))

    itp = InfiniteTransferPEPS(ipeps)
    rt = VUMPSRuntime(itp, χ)
    @test rt isa VUMPSRuntime
end

@testset "vumps one side runtime for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    ipeps = symmetrize!(ipeps, RotateReflect())

    alg = PEPSKit.VUMPS(maxiter=100, verbosity=2, ifupdown=false)

    itp = InfiniteTransferPEPS(ipeps)
    env = leading_boundary(itp, VUMPSRuntime(itp, χ, alg), alg)
    @test env isa VUMPSEnv

    Z = abs(norm(ipeps, env))

    ctm = MPSKit.leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    @test Z ≈ Z′ rtol = 1e-12
end

@testset "vumps two side runtime for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    alg = PEPSKit.VUMPS(maxiter=100, verbosity=2, ifupdown=true)

    itp = InfiniteTransferPEPS(ipeps)
    env = leading_boundary(itp, VUMPSRuntime(itp, χ, alg), alg)
    @test env isa VUMPSEnv

    Z = abs(norm(ipeps, env))

    ctm = MPSKit.leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    @test Z ≈ Z′ rtol = 1e-8
end
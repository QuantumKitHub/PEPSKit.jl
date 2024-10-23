using Test
using Random
using PEPSKit
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
    alg = VUMPS(verbosity=4)

    itp = InfiniteTransferPEPS(ipeps)
    vumpsruntime = VUMPSRuntime(itp, χ, alg)
    @test vumpsruntime isa VUMPSRuntime
end

@testset "vumps runtime for unitcell $Ni x $Nj" for Ni in 1:2, Nj in 1:2, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    alg = VUMPS(maxiter=100, verbosity=2)

    itp = InfiniteTransferPEPS(ipeps)
    rt = leading_boundary(itp, VUMPSRuntime(itp, χ, alg), alg)
    @test rt isa VUMPSRuntime

    Z = abs(norm(ipeps, rt))

    ctm = MPSKit.leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    @test Z ≈ Z′ rtol = 1e-2
end
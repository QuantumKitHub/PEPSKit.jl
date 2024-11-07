using Test
using Random
using PEPSKit
using PEPSKit: nearest_neighbour_energy
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

    rt = VUMPSRuntime(ipeps, χ)
    @test rt isa VUMPSRuntime
end

@testset "vumps one side runtime for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    ipeps = symmetrize!(ipeps, RotateReflect())

    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)

    rt = leading_boundary(VUMPSRuntime(ipeps, χ, alg), ipeps, alg)
    env = VUMPSEnv(rt, ipeps)
    @test env isa VUMPSEnv

    Z = abs(norm(ipeps, env))

    ctm = leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    @test Z ≈ Z′ rtol = 1e-12
end

@testset "vumps one side runtime energy for unitcell $Ni x $Nj" for Ni in 1:1, Nj in 1:1, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(100)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    ipeps = symmetrize!(ipeps, RotateReflect())

    alg = VUMPS(maxiter=100, verbosity=2, ifupdown=false)

    rt = leading_boundary(VUMPSRuntime(ipeps, χ, alg), ipeps, alg)
    env = VUMPSEnv(rt, ipeps)
    H = heisenberg_XYZ(InfiniteSquare())
    H = H.terms[1].second
    # Hh, Hv = H.terms[1]
    
    @show nearest_neighbour_energy(ipeps, H, H, env)
    # Z = abs(norm(ipeps, env))

    # ctm = leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    # Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    # @test Z ≈ Z′ rtol = 1e-12
end

@testset "vumps two side runtime for unitcell $Ni x $Nj" for Ni in 1:3, Nj in 1:3, (d, D, χ) in zip(ds, Ds, χs)
    Random.seed!(42)
    ipeps = InfinitePEPS(d, D; unitcell=(Ni, Nj))
    alg = PEPSKit.VUMPS(maxiter=100, verbosity=2, ifupdown=true)

    rt = leading_boundary(VUMPSRuntime(ipeps, χ, alg), ipeps, alg)
    env = VUMPSEnv(rt, ipeps)
    @test env isa VUMPSEnv

    Z = abs(norm(ipeps, env))

    ctm = leading_boundary(CTMRGEnv(ipeps, χ), ipeps, CTMRG(; verbosity=2))
    Z′ = abs(norm(ipeps, ctm))^(1/Ni/Nj)

    @test Z ≈ Z′ rtol = 1e-8
end
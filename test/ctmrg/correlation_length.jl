using Test: @test, @testset
using TensorKit
using TensorKit: ←, ⊗, SU2Irrep, Vect, permute
using MPSKit: correlation_length, leading_boundary
using PEPSKit: CTMRGEnv, InfinitePEPS

@testset "Correlation length" begin
    # regression test for https://github.com/QuantumKitHub/PEPSKit.jl/issues/197
    vD = Vect[SU2Irrep](1 // 2 => 1)
    vd = Vect[SU2Irrep](2 => 1)
    tAKLT_A = ones(vd ← vD ⊗ vD ⊗ vD ⊗ vD)
    tAKLT_B = permute(adjoint(tAKLT_A), ((5,), (1, 2, 3, 4)))

    ψ = InfinitePEPS([tAKLT_A tAKLT_B; tAKLT_B tAKLT_A])
    trunc = trunctol(; atol = 1.0e-12) & truncrank(48)
    boundary_alg = (; tol = 1.0e-10, trunc, maxiter = 400, verbosity = 3)
    env0 = CTMRGEnv(randn, Float64, ψ, oneunit(vD))
    env, info_ctmrg = leading_boundary(env0, ψ; boundary_alg...)

    ξ_h, ξ_v, λ_h, λ_v = correlation_length(ψ, env; sector = SU2Irrep(1))
    @test ξ_h isa Vector{Float64}
    @test size(ξ_h) == (2,)
    @test all(isfinite.(ξ_h))
    @test all(ξ_h .> 0)

    @test ξ_v isa Vector{Float64}
    @test size(ξ_v) == (2,)
    @test all(isfinite.(ξ_v))
    @test all(ξ_v .> 0)

    # AKLT state should have correlation length ~ 2sites
    # https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.121115
    @test all(x -> isapprox(x * size(ψ, 2), 2; atol = 0.1), ξ_h)
    @test all(x -> isapprox(x * size(ψ, 1), 2; atol = 0.1), ξ_v)
end

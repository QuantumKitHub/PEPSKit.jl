using Test: @test, @testset
using TensorKit: ←, ⊗, SU2Irrep, Vect, permute, truncdim, truncbelow
using MPSKit: correlation_length, leading_boundary
using PEPSKit: CTMRGEnv, InfinitePEPS

@testset "Correlation length" begin
    # regression test for https://github.com/QuantumKitHub/PEPSKit.jl/issues/197
    vD = Vect[SU2Irrep](1//2 => 1)
    vd = Vect[SU2Irrep](2 => 1)
    tAKLT_A = ones(vd ← vD ⊗ vD ⊗ vD ⊗ vD)
    tAKLT_B = permute(adjoint(tAKLT_A), (5,), (1, 2, 3, 4))

    ψ = InfinitePEPS([tAKLT_A tAKLT_B; tAKLT_B tAKLT_A])
    trscheme = truncdim(20) & truncbelow(1e-12)
    boundary_alg = (; tol=1e-10, trscheme=trscheme, maxiter=1, verbosity=0)
    env0 = CTMRGEnv(randn, Float64, ψ, oneunit(vD))
    env, info_ctmrg = leading_boundary(env0, ψ; boundary_alg...)

    ξ_h, ξ_v, λ_h, λ_v = correlation_length(ψ, env)
    @test ξ_h isa Vector{Float64}
    @test size(ξ_h) == (2,)
    @test all(isfinite.(ξ_h))
    @test all(ξ_h .> 0)

    @test ξ_v isa Vector{Float64}
    @test size(ξ_v) == (2,)
    @test all(isfinite.(ξ_v))
    @test all(ξ_v .> 0)
end

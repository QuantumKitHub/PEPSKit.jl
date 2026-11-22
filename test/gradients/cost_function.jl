using Test, Random
using TensorKit, Zygote
using PEPSKit

@testset "cost_function rrule" begin
    Random.seed!(20261118)
    H = j1_j2_model(ComplexF64, Trivial, InfiniteSquare(2, 2); J2 = 0.5)
    peps = InfinitePEPS(randn, ComplexF64, physicalspace(H), fill(ℂ^2, 2, 2))
    env = CTMRGEnv(randn, ComplexF64, peps, ℂ^6)

    # against the gradient without the rule
    res = Zygote.withgradient(cost_function, peps, env, H)
    ref = Zygote.withgradient((ψ, ε) -> real(expectation_value(ψ, H, ε)), peps, env)
    @test res.val ≈ ref.val
    @test res.grad[1] ≈ ref.grad[1]
    @test res.grad[2] ≈ ref.grad[2]
end

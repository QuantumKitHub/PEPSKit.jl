using Test
using Random
using LinearAlgebra
using TensorKit
using VectorInterface
using PEPSKit

dtype = ComplexF64
Vphyss = [ℂ^2, U1Space(0 => 1, -1 => 1, 1 => 1)]
Vpepss = [ℂ^4, U1Space(0 => 2, -1 => 1, 1 => 1)]

@testset "Norm-preserving tensor retractions for sectortype $(sectortype(Vphyss[i]))" for i in
                                                                                          eachindex(
    Vphyss
)
    Vphys = Vphyss[i]
    Vpeps = Vpepss[i]
    peps_space = Vphys ← Vpeps ⊗ Vpeps ⊗ Vpeps' ⊗ Vpeps'

    α = 1e-1 * randn(Float64)
    A = randn(dtype, peps_space)
    normalized_A = scale(A, inv(norm(A)))
    η = randn(dtype, peps_space)
    ζ = randn(dtype, peps_space)
    add!(η, normalized_A, -inner(normalized_A, η))
    add!(ζ, normalized_A, -inner(normalized_A, ζ))

    A´, ξ = PEPSKit.vector_retract(A, η, α)
    @test norm(A´) ≈ norm(A) rtol = 1e-12

    PEPSKit.vector_transport!(ζ, A, η, α, A´)
    @test inner(ζ, A´) ≈ 0 atol = 1e-12
end

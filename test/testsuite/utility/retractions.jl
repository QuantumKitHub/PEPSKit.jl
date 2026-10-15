using Test
using Random
using LinearAlgebra
using TensorKit
using VectorInterface
using PEPSKit
using Adapt

dtype = ComplexF64
Vphyss = [ℂ^2, U1Space(0 => 1, -1 => 1, 1 => 1)]
Vpepss = [ℂ^4, U1Space(0 => 2, -1 => 1, 1 => 1)]

"""Check real-sphere retractions and isometric transport of tensor directions."""
function utility_retractions(AT)
    return @testset "Norm-preserving tensor retractions for sectortype $(sectortype(Vphyss[i])) ($AT)" for i in
        eachindex(
            Vphyss
        )
        Vphys = Vphyss[i]
        Vpeps = Vpepss[i]
        peps_space = Vphys ← Vpeps ⊗ Vpeps ⊗ Vpeps' ⊗ Vpeps'

        α = 1.0e-1 * randn(Float64)
        A = adapt(AT, randn(dtype, peps_space))
        normalized_A = scale(A, inv(norm(A)))
        η = adapt(AT, randn(dtype, peps_space))
        ζ = adapt(AT, randn(dtype, peps_space))
        add!(η, normalized_A, -real(inner(normalized_A, η)))
        add!(ζ, normalized_A, -real(inner(normalized_A, ζ)))

        A′, ξ = PEPSKit.norm_preserving_retract(A, η, α)
        @test norm(A′) ≈ norm(A) rtol = 1.0e-12

        @test real(inner(A′, ξ)) ≈ 0 atol = 1.0e-12
        norm_ζ = norm(ζ)
        PEPSKit.norm_preserving_transport!(ζ, A, η, α, A′)
        @test real(inner(ζ, A′)) ≈ 0 atol = 1.0e-12
        @test norm(ζ) ≈ norm_ζ rtol = 1.0e-12

        for point in (A, [A], VectorInterface.MinimalMVec([A]))
            norm_point = norm(point)
            unchanged_A, zero_derivative = PEPSKit.norm_preserving_retract(point, zerovector(point), α)
            @test norm(unchanged_A) ≈ norm_point
            @test norm(zero_derivative) == 0
            scale!(unchanged_A, 0)
            @test norm(point) ≈ norm_point
        end
        @test PEPSKit.norm_preserving_transport!(ζ, A, zerovector(A), α, A) === ζ
    end
end

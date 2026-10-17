using Test
using Random
using LinearAlgebra
using TensorKit
using VectorInterface
using PEPSKit
using PEPSKit: real_inner
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
        # Remove radial components so both directions are tangent at A.
        add!(η, normalized_A, -real_inner(normalized_A, η))
        add!(ζ, normalized_A, -real_inner(normalized_A, ζ))

        # Retraction preserves the tensor norm and gives a tangent path derivative.
        A′, ξ = PEPSKit.norm_preserving_retract(A, η, α)
        @test norm(A′) ≈ norm(A) rtol = 1.0e-12
        @test real_inner(A′, ξ) ≈ 0 atol = 1.0e-12

        # Transport preserves the direction norm and makes it tangent at A′.
        norm_ζ = norm(ζ)
        PEPSKit.norm_preserving_transport!(ζ, A, η, α, A′)
        @test real_inner(ζ, A′) ≈ 0 atol = 1.0e-12
        @test norm(ζ) ≈ norm_ζ rtol = 1.0e-12

        # Zero update direction returns independent copies,
        # even for minimal VectorInterface vectors.
        for point in (A, [A], VectorInterface.MinimalMVec([A]))
            norm_point = norm(point)
            unchanged_A, zero_derivative = PEPSKit.norm_preserving_retract(point, zerovector(point), α)
            @test norm(unchanged_A) ≈ norm_point
            @test norm(zero_derivative) == 0
            # Mutating the returned copy must not change the input.
            scale!(unchanged_A, 0)
            @test norm(point) ≈ norm_point
        end
        # Zero-step transport returns the original direction.
        @test PEPSKit.norm_preserving_transport!(ζ, A, zerovector(A), α, A) === ζ

        # Make η Hermitian-orthogonal to A so ζ has purely imaginary overlaps with both.
        η = add(η, normalized_A, -inner(normalized_A, η))
        # Such overlaps must leave ζ unchanged under real-sphere transport.
        ζ = scale(add(A, η), im)
        α = 0.3
        A′, = PEPSKit.norm_preserving_retract(A, η, α)
        @test PEPSKit.norm_preserving_transport!(copy(ζ), A, η, α, A′) ≈ ζ
    end
end

using TensorKit
using PEPSKit
using Random
using Test

Vphys = ComplexSpace(2)
Vbond = ComplexSpace(3)
maxiter_approx = 8
maxiter_boundary = 250
sd = 123456

@testset "approximate methods with FidelityMaxCrude" begin
    peps1 = InfinitePEPS(Vphys, Vbond)
    peps2 = InfinitePEPS(Vphys, Vbond)
    Random.seed!(sd)
    peps_approx1, = approximate(
        peps1, peps2;
        maxiter = maxiter_approx,
        boundary_alg = (; maxiter = maxiter_boundary, trunc = truncrank(6), verbosity = 1)
    )
    Random.seed!(sd) # reseeding with the same seed is required for the test to pass
    peps_approx2, = approximate(
        peps1, peps2, FidelityMaxCrude(;
            maxiter = maxiter_approx,
            boundary_alg = (; maxiter = maxiter_boundary, trunc = truncrank(6), verbosity = 1)
        )
    )

    @test peps_approx1 ≈ peps_approx2
end

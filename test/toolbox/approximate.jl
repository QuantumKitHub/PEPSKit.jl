using TensorKit
using PEPSKit
using Random
using Test

@testset "approximate methods with FidelityMaxCrude" begin
    peps1 = InfinitePEPS(2, 2)
    peps2 = InfinitePEPS(2, 2)
    Random.seed!(1234)
    peps_approx1, = approximate(
        peps1, peps2; maxiter = 3,
        boundary_alg = (; trscheme = truncdim(6), verbosity = 1)
    )
    Random.seed!(1234)
    peps_approx2, = approximate(
        peps1, peps2, FidelityMaxCrude(;
            maxiter = 3,
            boundary_alg = (; trscheme = truncdim(6), verbosity = 1)
        )
    )

    @test peps_approx1 ≈ peps_approx2
end

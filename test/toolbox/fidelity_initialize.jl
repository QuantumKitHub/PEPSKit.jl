using TensorKit
using PEPSKit
using Random
using Test

@testset "Fidelity initialization with approximate!" begin
    Random.seed!(18092025)
    unitcell = (3, 3)
    Pspaces = fill(2, unitcell...)
    Nspaces = rand(2:4, unitcell...)
    Espaces = rand(2:4, unitcell...)
    pepssrc = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)

    peps_single, = single_site_fidelity_initialize(
        pepssrc, ℂ^6; maxiter = 10, noise_amp = 0.1,
        boundary_alg = (; tol = 1.0e-6, maxiter = 20, verbosity = 1)
    )
    @test peps_single isa InfinitePEPS
end

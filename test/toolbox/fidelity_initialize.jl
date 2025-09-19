using TensorKit
using PEPSKit
using PEPSKit: _spacemax
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

@testset "_spacemax for non-uniform unit cell and symmetry sectors" begin
    Pspaces = fill(Z2Space(0 => 1, 1 => 1), (2, 2))
    Nspaces = [
        Z2Space(0 => 2, 1 => 3) Z2Space(0 => 2, 1 => 5)
        Z2Space(0 => 4, 1 => 3) Z2Space(0 => 2, 1 => 1)
    ]
    Espaces = [
        Z2Space(0 => 6, 1 => 3) Z2Space(0 => 2, 1 => 1)
        Z2Space(0 => 2, 1 => 3) Z2Space(0 => 2, 1 => 4)
    ]
    peps = InfinitePEPS(Pspaces, Nspaces, Espaces)
    @test _spacemax(peps) == Z2Space(0 => 6, 1 => 5)
end

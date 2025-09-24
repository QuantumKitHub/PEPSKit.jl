using TensorKit
using PEPSKit
using PEPSKit: single_site_fidelity_initialize, _spacemax
using Random
using Test

@testset "Fidelity initialization with single_site_fidelity_initialize" begin
    Random.seed!(18092025)

    peps_1x1 = InfinitePEPS(randn, ComplexF64, 2, 2)
    peps_single1 = single_site_fidelity_initialize(
        peps_1x1, ℂ^3; noise_amp = 0.3, tol = 1.0e-2, miniter = 5,
        boundary_alg = (; tol = 1.0e-6, trscheme = truncdim(10), verbosity = 1)
    )
    @test peps_single1 isa InfinitePEPS
    @test size(peps_single1) == (1, 1)

    unitcell = (3, 3)
    Pspaces = fill(2, unitcell...)
    Nspaces = rand(2:4, unitcell...)
    Espaces = rand(2:4, unitcell...)
    peps_3x3 = InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)

    peps_single2 = single_site_fidelity_initialize(
        peps_3x3; maxiter = 10, noise_amp = 0.1,
        boundary_alg = (; tol = 1.0e-6, maxiter = 20, verbosity = 1)
    )
    @test peps_single2 isa InfinitePEPS
    @test size(peps_single2) == (1, 1)
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

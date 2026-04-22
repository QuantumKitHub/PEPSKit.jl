using Random
using Test
using TensorKit
using PEPSKit
using PEPSKit: _∂local_norm, _local_norm

maxiter_approx = 10
maxiter_boundary = 250
χ = 6
Vphys = ComplexSpace(2)
Vbond = ComplexSpace(3)
Venv = ComplexSpace(χ)
sd = 123456

@testset "FidelityMaxCrude internals" begin
    ket = InfinitePEPS(Vphys, Vbond)
    bra = InfinitePEPS(Vphys, Vbond)
    network = InfiniteSquareNetwork(ket, bra)
    env = CTMRGEnv(network, Venv)

    nrm1 = _local_norm(ket, bra, env)
    ∂norm = _∂local_norm(ket, env)
    nrm2 = _local_norm(bra, ∂norm)
    @test nrm1 ≈ nrm2
end

@testset "FidelityMaxCrude approximate methods" begin
    peps1 = InfinitePEPS(Vphys, Vbond)
    peps2 = InfinitePEPS(Vphys, Vbond)
    Random.seed!(sd)
    peps_approx1, env1 = approximate(
        peps1, peps2;
        tol = 1.0e-5,
        maxiter = maxiter_approx,
        boundary_alg = (; maxiter = maxiter_boundary, trunc = truncrank(χ), verbosity = 1)
    )

    # run with same seed to check if keyword argument selection works properly
    Random.seed!(sd)
    peps_approx2, env2 = approximate(
        peps1, peps2, FidelityMaxCrude(;
            tol = 1.0e-5,
            maxiter = maxiter_approx,
            boundary_alg = (; maxiter = maxiter_boundary, trunc = truncrank(χ), verbosity = 1)
        )
    )
    @test peps_approx1 ≈ peps_approx2

    # run with different seed and check if fidelities approach 1
    Random.seed!(floor(Int, 2.3 * sd))
    peps_approx3, = approximate(
        peps1, peps2;
        tol = 1.0e-5,
        maxiter = maxiter_approx,
        boundary_alg = (; maxiter = maxiter_boundary, trunc = truncrank(χ), verbosity = 1)
    )
    envnw, = leading_boundary(env1, InfiniteSquareNetwork(peps_approx1, peps_approx3); maxiter = maxiter_boundary)
    ∂norm = _∂local_norm(peps_approx1, envnw)
    fid = abs2(_local_norm(peps_approx3, ∂norm))
    @test fid ≈ 1 rtol = 0.1
end

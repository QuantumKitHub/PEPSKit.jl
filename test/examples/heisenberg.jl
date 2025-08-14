using Test
using Random
using Accessors
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
Dbond = 2
χenv = 16
gradtol = 1.0e-3
# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
E_ref = -0.6602310934799577

@testset "(1, 1) unit cell AD optimization" begin
    # initialize states
    Random.seed!(123)
    H = heisenberg_XYZ(InfiniteSquare())
    peps₀ = InfinitePEPS(2, Dbond)
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀)

    # optimize energy and compute correlation lengths
    peps, env, E, = fixedpoint(H, peps₀, env₀; optimizer_alg = (; tol = gradtol, maxiter = 25))
    ξ_h, ξ_v, = correlation_length(peps, env)

    @test E ≈ E_ref atol = 1.0e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "(1, 2) unit cell AD optimization" begin
    # initialize states
    Random.seed!(456)
    unitcell = (1, 2)
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    peps₀ = InfinitePEPS(2, Dbond; unitcell)
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀)

    # optimize energy and compute correlation lengths
    peps, env, E, = fixedpoint(H, peps₀, env₀; optimizer_alg = (; tol = gradtol, maxiter = 25))
    ξ_h, ξ_v, = correlation_length(peps, env)

    @test E ≈ 2 * E_ref atol = 1.0e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "Simple update into AD optimization" begin
    # random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
    Random.seed!(100)
    N1, N2 = 2, 2
    Pspace = ℂ^2
    Vspace = ℂ^Dbond
    Espace = ℂ^χenv
    ctmrg_tol = 1e-8
    peps = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))
    wts = SUWeight(peps)
    normalize!.(peps.A, Inf)

    # Heisenberg model Hamiltonian
    ham = heisenberg_XYZ(InfiniteSquare(N1, N2); Jx = 1.0, Jy = 1.0, Jz = 1.0)
    # assert imaginary part is zero
    @assert length(imag(ham).terms) == 0
    ham = real(ham)

    # simple update
    dts = [1.0e-2, 1.0e-3, 1.0e-3, 1.0e-4]
    tols = [1.0e-7, 1.0e-8, 1.0e-8, 1.0e-8]
    maxiter = 5000
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        Dbond2 = (n == 2) ? Dbond + 2 : Dbond
        trscheme = truncerr(1.0e-10) & truncdim(Dbond2)
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        peps, wts, = simpleupdate(peps, ham, alg, wts; bipartite=false)
    end

    # measure physical quantities with CTMRG
    normalize!.(peps.A, Inf)
    env, = leading_boundary(CTMRGEnv(rand, Float64, peps, Espace), peps; tol=ctmrg_tol)
    e_site = cost_function(peps, env, ham) / (N1 * N2)
    @info "Simple update energy = $e_site"
    # benchmark data from Phys. Rev. B 94, 035133 (2016)
    @test isapprox(e_site, -0.6594; atol = 1.0e-3)

    # continue with auto differentiation
    peps_final, env_final, E_final, = fixedpoint(
        ham,
        peps,
        env;
        optimizer_alg = (; tol = gradtol, maxiter = 25),
        boundary_alg = (; maxiter = 150, svd_alg = (; rrule_alg = (; alg = :full, tol = 1.0e-5))),
    )  # sensitivity warnings and degeneracies due to SU(2)?
    ξ_h, ξ_v, = correlation_length(peps_final, env_final)
    e_site2 = E_final / (N1 * N2)
    @info "Auto diff energy = $e_site2"
    @test e_site2 ≈ E_ref atol = 1.0e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

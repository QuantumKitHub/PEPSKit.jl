using Test
using Random
using Accessors
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit
using PEPSKit: peps_normalize
using MPSKitModels: S_xx, S_yy, S_zz

# initialize parameters
Dbond = 2
χenv = 16
gradtol = 1.0e-3
# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
E_ref = -0.6602310934799577

# Heisenberg model assuming C4v symmetric PEPS and environment, which only evaluates necessary term
function heisenberg_XYZ_c4v(lattice::InfiniteSquare; kwargs...)
    return heisenberg_XYZ_c4v(ComplexF64, Trivial, lattice; kwargs...)
end
function heisenberg_XYZ_c4v(
        T::Type{<:Number}, S::Type{<:Sector}, lattice::InfiniteSquare;
        Jx = -1.0, Jy = 1.0, Jz = -1.0, spin = 1 // 2,
    )
    @assert size(lattice) == (1, 1) "only trivial unit cells supported by C4v-symmetric Hamiltonians"
    term =
        rmul!(S_xx(T, S; spin = spin), Jx) +
        rmul!(S_yy(T, S; spin = spin), Jy) +
        rmul!(S_zz(T, S; spin = spin), Jz)
    spaces = fill(domain(term)[1], (1, 1))
    return LocalOperator( # horizontal and vertical contributions are identical
        spaces, (CartesianIndex(1, 1), CartesianIndex(1, 2)) => 2 * term
    )
end

@testset "(1, 1) unit cell AD optimization" begin
    # initialize states
    Random.seed!(123)
    H = heisenberg_XYZ(InfiniteSquare())
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀)

    # optimize energy and compute correlation lengths
    peps, env, E, = fixedpoint(H, peps₀, env₀; optimizer_alg = (; tol = gradtol, maxiter = 25))
    ξ_h, ξ_v, = correlation_length(peps, env)

    @test E ≈ E_ref atol = 1.0e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "C4v AD optimization" begin
    # initialize symmetric states
    Random.seed!(123)
    symm = RotateReflect()
    H = heisenberg_XYZ_c4v(InfiniteSquare())
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    peps₀ = peps_normalize(symmetrize!(peps₀, symm))
    e₀ = initialize_random_c4v_env(peps₀, ComplexSpace(χenv))
    env₀, = leading_boundary(e₀, peps₀; alg = :c4v)

    # optimize energy and compute correlation lengths
    peps, env, E, = fixedpoint(
        H, peps₀, env₀;
        optimizer_alg = (; tol = gradtol, maxiter = 25),
        boundary_alg = (; alg = :c4v),
    )
    ξ_h, ξ_v, = correlation_length(peps, env)

    @test E ≈ E_ref atol = 1.0e-2
    @test only(ξ_h) ≈ only(ξ_v)
end

@testset "(1, 2) unit cell AD optimization" begin
    # initialize states
    Random.seed!(456)
    unitcell = (1, 2)
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    peps₀ = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond); unitcell)
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
    ctmrg_tol = 1.0e-8
    ctmrg_maxiter = 200
    peps = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (N1, N2))
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
    nstep = 5000
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        Dbond2 = (n == 2) ? Dbond + 2 : Dbond
        trunc = truncerror(; atol = 1.0e-10) & truncrank(Dbond2)
        alg = SimpleUpdate(; trunc, bipartite = false)
        peps, wts, = time_evolve(peps, ham, dt, nstep, alg, wts; tol)
    end

    # measure physical quantities with CTMRG
    normalize!.(peps.A, Inf)
    env, = leading_boundary(CTMRGEnv(rand, Float64, peps, Espace), peps; tol = ctmrg_tol, maxiter = ctmrg_maxiter)
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
        boundary_alg = (; maxiter = ctmrg_maxiter),
        gradient_alg = (; alg = :linsolver, solver_alg = (; alg = :gmres)),
    )  # sensitivity warnings and degeneracies due to SU(2)?
    ξ_h, ξ_v, = correlation_length(peps_final, env_final)
    e_site2 = E_final / (N1 * N2)
    @info "Auto diff energy = $e_site2"
    @test e_site2 ≈ E_ref atol = 1.0e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

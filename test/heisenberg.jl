using Test
using Random
using Printf
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
Dbond = 2
χenv = 16
ctm_alg = CTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=2)
)
# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
E_ref = -0.6602310934799577

@testset "(1, 1) unit cell AD optimization" begin
    # initialize states
    Random.seed!(123)
    H = heisenberg_XYZ(InfiniteSquare())
    psi_init = InfinitePEPS(2, Dbond)
    env_init = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

    # optimize energy and compute correlation lengths
    result = fixedpoint(psi_init, H, opt_alg, env_init)
    ξ_h, ξ_v, = correlation_length(result.peps, result.env)

    @test result.E ≈ E_ref atol = 1e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "(1, 2) unit cell AD optimization" begin
    # initialize states
    Random.seed!(456)
    unitcell = (1, 2)
    H_1x2 = heisenberg_XYZ(InfiniteSquare(unitcell...))
    psi_init_1x2 = InfinitePEPS(2, Dbond; unitcell)
    env_init_1x2 = leading_boundary(
        CTMRGEnv(psi_init_1x2, ComplexSpace(χenv)), psi_init_1x2, ctm_alg
    )

    # optimize energy and compute correlation lengths
    result_1x2 = fixedpoint(psi_init_1x2, H_1x2, opt_alg, env_init_1x2)
    ξ_h_1x2, ξ_v_1x2, = correlation_length(result_1x2.peps, result_1x2.env)

    @test result_1x2.E ≈ 2 * E_ref atol = 1e-2
    @test all(@. ξ_h_1x2 > 0 && ξ_v_1x2 > 0)
end

@testset "Simple update into AD optimization" begin
    # random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
    Random.seed!(789)
    N1, N2 = 2, 2
    Pspace = ℂ^2
    Vspace = ℂ^Dbond
    Espace = ℂ^χenv
    peps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))

    # normalize vertex tensors
    for ind in CartesianIndices(peps.vertices)
        peps.vertices[ind] /= norm(peps.vertices[ind], Inf)
    end
    # Heisenberg model Hamiltonian (already only includes nearest neighbor terms)
    ham = heisenberg_XYZ(InfiniteSquare(N1, N2); Jx=1.0, Jy=1.0, Jz=1.0)
    # convert to real tensors
    ham = LocalOperator(ham.lattice, Tuple(ind => real(op) for (ind, op) in ham.terms)...)

    # simple update
    dts = [1e-2, 1e-3, 4e-4, 1e-4]
    tols = [1e-7, 1e-8, 1e-8, 1e-8]
    maxiter = 5000
    for (n, (dt, tol)) in enumerate(zip(dts, tols))
        Dbond2 = (n == 1) ? Dbond + 2 : Dbond
        trscheme = truncerr(1e-10) & truncdim(Dbond2)
        alg = SimpleUpdate(dt, tol, maxiter, trscheme)
        result = simpleupdate(peps, ham, alg; bipartite=false)
        peps = result[1]
    end

    # absorb weight into site tensors and CTMRG
    peps = InfinitePEPS(peps)
    envs = CTMRGEnv(rand, Float64, peps, Espace)
    trscheme = truncerr(1e-9) & truncdim(χenv)
    envs = leading_boundary(envs, peps, CTMRG(; trscheme, ctmrgscheme=:sequential))

    # measure physical quantities
    e_site = costfun(peps, envs, ham) / (N1 * N2)
    @info @sprintf("Simple update energy = %.8f\n", e_site)
    # benchmark data from Phys. Rev. B 94, 035133 (2016)
    @test isapprox(e_site, -0.6594; atol=1e-3)

    # continue with auto differentiation
    result = fixedpoint(peps, ham, opt_alg, envs)
    ξ_h, ξ_v, = correlation_length(result.peps, result.env)
    e_site2 = result.E / (N1 * N2)
    @info @sprintf("Auto diff energy = %.8f\n", e_site)
    @test e_site2 ≈ E_ref atol = 1e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

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
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=3)
)
# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.0/state_1s_A1_j20.0_D2_chi_opt48.dat
E_ref = -0.6602310934799577

@testset "(1, 1) unit cell AD optimization" begin
    # initialize states
    Random.seed!(123)
    H = heisenberg_XYZ(InfiniteSquare())
    psi_init = InfinitePEPS(2, Dbond)
    env_init, = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

    # optimize energy and compute correlation lengths
    result = fixedpoint(H, psi_init, env_init, opt_alg)
    ξ_h, ξ_v, = correlation_length(result.peps, result.env)

    @test result.E ≈ E_ref atol = 1e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "(1, 2) unit cell AD optimization" begin
    # initialize states
    Random.seed!(456)
    unitcell = (1, 2)
    H = heisenberg_XYZ(InfiniteSquare(unitcell...))
    psi_init = InfinitePEPS(2, Dbond; unitcell)
    env_init, = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

    # optimize energy and compute correlation lengths
    result = fixedpoint(H, psi_init, env_init, opt_alg)
    ξ_h, ξ_v, = correlation_length(result.peps, result.env)

    @test result.E ≈ 2 * E_ref atol = 1e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

@testset "Simple update into AD optimization" begin
    # random initialization of 2x2 iPEPS with weights and CTMRGEnv (using real numbers)
    Random.seed!(234829)
    N1, N2 = 2, 2
    Pspace = ℂ^2
    Vspace = ℂ^Dbond
    Espace = ℂ^χenv
    wpeps = InfiniteWeightPEPS(rand, Float64, Pspace, Vspace; unitcell=(N1, N2))

    # normalize vertex tensors
    for ind in CartesianIndices(wpeps.vertices)
        wpeps.vertices[ind] /= norm(wpeps.vertices[ind], Inf)
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
        result = simpleupdate(wpeps, ham, alg; bipartite=false)
        wpeps = result[1]
    end

    # absorb weight into site tensors and CTMRG
    peps = InfinitePEPS(wpeps)
    envs₀ = CTMRGEnv(rand, Float64, peps, Espace)
    envs, = leading_boundary(envs₀, peps, SimultaneousCTMRG())

    # measure physical quantities
    e_site = cost_function(peps, envs, ham) / (N1 * N2)
    @info "Simple update energy = $e_site"
    # benchmark data from Phys. Rev. B 94, 035133 (2016)
    @test isapprox(e_site, -0.6594; atol=1e-3)

    # continue with auto differentiation
    svd_alg_gmres = SVDAdjoint(; rrule_alg=GMRES(; tol=1e-5))
    opt_alg_gmres = @set opt_alg.boundary_alg.projector_alg.svd_alg = svd_alg_gmres
    result_final = fixedpoint(ham, peps, envs, opt_alg_gmres)  # sensitivity warnings and degeneracies due to SU(2)?
    ξ_h, ξ_v, = correlation_length(result_final.peps, result_final.env)
    e_site2 = result_final.E / (N1 * N2)
    @info "Auto diff energy = $e_site2"
    @test e_site2 ≈ E_ref atol = 1e-2
    @test all(@. ξ_h > 0 && ξ_v > 0)
end

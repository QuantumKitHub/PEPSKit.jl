using Test
using TestExtras: @constinferred
using Accessors
using Random
using LinearAlgebra
using TensorKit, KrylovKit
using PEPSKit
using AMDGPU, Adapt
using PEPSKit:
    ctmrg_iteration,
    compute_gauge_fix_gauge,
    fix_phases,
    fix_relative_phases,
    calc_elementwise_convergence,
    peps_normalize,
    ScramblingEnvGauge,
    ScramblingEnvGaugeC4v
using PEPSKit.Defaults: ctmrg_tol

# initialize parameters
D = 2
χ = 16
svd_algs = [(; alg = :QRIteration), (; alg = :GKL)]
projector_algs_asymm = [:HalfInfiniteProjector] #, :FullInfiniteProjector]
unitcells = [(1, 1), (3, 4)]
atol = 1.0e-5

# test for element-wise convergence after application of fixed step
@testset "$unitcell unit cell with $(decomposition_alg.alg) and $projector_alg" for (
        unitcell, decomposition_alg, projector_alg,
    ) in Iterators.product(
        unitcells, svd_algs, projector_algs_asymm
    )
    ctm_alg = SimultaneousCTMRG(; decomposition_alg, projector_alg)

    # initialize states
    Random.seed!(2394823842)
    psi = adapt(ROCArray, InfinitePEPS(ComplexSpace(2), ComplexSpace(D); unitcell))
    @test storagetype(psi) <: ROCArray
    n = InfiniteSquareNetwork(psi)

    env_conv1, = leading_boundary(CTMRGEnv(psi, ComplexSpace(χ)), psi, ctm_alg)

    # do extra iteration and gauge fix
    env_conv2, = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg)
    env_fixed = gauge_fix(env_conv2, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fixed) ≈ 0 atol = atol

    # fix gauge of single iteration
    signs, corner_phases, edge_phases =
        compute_gauge_fix_gauge(env_conv2, env_conv1, ScramblingEnvGauge())
    gauge_fixed_iteration(env::CTMRGEnv) = fix_phases(
        ctmrg_iteration(n, env, ctm_alg)[1],
        signs, corner_phases, edge_phases,
    )

    # do gauge-fixed iteration
    env_fixed2 = @constinferred gauge_fixed_iteration(env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixed2) ≈ 0 atol = atol
end

# test same thing for C4v CTMRG
c4v_algs = [
    (:C4vQRProjector, (; alg = :Householder)),
    (:C4vEighProjector, (; alg = :DivideAndConquer)),
    (:C4vEighProjector, (; alg = :Lanczos)),
]
@testset "$(decomposition_alg.alg) and $projector_alg" for
    (projector_alg, decomposition_alg) in c4v_algs
    # initialize states
    Random.seed!(2394823842)
    ctm_alg = C4vCTMRG(;
        projector_alg, decomposition_alg, maxiter = 200,
        tol = (projector_alg == :C4vQRProjector ? 1.0e-12 : ctmrg_tol)
    )
    symm = RotateReflect()

    psi = adapt(ROCArray, InfinitePEPS(ComplexSpace(2), ComplexSpace(D)))
    @test storagetype(psi) <: ROCArray
    psi = peps_normalize(symmetrize!(psi, symm))
    @test storagetype(psi) <: ROCArray
    n = InfiniteSquareNetwork(psi)
    @test storagetype(n) <: ROCArray

    env₀ = initialize_random_c4v_env(psi, ComplexSpace(χ))
    @test storagetype(env₀) <: ROCArray
    env_conv1, info = leading_boundary(env₀, psi, ctm_alg)

    # do extra iteration to check gauge fixing
    env_conv2, info = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg) # CHECK

    env_fixed = gauge_fix(env_conv2, env_conv1, ScramblingEnvGaugeC4v())
    env_diff = calc_elementwise_convergence(env_conv1, env_fixed)
    @info "Diff between iters = $(env_diff)"
    @test env_diff ≈ 0 atol = atol

    # fix gauge of single iteration
    signs, corner_phases, edge_phases =
        compute_gauge_fix_gauge(env_conv2, env_conv1, ScramblingEnvGaugeC4v())
    gauge_fixed_iteration(env::CTMRGEnv) = fix_phases(
        ctmrg_iteration(n, env, ctm_alg)[1],
        signs, corner_phases, edge_phases,
    )

    # do gauge-fixed iteration
    env_fixed2 = @constinferred gauge_fixed_iteration(env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixed2) ≈ 0 atol = atol
end

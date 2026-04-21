using Test
using TestExtras: @constinferred
using Accessors
using Random
using LinearAlgebra
using TensorKit, KrylovKit
using PEPSKit
using PEPSKit:
    ctmrg_iteration,
    fix_phases,
    fix_relative_phases,
    fix_global_phases,
    calc_elementwise_convergence,
    peps_normalize,
    ScramblingEnvGauge,
    ScramblingEnvGaugeC4v
using PEPSKit.Defaults: ctmrg_tol

# initialize parameters
D = 2
χ = 16
svd_algs = [(; alg = :DivideAndConquer), (; alg = :iterative)]
projector_algs_asymm = [:halfinfinite] #, :fullinfinite]
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
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(D); unitcell)
    n = InfiniteSquareNetwork(psi)

    env_conv1, = leading_boundary(CTMRGEnv(psi, ComplexSpace(χ)), psi, ctm_alg)

    # do extra iteration to get SVD
    env_conv2, info = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg)
    env_fixed, signs, corner_phases, edge_phases = gauge_fix(env_conv2, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fixed) ≈ 0 atol = atol

    # fix gauge of single iteration
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
    (:c4v_qr, (; alg = :Householder)),
    (:c4v_eigh, (; alg = :QRIteration)),
    (:c4v_eigh, (; alg = :Lanczos)),
]
@testset "$(decomposition_alg.alg) and $projector_alg" for
    (projector_alg, decomposition_alg) in c4v_algs
    # initialize states
    Random.seed!(2394823842)
    ctm_alg = C4vCTMRG(;
        projector_alg, decomposition_alg, maxiter = 200,
        tol = (projector_alg == :c4v_qr ? 1.0e-12 : ctmrg_tol)
    )
    symm = RotateReflect()

    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    psi = peps_normalize(symmetrize!(psi, symm))
    n = InfiniteSquareNetwork(psi)

    env₀ = initialize_random_c4v_env(psi, ComplexSpace(χ))
    env_conv1, info = leading_boundary(env₀, psi, ctm_alg)

    # do extra iteration to check gauge fixing
    env_conv2, info = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg) # CHECK
    env_fixed, signs, corner_phases, edge_phases =
        gauge_fix(env_conv2, env_conv1, ScramblingEnvGaugeC4v())
    env_diff = calc_elementwise_convergence(env_conv1, env_fixed)
    @info "Diff between iters = $(env_diff)"
    @test env_diff ≈ 0 atol = atol

    # fix gauge of single iteration
    gauge_fixed_iteration(env::CTMRGEnv) = fix_phases(
        ctmrg_iteration(n, env, ctm_alg)[1],
        signs, corner_phases, edge_phases,
    )

    # do gauge-fixed iteration
    env_fixed2 = @constinferred gauge_fixed_iteration(env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixed2) ≈ 0 atol = atol
end

@testset "Element-wise consistency of :DivideAndConquer and :iterative" begin
    ctm_alg_iter = SimultaneousCTMRG(;
        maxiter = 200,
        decomposition_alg = (; alg = :iterative, krylovdim = χ + 10),
    )
    ctm_alg_full = SimultaneousCTMRG(; decomposition_alg = (; alg = :DivideAndConquer))

    # initialize states
    Random.seed!(91283219347)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    n = InfiniteSquareNetwork(psi)
    env₀ = CTMRGEnv(psi, ComplexSpace(χ))
    env_conv1, = leading_boundary(env₀, psi, ctm_alg_iter)

    # do extra iteration to get gauge fixing
    env_conv2_iter, info_iter = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg_iter)
    env_fix_iter, signs_iter, corner_phases_iter, edge_phases_iter =
        gauge_fix(env_conv2_iter, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fix_iter) ≈ 0 atol = atol

    env_conv2_full, info_full = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg_full)
    env_fix_full, signs_full, corner_phases_full, edge_phases_full =
        gauge_fix(env_conv2_full, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fix_full) ≈ 0 atol = atol

    # fix gauge of single iteration
    gauge_fixed_iteration_iter(env::CTMRGEnv) = fix_phases(
        ctmrg_iteration(n, env, ctm_alg_iter)[1],
        signs_iter, corner_phases_iter, edge_phases_iter,
    )
    gauge_fixed_iteration_full(env::CTMRGEnv) = fix_phases(
        ctmrg_iteration(n, env, ctm_alg_full)[1],
        signs_full, corner_phases_full, edge_phases_full,
    )

    # do gauge-fixed iteration
    env_fixedsvd_iter = @constinferred gauge_fixed_iteration_iter(env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd_iter) ≈ 0 atol = atol  # This doesn't work for x₀ = rand(size(b, 1))?

    env_fixedsvd_full = @constinferred gauge_fixed_iteration_full(env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd_full) ≈ 0 atol = atol

    # check matching decompositions
    decomposition_check = all(
        zip(info_iter.U, info_iter.S, info_iter.V, info_full.U, info_full.S, info_full.V),
    ) do (U_iter, S_iter, V_iter, U_full, S_full, V_full)
        diff = U_iter * S_iter * V_iter - U_full * S_full * V_full
        all(x -> isapprox(abs(x), 0; atol), diff.data)
    end
    @test decomposition_check

    # check matching singular values
    svalues_check = all(zip(info_iter.S, info_full.S)) do (S_iter, S_full)
        diff = S_iter - S_full
        all(x -> isapprox(abs(x), 0; atol), diff.data)
    end
    @test svalues_check

    # gauge-fix the isometries using computed relative signs
    U_iter_fix, V_iter_fix = fix_relative_phases(info_iter.U, info_iter.V, signs_iter)
    U_full_fix, V_full_fix = fix_relative_phases(info_full.U, info_full.V, signs_full)

    # check normalization of U's and V's
    Us = [info_iter.U, U_iter_fix, info_full.U, U_full_fix]
    Vs = [info_iter.V, V_iter_fix, info_full.V, V_full_fix]
    for (U, V) in zip(Us, Vs)
        U_check = all(U) do u
            uu = u' * u
            diff = uu - id(space(uu, 1))
            all(x -> isapprox(abs(x), 0; atol), diff.data)
        end
        @test U_check
        V_check = all(V) do v
            vv = v * v'
            diff = vv - id(space(vv, 1))
            all(x -> isapprox(abs(x), 0; atol), diff.data)
        end
        @test V_check
    end
end

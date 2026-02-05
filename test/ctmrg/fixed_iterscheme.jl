using Test
using TestExtras: @constinferred
using Accessors
using Random
using LinearAlgebra
using TensorKit, KrylovKit
using PEPSKit
using PEPSKit:
    FixedSVD,
    ctmrg_iteration,
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
svd_algs = [SVDAdjoint(; fwd_alg = (; alg = :sdd)), SVDAdjoint(; fwd_alg = (; alg = :iterative))]
projector_algs_asymm = [:halfinfinite] #, :fullinfinite]
unitcells = [(1, 1), (3, 4)]
atol = 1.0e-5

# test for element-wise convergence after application of fixed step
@testset "$unitcell unit cell with $(typeof(decomposition_alg.fwd_alg)) and $projector_alg" for (
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
    env_fix, signs = gauge_fix(env_conv2, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fix) ≈ 0 atol = atol

    # fix gauge of SVD
    ctm_alg_fix = gauge_fix(ctm_alg, signs, info)

    # do iteration with FixedSVD
    env_fixedsvd, = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg_fix)
    env_fixedsvd = fix_global_phases(env_fixedsvd, env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd) ≈ 0 atol = atol
end

# test same thing for C4v CTMRG
c4v_algs = [
    (:c4v_qr, QRAdjoint(; fwd_alg = (; alg = :qr))),
    (:c4v_eigh, EighAdjoint(; fwd_alg = (; alg = :qriteration))),
    (:c4v_eigh, EighAdjoint(; fwd_alg = (; alg = :lanczos))),
]
@testset "$(typeof(decomposition_alg.fwd_alg)) and $projector_alg" for
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
    env_conv1, = leading_boundary(env₀, psi, ctm_alg)

    # do extra iteration to get SVD
    env_conv2, info = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg)
    env_fix, signs = gauge_fix(env_conv2, env_conv1, ScramblingEnvGaugeC4v())
    @test calc_elementwise_convergence(env_conv1, env_fix) ≈ 0 atol = atol

    if projector_alg == :c4v_eigh
        # fix gauge of SVD
        ctm_alg_fix = gauge_fix(ctm_alg, signs, info)
        # do iteration with FixedSVD
        env_fixedsvd, = @constinferred ctmrg_iteration(n, env_conv1, ctm_alg_fix)
        env_fixedsvd = fix_global_phases(env_fixedsvd, env_conv1)
        @test calc_elementwise_convergence(env_conv1, env_fixedsvd) ≈ 0 atol = atol
    end
end

@testset "Element-wise consistency of :sdd and :iterative" begin
    ctm_alg_iter = SimultaneousCTMRG(;
        maxiter = 200,
        decomposition_alg = SVDAdjoint(; fwd_alg = (; alg = :iterative, krylovdim = χ + 10)),
    )
    ctm_alg_full = SimultaneousCTMRG(; decomposition_alg = SVDAdjoint(; fwd_alg = (; alg = :sdd)))

    # initialize states
    Random.seed!(91283219347)
    psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(D))
    n = InfiniteSquareNetwork(psi)
    env₀ = CTMRGEnv(psi, ComplexSpace(χ))
    env_conv1, = leading_boundary(env₀, psi, ctm_alg_iter)

    # do extra iteration to get SVD
    env_conv2_iter, info_iter = ctmrg_iteration(n, env_conv1, ctm_alg_iter)
    env_fix_iter, signs_iter = gauge_fix(env_conv2_iter, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fix_iter) ≈ 0 atol = atol

    env_conv2_full, info_full = ctmrg_iteration(n, env_conv1, ctm_alg_full)
    env_fix_full, signs_full = gauge_fix(env_conv2_full, env_conv1, ScramblingEnvGauge())
    @test calc_elementwise_convergence(env_conv1, env_fix_full) ≈ 0 atol = atol

    # fix gauge of SVD
    ctm_alg_fix_iter = gauge_fix(ctm_alg_iter, signs_iter, info_iter)
    ctm_alg_fix_full = gauge_fix(ctm_alg_full, signs_full, info_full)

    # do iteration with FixedSVD
    env_fixedsvd_iter, = ctmrg_iteration(n, env_conv1, ctm_alg_fix_iter)
    env_fixedsvd_iter = fix_global_phases(env_fixedsvd_iter, env_conv1)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd_iter) ≈ 0 atol = atol  # This doesn't work for x₀ = rand(size(b, 1))?

    env_fixedsvd_full, = ctmrg_iteration(n, env_conv1, ctm_alg_fix_full)
    env_fixedsvd_full = fix_global_phases(env_fixedsvd_full, env_conv1)
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

    # check normalization of U's and V's
    salg_fix_iter = ctm_alg_fix_iter.projector_alg.decomposition_alg.fwd_alg
    salg_fix_full = ctm_alg_fix_full.projector_alg.decomposition_alg.fwd_alg
    Us = [info_iter.U, salg_fix_iter.U, info_full.U, salg_fix_full.U]
    Vs = [info_iter.V, salg_fix_iter.V, info_full.V, salg_fix_full.V]
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

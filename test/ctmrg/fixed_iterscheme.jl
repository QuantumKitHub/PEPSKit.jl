using Test
using Accessors
using Random
using LinearAlgebra
using TensorKit, KrylovKit
using PEPSKit
using PEPSKit:
    FixedSVD,
    ctmrg_iteration,
    gauge_fix,
    fix_relative_phases,
    fix_global_phases,
    calc_elementwise_convergence,
    _fix_svd_algorithm

# initialize parameters
χbond = 2
χenv = 16
svd_algs = [SVDAdjoint(; fwd_alg=TensorKit.SDD()), SVDAdjoint(; fwd_alg=IterSVD())]
projector_algs = [:halfinfinite] #, :fullinfinite]
unitcells = [(1, 1), (3, 4)]
atol = 1e-5

# test for element-wise convergence after application of fixed step
@testset "$unitcell unit cell with $(typeof(svd_alg.fwd_alg)) and $projector_alg" for (
    unitcell, svd_alg, projector_alg
) in Iterators.product(
    unitcells, svd_algs, projector_algs
)
    ctm_alg = SimultaneousCTMRG(; svd_alg, projector_alg)

    # initialize states
    Random.seed!(2394823842)
    psi = InfinitePEPS(2, χbond; unitcell)
    n = InfiniteSquareNetwork(psi)

    env_conv1, = leading_boundary(CTMRGEnv(psi, ComplexSpace(χenv)), psi, ctm_alg)

    # do extra iteration to get SVD
    env_conv2, info = ctmrg_iteration(n, env_conv1, ctm_alg)
    env_fix, signs = gauge_fix(env_conv1, env_conv2)
    @test calc_elementwise_convergence(env_conv1, env_fix) ≈ 0 atol = atol

    # fix gauge of SVD
    svd_alg_fix = _fix_svd_algorithm(ctm_alg.projector_alg.svd_alg, signs, info)
    ctm_alg_fix = @set ctm_alg.projector_alg.svd_alg = svd_alg_fix
    ctm_alg_fix = @set ctm_alg_fix.projector_alg.trscheme = notrunc()

    # do iteration with FixedSVD
    env_fixedsvd, = ctmrg_iteration(n, env_conv1, ctm_alg_fix)
    env_fixedsvd = fix_global_phases(env_conv1, env_fixedsvd)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd) ≈ 0 atol = atol
end

@testset "Element-wise consistency of TensorKit.SDD and IterSVD" begin
    ctm_alg_iter = SimultaneousCTMRG(;
        maxiter=200,
        svd_alg=SVDAdjoint(; fwd_alg=IterSVD(; alg=GKL(; tol=1e-14, krylovdim=χenv + 10))),
    )
    ctm_alg_full = SimultaneousCTMRG(; svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SDD()))

    # initialize states
    Random.seed!(91283219347)
    psi = InfinitePEPS(2, χbond)
    n = InfiniteSquareNetwork(psi)
    env₀ = CTMRGEnv(psi, ComplexSpace(χenv))
    env_conv1, = leading_boundary(env₀, psi, ctm_alg_iter)

    # do extra iteration to get SVD
    env_conv2_iter, info_iter = ctmrg_iteration(n, env_conv1, ctm_alg_iter)
    env_fix_iter, signs_iter = gauge_fix(env_conv1, env_conv2_iter)
    @test calc_elementwise_convergence(env_conv1, env_fix_iter) ≈ 0 atol = atol

    env_conv2_full, info_full = ctmrg_iteration(n, env_conv1, ctm_alg_full)
    env_fix_full, signs_full = gauge_fix(env_conv1, env_conv2_full)
    @test calc_elementwise_convergence(env_conv1, env_fix_full) ≈ 0 atol = atol

    # fix gauge of SVD
    svd_alg_fix_iter = _fix_svd_algorithm(
        ctm_alg_iter.projector_alg.svd_alg, signs_iter, info_iter
    )
    ctm_alg_fix_iter = @set ctm_alg_iter.projector_alg.svd_alg = svd_alg_fix_iter
    ctm_alg_fix_iter = @set ctm_alg_fix_iter.projector_alg.trscheme = notrunc()

    svd_alg_fix_full = _fix_svd_algorithm(
        ctm_alg_full.projector_alg.svd_alg, signs_full, info_full
    )
    ctm_alg_fix_full = @set ctm_alg_full.projector_alg.svd_alg = svd_alg_fix_full
    ctm_alg_fix_full = @set ctm_alg_fix_full.projector_alg.trscheme = notrunc()

    # do iteration with FixedSVD
    env_fixedsvd_iter, = ctmrg_iteration(n, env_conv1, ctm_alg_fix_iter)
    env_fixedsvd_iter = fix_global_phases(env_conv1, env_fixedsvd_iter)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd_iter) ≈ 0 atol = atol  # This doesn't work for x₀ = rand(size(b, 1))?

    env_fixedsvd_full, = ctmrg_iteration(n, env_conv1, ctm_alg_fix_full)
    env_fixedsvd_full = fix_global_phases(env_conv1, env_fixedsvd_full)
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
    Us = [info_iter.U, svd_alg_fix_iter.fwd_alg.U, info_full.U, svd_alg_fix_full.fwd_alg.U]
    Vs = [info_iter.V, svd_alg_fix_iter.fwd_alg.V, info_full.V, svd_alg_fix_full.fwd_alg.V]
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

using Test
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
    calc_elementwise_convergence

# initialize parameters
χbond = 2
χenv = 16
svd_algs = [SVDAdjoint(; fwd_alg=TensorKit.SDD()), SVDAdjoint(; fwd_alg=IterSVD())]
projector_algs = [HalfInfiniteProjector] #, FullInfiniteProjector]
unitcells = [(1, 1), (3, 4)]
atol = 1e-5

# test for element-wise convergence after application of fixed step
@testset "$unitcell unit cell with $(typeof(svd_alg.fwd_alg)) and $projector_alg" for (
    unitcell, svd_alg, projector_alg
) in Iterators.product(
    unitcells, svd_algs, projector_algs
)
    ctm_alg = CTMRG(; svd_alg, projector_alg)

    # initialize states
    Random.seed!(2394823842)
    psi = InfinitePEPS(2, χbond; unitcell)
    env_conv1 = leading_boundary(CTMRGEnv(psi, ComplexSpace(χenv)), psi, ctm_alg)

    # do extra iteration to get SVD
    env_conv2, info = ctmrg_iteration(psi, env_conv1, ctm_alg)
    env_fix, signs = gauge_fix(env_conv1, env_conv2)
    @test calc_elementwise_convergence(env_conv1, env_fix) ≈ 0 atol = atol

    # fix gauge of SVD
    U_fix, V_fix = fix_relative_phases(info.U, info.V, signs)
    svd_alg_fix = SVDAdjoint(; fwd_alg=FixedSVD(U_fix, info.S, V_fix))
    ctm_alg_fix = CTMRG(; projector_alg, svd_alg=svd_alg_fix, trscheme=notrunc())

    # do iteration with FixedSVD
    env_fixedsvd, = ctmrg_iteration(psi, env_conv1, ctm_alg_fix)
    env_fixedsvd = fix_global_phases(env_conv1, env_fixedsvd)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd) ≈ 0 atol = atol
end

@testset "Element-wise consistency of TensorKit.SDD and IterSVD" begin
    ctm_alg_iter = CTMRG(;
        maxiter=200,
        svd_alg=SVDAdjoint(; fwd_alg=IterSVD(; alg=GKL(; tol=1e-14, krylovdim=χenv + 10))),
    )
    ctm_alg_full = CTMRG(; svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SDD()))

    # initialize states
    Random.seed!(91283219347)
    psi = InfinitePEPS(2, χbond)
    env_init = CTMRGEnv(psi, ComplexSpace(χenv))
    env_conv1 = leading_boundary(env_init, psi, ctm_alg_iter)

    # do extra iteration to get SVD
    env_conv2_iter, info_iter = ctmrg_iteration(psi, env_conv1, ctm_alg_iter)
    env_fix_iter, signs_iter = gauge_fix(env_conv1, env_conv2_iter)
    @test calc_elementwise_convergence(env_conv1, env_fix_iter) ≈ 0 atol = atol

    env_conv2_full, info_full = ctmrg_iteration(psi, env_conv1, ctm_alg_full)
    env_fix_full, signs_full = gauge_fix(env_conv1, env_conv2_full)
    @test calc_elementwise_convergence(env_conv1, env_fix_full) ≈ 0 atol = atol

    # fix gauge of SVD
    U_fix_iter, V_fix_iter = fix_relative_phases(info_iter.U, info_iter.V, signs_iter)
    svd_alg_fix_iter = SVDAdjoint(; fwd_alg=FixedSVD(U_fix_iter, info_iter.S, V_fix_iter))
    ctm_alg_fix_iter = CTMRG(; svd_alg=svd_alg_fix_iter, trscheme=notrunc())

    U_fix_full, V_fix_full = fix_relative_phases(info_full.U, info_full.V, signs_full)
    svd_alg_fix_full = SVDAdjoint(; fwd_alg=FixedSVD(U_fix_full, info_full.S, V_fix_full))
    ctm_alg_fix_full = CTMRG(; svd_alg=svd_alg_fix_full, trscheme=notrunc())

    # do iteration with FixedSVD
    env_fixedsvd_iter, = ctmrg_iteration(psi, env_conv1, ctm_alg_fix_iter)
    env_fixedsvd_iter = fix_global_phases(env_conv1, env_fixedsvd_iter)
    @test calc_elementwise_convergence(env_conv1, env_fixedsvd_iter) ≈ 0 atol = atol  # This doesn't work for x₀ = rand(size(b, 1))?

    env_fixedsvd_full, = ctmrg_iteration(psi, env_conv1, ctm_alg_fix_full)
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
    Us = [info_iter.U, U_fix_iter, info_full.U, U_fix_full]
    Vs = [info_iter.V, V_fix_iter, info_full.V, V_fix_full]
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

using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit:
    FixedSVD,
    ctmrg_iter,
    gauge_fix,
    fix_relative_phases,
    fix_global_phases,
    check_elementwise_convergence

# initialize parameters
χbond = 2
χenv = 16
svd_algs = [SVDAdjoint(; fwd_alg=TensorKit.SVD()), SVDAdjoint(; fwd_alg=IterSVD())]
unitcells = [(1, 1), (3, 4)]

# test for element-wise convergence after application of fixed step
@testset "$unitcell unit cell with $(typeof(svd_alg.fwd_alg))" for (unitcell, svd_alg) in
                                                                   Iterators.product(
    unitcells, svd_algs
)
    ctm_alg = CTMRG(; tol=1e-12, verbosity=1, ctmrgscheme=:simultaneous, svd_alg)

    # initialize states
    Random.seed!(2394823842)
    psi = InfinitePEPS(2, χbond; unitcell)
    env_conv1 = leading_boundary(CTMRGEnv(psi; Venv=ComplexSpace(χenv)), psi, ctm_alg)

    # do extra iteration to get SVD
    env_conv2, info = ctmrg_iter(psi, env_conv1, ctm_alg)
    env_fix, signs = gauge_fix(env_conv1, env_conv2)
    @test check_elementwise_convergence(env_conv1, env_fix; atol=1e-6)

    # fix gauge of SVD
    U_fix, V_fix = fix_relative_phases(info.U, info.V, signs)
    svd_alg_fix = SVDAdjoint(; fwd_alg=FixedSVD(U_fix, info.S, V_fix))
    ctm_alg_fix = CTMRG(; svd_alg=svd_alg_fix, trscheme=notrunc(), ctmrgscheme=:simultaneous)

    # do iteration with FixedSVD
    env_fixedsvd, = ctmrg_iter(psi, env_conv1, ctm_alg_fix)
    env_fixedsvd = fix_global_phases(env_conv1, env_fixedsvd)
    @test check_elementwise_convergence(env_conv1, env_fixedsvd; atol=1e-6)
end

# TODO: Why doesn't fixed work with IterSVD?
##
# ctm_alg = CTMRG(;
#     tol=1e-12,
#     miniter=4,
#     maxiter=100,
#     verbosity=1,
#     ctmrgscheme=:simultaneous,
#     svd_alg=SVDAdjoint(; fwd_alg=IterSVD()),
# )

# # initialize states
# Random.seed!(91283219347)
# psi = InfinitePEPS(2, χbond)
# env_conv1 = leading_boundary(CTMRGEnv(psi; Venv=ComplexSpace(χenv)), psi, ctm_alg);

# # do extra iteration to get SVD
# env_conv2, info = ctmrg_iter(psi, env_conv1, ctm_alg);
# env_fix, signs = gauge_fix(env_conv1, env_conv2);
# @test check_elementwise_convergence(env_conv1, env_fix)

# # fix gauge of SVD
# U_fix, V_fix = fix_relative_phases(info.U, info.V, signs);
# svd_alg_fix = SVDAdjoint(; fwd_alg=FixedSVD(U_fix, info.S, V_fix));
# ctm_alg_fix = CTMRG(; svd_alg=svd_alg_fix, trscheme=notrunc(), ctmrgscheme=:simultaneous);

# # do iteration with FixedSVD
# env_fixedsvd, = ctmrg_iter(psi, env_conv1, ctm_alg_fix);
# env_fixedsvd = fix_global_phases(env_conv1, env_fixedsvd);
# @test check_elementwise_convergence(env_conv1, env_fixedsvd)
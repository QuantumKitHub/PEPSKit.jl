using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = CTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=2,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10)),
    ctmrgscheme=:simultaneous,
)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6), iterscheme=:fixed),
    reuse_env=true,
)

# initialize states
Random.seed!(91283219347)
H = heisenberg_XYZ(InfiniteSquare())
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)
ξ_h, ξ_v, = correlation_length(result.peps, result.env)

@test result.E ≈ -0.6694421 atol = 1e-2
@test all(@. ξ_h > 0 && ξ_v > 0)

# same test but for 1x2 unit cell
unitcell = (1, 2)
H_1x2 = heisenberg_XYZ(InfiniteSquare(unitcell...))
psi_init_1x2 = InfinitePEPS(2, χbond; unitcell)
env_init_1x2 = leading_boundary(
    CTMRGEnv(psi_init_1x2, ComplexSpace(χenv)), psi_init_1x2, ctm_alg
)
result_1x2 = fixedpoint(psi_init_1x2, H_1x2, opt_alg, env_init_1x2)
ξ_h_1x2, ξ_v_1x2, = correlation_length(result_1x2.peps, result_1x2.env)

@test result_1x2.E ≈ 2 * result.E atol = 1e-2
@test all(@. ξ_h_1x2 > 0 && ξ_v_1x2 > 0)

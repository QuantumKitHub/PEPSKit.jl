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
    gradient_alg=LinSolver(; solver=GMRES(; tol=1e-6, maxiter=100), iterscheme=:fixed),
    reuse_env=true,
)

# initialize states
Random.seed!(91283219347)
H = square_lattice_heisenberg()
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg);

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

@test result.E ≈ -0.6694421 atol = 1e-2

# same test but for 2x2 unit cell
H_2x2 = square_lattice_heisenberg(; unitcell=(2, 2))
psi_init_2x2 = InfinitePEPS(2, χbond; unitcell=(2, 2))
env_init_2x2 = leading_boundary(
    CTMRGEnv(psi_init_2x2, ComplexSpace(χenv)), psi_init_2x2, ctm_alg
)
result_2x2 = fixedpoint(psi_init_2x2, H_2x2, opt_alg, env_init_2x2)

@test result_2x2.E ≈ 4 * result.E atol = 1e-2

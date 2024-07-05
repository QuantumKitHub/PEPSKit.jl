using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# Initialize parameters
H = square_lattice_heisenberg()
χbond = 2
χenv = 16
projector_alg = ProjectorAlg(; trscheme=truncdim(χenv))
ctm_alg = CTMRG(; tol=1e-10, miniter=4, maxiter=100, verbosity=1, projector_alg)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity=2,
)

# initialize states
Random.seed!(91283219347)
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init; Venv=ComplexSpace(χenv)), psi_init, ctm_alg)

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

@test result.E ≈ -0.6694421 atol = 1e-2

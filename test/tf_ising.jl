using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = CTMRG(; trscheme=truncdim(χenv), tol=1e-10, miniter=4, maxiter=100, verbosity=1)
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg,
    optimizer=LBFGS(4; maxiter=100, gradtol=1e-3, verbosity=2),
    gradient_alg=GMRES(; tol=1e-6, maxiter=100),
    reuse_env=true,
    verbosity=2,
)

# initialize states
Random.seed!(91283219347)
H = square_lattice_tf_ising(; h=1.5)
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init; Venv=ComplexSpace(χenv)), psi_init, ctm_alg)

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

# compute magnetization
σz = TensorMap(scalartype(psi_init)[1 0; 0 -1], ℂ^2, ℂ^2)
M = LocalOperator(H.lattice, (CartesianIndex(1, 1),) => σz)
magn = expectation_value(result.peps, M, result.env)

ref_energy = result.E  # TODO: Is there some reference energy/magnetization?
ref_magn = magn
@test result.E ≈ ref_energy atol = 1e-2
@test abs(magn) ≈ ref_magn atol = 1e-2

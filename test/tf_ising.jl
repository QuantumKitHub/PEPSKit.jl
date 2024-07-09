using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# References
# ----------
# Classical Simulation of Infinite-Size Quantum Lattice Systems in Two Spatial Dimensions
# J. Jordan, R. Orús, G. Vidal, F. Verstraete, and J. I. Cirac
# Phys. Rev. Lett. 101, 250602 – Published 18 December 2008
# (values estimated from plots)
# (factor of 2 in the energy and magnetisation due to convention differences)
h = 0.5
e = -1.05
mᶻ = 0.98

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
H = square_lattice_tf_ising(; h)
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init; Venv=ComplexSpace(χenv)), psi_init, ctm_alg)

# find fixedpoint
result = fixedpoint(psi_init, H, opt_alg, env_init)

# compute magnetization
σz = TensorMap(scalartype(psi_init)[1 0; 0 -1], ℂ^2, ℂ^2)
M = LocalOperator(H.lattice, (CartesianIndex(1, 1),) => σz)
magn = expectation_value(result.peps, M, result.env)

@test result.E ≈ e atol = 5e-2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mᶻ atol = 5e-2

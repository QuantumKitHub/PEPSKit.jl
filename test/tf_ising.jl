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
# (factor of 2 in the energy due to convention differences)
g = 3.1
e = -1.6417 * 2
mˣ = 0.91

# initialize parameters
χbond = 2
χenv = 16
ctm_alg = SimultaneousCTMRG()
opt_alg = PEPSOptimize(;
    boundary_alg=ctm_alg, optimizer=LBFGS(4; gradtol=1e-3, verbosity=3)
)

# initialize states
H = transverse_field_ising(InfiniteSquare(); g)
Random.seed!(2928528935)
psi_init = InfinitePEPS(2, χbond)
env_init = leading_boundary(CTMRGEnv(psi_init, ComplexSpace(χenv)), psi_init, ctm_alg)

# find fixedpoint
result = fixedpoint(H, psi_init, env_init, opt_alg)
ξ_h, ξ_v, = correlation_length(result.peps, result.env)

# compute magnetization
σx = TensorMap(scalartype(psi_init)[0 1; 1 0], ℂ^2, ℂ^2)
M = LocalOperator(H.lattice, (CartesianIndex(1, 1),) => σx)
magn = expectation_value(result.peps, M, result.env)

@test result.E ≈ e atol = 1e-2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mˣ atol = 5e-2

# find fixedpoint in polarized phase and compute correlations lengths
H_polar = transverse_field_ising(InfiniteSquare(); g=4.5)
result_polar = fixedpoint(H_polar, psi_init, env_init, opt_alg)
ξ_h_polar, ξ_v_polar, = correlation_length(result_polar.peps, result_polar.env)
@test ξ_h_polar < ξ_h
@test ξ_v_polar < ξ_v

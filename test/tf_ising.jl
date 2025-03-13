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
χenv = 14
gradtol = 1e-3

# initialize states
H = transverse_field_ising(InfiniteSquare(); g)
Random.seed!(2928528935)
peps₀ = InfinitePEPS(2, χbond)
env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀)

# find fixedpoint
peps, env, E, = fixedpoint(H, peps₀, env₀; tol=gradtol)
ξ_h, ξ_v, = correlation_length(peps, env)

# compute magnetization
σx = TensorMap(scalartype(peps₀)[0 1; 1 0], ℂ^2, ℂ^2)
M = LocalOperator(H.lattice, (CartesianIndex(1, 1),) => σx)
magn = expectation_value(peps, M, env)

@test E ≈ e atol = 1e-2
@test imag(magn) ≈ 0 atol = 1e-6
@test abs(magn) ≈ mˣ atol = 5e-2

# find fixedpoint in polarized phase and compute correlations lengths
H_polar = transverse_field_ising(InfiniteSquare(); g=4.5)
peps_polar, env_polar, = fixedpoint(H_polar, peps₀, env₀; tol=gradtol)
ξ_h_polar, ξ_v_polar, = correlation_length(peps_polar, env_polar)
@test ξ_h_polar < ξ_h
@test ξ_v_polar < ξ_v

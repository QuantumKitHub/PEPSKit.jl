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
σz = TensorMap(scalartype(peps₀)[1 0; 0 -1], ℂ^2, ℂ^2)
Mx = LocalOperator(physicalspace(H), (CartesianIndex(1, 1),) => σx)
Mz = LocalOperator(physicalspace(H), (CartesianIndex(1, 1),) => σz)
magnx = expectation_value(peps, Mx, env)
magnz = expectation_value(peps, Mz, env)

@test E ≈ e atol = 1e-2
@test imag(magnx) ≈ 0 atol = 1e-6
@test abs(magnx) ≈ mˣ atol = 5e-2

# compute connected correlation functions
corrh =
    correlator(
        peps,
        (σz, σz),
        CartesianIndex(1, 1),
        CartesianIndex(1, 2):CartesianIndex(1, 21),
        env,
    ) .- magnz^2
corrh_2 =
    correlator(
        peps, σz ⊗ σz, CartesianIndex(1, 1), CartesianIndex(1, 2):CartesianIndex(1, 21), env
    ) .- magnz^2
corrv =
    correlator(
        peps,
        (σz, σz),
        CartesianIndex(1, 1),
        CartesianIndex(2, 1):CartesianIndex(21, 1),
        env,
    ) .- magnz^2
corrv_2 =
    correlator(
        peps, σz ⊗ σz, CartesianIndex(1, 1), CartesianIndex(2, 1):CartesianIndex(21, 1), env
    ) .- magnz^2

@test corrh[end] ≈ 0.0 atol = 1e-5
@test 1 / log(corrh[18] / corrh[19]) ≈ ξ_h[1] atol = 2e-2 # test correlation length far away from short-range effects
@test corrv[end] ≈ 0.0 atol = 1e-5
@test 1 / log(corrv[18] / corrv[19]) ≈ ξ_v[1] atol = 3e-2 # test correlation length far away from short-range effects
@test maximum(abs.(corrh - corrh_2)) < 1e-14
@test maximum(abs.(corrv - corrv_2)) < 1e-14
@test maximum(abs.(corrv - corrh)) < 1e-4

# Change from specific values and distances to a range
corrh_int =
    correlator(peps, (σz, σz), CartesianIndex(1, 1), CartesianIndex(1, 21), env) - magnz^2
corrv_int =
    correlator(peps, (σz, σz), CartesianIndex(1, 1), CartesianIndex(21, 1), env) - magnz^2

@test corrh_int ≈ corrh[20]
@test corrv_int ≈ corrv[20]

@test_broken correlator(peps, (σz, σz), CartesianIndex(1, 1), CartesianIndex(2, 2), env)

# find fixedpoint in polarized phase and compute correlations lengths
H_polar = transverse_field_ising(InfiniteSquare(); g=4.5)
peps_polar, env_polar, = fixedpoint(H_polar, peps₀, env₀; tol=gradtol)
ξ_h_polar, ξ_v_polar, = correlation_length(peps_polar, env_polar)
@test ξ_h_polar < ξ_h
@test ξ_v_polar < ξ_v

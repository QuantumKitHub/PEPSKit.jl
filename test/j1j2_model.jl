using Test
using Random
using PEPSKit
using TensorKit
using KrylovKit
using OptimKit

# initialize parameters
χbond = 2
χenv = 12

# initialize states
Random.seed!(91283219347)
H = j1_j2(InfiniteSquare(); J2=0.25)
peps₀ = product_peps(2, χbond; noise_amp=1e-1)
peps₀ = symmetrize!(peps₀, RotateReflect())
env₀, = leading_boundary(CTMRGEnv(peps₀, ComplexSpace(χenv)), peps₀)

# find fixedpoint
peps, env, E, = fixedpoint(
    H,
    peps₀,
    env₀;
    tol=1e-3,
    gradient_alg=(; iterscheme=:diffgauge),
    symmetrization=RotateReflect()
)
ξ_h, ξ_v, = correlation_length(peps, env)

# compare against Juraj Hasik's data:
# https://github.com/jurajHasik/j1j2_ipeps_states/blob/main/single-site_pg-C4v-A1/j20.25/state_1s_A1_j20.25_D2_chi_opt48.dat
ξ_ref = -1 / log(0.2723596743547324)
@test E ≈ -0.5618837021945925 atol = 1e-3
@test all(@. isapprox(ξ_h, ξ_ref; atol=1e-1) && isapprox(ξ_v, ξ_ref; atol=1e-1))
@test ξ_h ≈ ξ_v atol = 1e-6  # Test symmetrization of optimized PEPS and environment

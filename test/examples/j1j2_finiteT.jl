using Test
using Random
using LinearAlgebra
using TensorKit
import MPSKitModels: σˣ, σᶻ
using PEPSKit

Random.seed!(10235876)

# Benchmark energy from high-temperature expansion
# at β = 0.3, 0.6
# Physical Review B 86, 045139 (2012) Fig. 15-16
bm = [-0.1235, -0.213]

function converge_env(state, χ::Int)
    trunc1 = truncrank(χ) & truncerror(; atol = 1.0e-12)
    env0 = CTMRGEnv(randn, Float64, state, Vect[SU2Irrep](0 => 1))
    env, = leading_boundary(env0, state; alg = :sequential, trunc = trunc1, tol = 1.0e-10)
    return env
end

Nr, Nc = 2, 2
ham = j1_j2_model(
    Float64, SU2Irrep, InfiniteSquare(Nr, Nc);
    J1 = 1.0, J2 = 0.5, sublattice = false
)
pepo0 = PEPSKit.infinite_temperature_density_matrix(ham)
wts0 = SUWeight(pepo0)
# 7 = 1 (spin-0) + 2 x 3 (spin-1)
trunc_pepo = truncrank(7) & truncerror(; atol = 1.0e-12)
check_interval = 100

# PEPO approach
dt, nstep = 1.0e-3, 600
alg = SimpleUpdate(;
    ψ0 = pepo0, env0 = wts0, H = ham, dt, nstep, trunc = trunc_pepo,
    gate_bothsides = true, check_interval
)
pepo, wts, = time_evolve(alg)
env = converge_env(InfinitePartitionFunction(pepo), 16)
energy = expectation_value(pepo, ham, env) / (Nr * Nc)
@info "β = $(dt * nstep): tr(ρH) = $(energy)"
@test energy ≈ bm[2] atol = 5.0e-3

# PEPS (purified PEPO) approach
alg = SimpleUpdate(;
    ψ0 = pepo0, env0 = wts0, H = ham, dt, nstep, trunc = trunc_pepo,
    gate_bothsides = false, check_interval
)
pepo, wts, = time_evolve(alg)
env = converge_env(InfinitePartitionFunction(pepo), 16)
energy = expectation_value(pepo, ham, env) / (Nr * Nc)
@info "β = $(dt * nstep) / 2: tr(ρH) = $(energy)"
@test energy ≈ bm[1] atol = 5.0e-3

env = converge_env(InfinitePEPS(pepo), 16)
energy = expectation_value(pepo, ham, pepo, env) / (Nr * Nc)
@info "β = $(dt * nstep): ⟨ρ|ham|ρ⟩ = $(energy)"
@test energy ≈ bm[2] atol = 5.0e-3

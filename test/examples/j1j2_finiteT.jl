using Test
using LinearAlgebra
using TensorKit
import MPSKitModels: σˣ, σᶻ
using PEPSKit

# Benchmark energy from high-temperature expansion
# at β = 0.3, 0.6
# Physical Review B 86, 045139 (2012) Fig. 15-16
bm = [-0.1235, -0.213]

function converge_env(state, χ::Int)
    trunc1 = truncrank(χ) & truncerror(; atol = 1.0e-12)
    env0 = CTMRGEnv(ones, Float64, state, Vect[SU2Irrep](0 => 1))
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
alg = SimpleUpdate(; trunc = trunc_pepo, purified = false)
evolver = TimeEvolver(pepo0, ham, dt, nstep, alg, wts0)
pepo, wts, info = time_evolve(evolver; check_interval)
env = converge_env(InfinitePartitionFunction(pepo), 16)
energy = expectation_value(pepo, ham, env) / (Nr * Nc)
@info "β = $(dt * nstep): tr(ρH) = $(energy)"
@test dt * nstep ≈ info.t
@test energy ≈ bm[2] atol = 5.0e-3

# PEPS (purified PEPO) approach
alg = SimpleUpdate(; trunc = trunc_pepo, purified = true)
evolver = TimeEvolver(pepo0, ham, dt, nstep, alg, wts0)
pepo, wts, info = time_evolve(evolver; check_interval)
env = converge_env(InfinitePartitionFunction(pepo), 16)
energy = expectation_value(pepo, ham, env) / (Nr * Nc)
@info "β = $(dt * nstep) / 2: tr(ρH) = $(energy)"
@test energy ≈ bm[1] atol = 5.0e-3

env = converge_env(InfinitePEPS(pepo), 16)
energy = expectation_value(pepo, ham, pepo, env) / (Nr * Nc)
@info "β = $(dt * nstep): ⟨ρ|H|ρ⟩ = $(energy)"
@test dt * nstep ≈ info.t
@test energy ≈ bm[2] atol = 5.0e-3

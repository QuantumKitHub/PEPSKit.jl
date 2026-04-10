using Test
using LinearAlgebra
using TensorKit
import MPSKitModels: σˣ, σᶻ
using PEPSKit

# Benchmark energy from high-temperature expansion
const βs = [0.2, 0.4, 0.6]
const bm = [-0.08624893, -0.15688984, -0.21300888]

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
dt, nstep, check_interval = 5.0e-3, 40, 40

@testset "Simple update" begin
    # 7 = 1 (spin-0) + 2 x 3 (spin-1)
    trunc_pepo = truncrank(7) & truncerror(; atol = 1.0e-12)
    alg = SimpleUpdate(; trunc = trunc_pepo, purified = true)
    pepo, wts = deepcopy(pepo0), deepcopy(wts0)
    for (β, bme) in zip(βs, bm)
        t0 = β - βs[1]
        pepo, wts, info = time_evolve(pepo, ham, dt, nstep, alg, wts; t0, check_interval)
        # measure energy
        env = converge_env(InfinitePEPS(pepo), 16)
        energy = expectation_value(pepo, ham, pepo, env) / (Nr * Nc)
        @info "β = $(info.t): ⟨ρ|H|ρ⟩ = $(energy)"
        @test energy ≈ bme atol = 5.0e-3
    end
end

@testset "Neighbourhood tensor update" begin
    trunc_pepo = truncrank(4) & truncerror(; atol = 1.0e-12)
    opt_alg = ALSTruncation(; trunc = trunc_pepo, tol = 1.0e-10)
    alg = NeighbourUpdate(; opt_alg, bondenv_alg = NNEnv())
    pepo = deepcopy(pepo0)
    for (β, bme) in zip(βs, bm)
        t0 = β - βs[1]
        pepo, info = time_evolve(pepo, ham, dt, nstep, alg; t0, check_interval)
        # measure energy
        env = converge_env(InfinitePEPS(pepo), 16)
        energy = expectation_value(pepo, ham, pepo, env) / (Nr * Nc)
        @info "β = $(info.t): ⟨ρ|H|ρ⟩ = $(energy)"
        @test energy ≈ bme atol = 2.0e-2
    end
end

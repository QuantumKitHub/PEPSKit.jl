using Test
using LinearAlgebra
using TensorKit
import MPSKitModels: σˣ, σᶻ
using PEPSKit

# Benchmark data of [σx, σz] from HOTRG
# Physical Review B 86, 045139 (2012) Fig. 15-16
bm_β = [0.5632, 0.0]
bm_2β = [0.5297, 0.8265]

# only contains 2-site gates, convenient for time evolution
function tfising_model(T::Type{<:Number}, lattice::InfiniteSquare; J = 1.0, g = 1.0)
    pspace, S = ℂ^2, Trivial
    ZZ = rmul!(σᶻ(T, S) ⊗ σᶻ(T, S), -J)
    X = rmul!(σˣ(T, S), g * -J)
    unit = TensorKit.id(pspace)
    spaces = fill(pspace, (lattice.Nrows, lattice.Ncols))
    gate = ZZ + (unit ⊗ X + X ⊗ unit) / 4
    return PEPSKit.nearest_neighbour_hamiltonian(spaces, gate)
end

function converge_env(state, χ::Int)
    trunc1 = truncrank(4) & truncerror(; atol = 1.0e-12)
    env0 = CTMRGEnv(ones, Float64, state, ℂ^1)
    env, = leading_boundary(env0, state; alg = :sequential, trunc = trunc1, tol = 1.0e-10)
    trunc2 = truncrank(χ) & truncerror(; atol = 1.0e-12)
    env, = leading_boundary(env, state; alg = :sequential, trunc = trunc2, tol = 1.0e-10)
    return env
end

function measure_mag(pepo::InfinitePEPO, env::CTMRGEnv; purified::Bool = false)
    r, c = 1, 1
    lattice = physicalspace(pepo)
    Mx = LocalOperator(lattice, ((r, c),) => σˣ(Float64, Trivial))
    Mz = LocalOperator(lattice, ((r, c),) => σᶻ(Float64, Trivial))
    if purified
        magx = expectation_value(pepo, Mx, pepo, env)
        magz = expectation_value(pepo, Mz, pepo, env)
    else
        magx = expectation_value(pepo, Mx, env)
        magz = expectation_value(pepo, Mz, env)
    end
    return [magx, magz]
end

Nr, Nc = 2, 2
ham = tfising_model(Float64, InfiniteSquare(Nr, Nc); J = 1.0, g = 2.0)
pepo0 = PEPSKit.infinite_temperature_density_matrix(ham)
wts0 = SUWeight(pepo0)

trunc_pepo = truncrank(8) & truncerror(; atol = 1.0e-12)

dt, nstep = 1.0e-3, 400
β = dt * nstep

# when g = 2, β = 0.4 and 2β = 0.8 belong to two phases (without and with nonzero σᶻ)
@testset "Finite-T SU (bipartite = $(bipartite))" for bipartite in (true, false)
    # PEPO approach: results at β, or T = 2.5
    alg = SimpleUpdate(; trunc = trunc_pepo, purified = false, bipartite)
    pepo, wts, info = time_evolve(pepo0, ham, dt, nstep, alg, wts0)

    ## BP gauge fixing
    bp_alg = BeliefPropagation(; maxiter = 100, tol = 1.0e-9, bipartite)
    bp_env, = leading_boundary(BPEnv(ones, Float64, pepo), pepo, bp_alg)
    pepo, = gauge_fix(pepo, BPGauge(), bp_env)

    env = converge_env(InfinitePartitionFunction(pepo), 16)
    result_β = measure_mag(pepo, env)
    @info "tr(σ(x,z)ρ) at T = $(1 / β): $(result_β)."
    @test β ≈ info.t
    @test isapprox(abs.(result_β), bm_β, rtol = 1.0e-2)

    # continue to get results at 2β, or T = 1.25
    pepo, wts, info = time_evolve(pepo, ham, dt, nstep, alg, wts; t0 = β)
    env = converge_env(InfinitePartitionFunction(pepo), 16)
    result_2β = measure_mag(pepo, env)
    @info "tr(σ(x,z)ρ) at T = $(1 / (2β)): $(result_2β)."
    @test 2 * β ≈ info.t
    @test isapprox(abs.(result_2β), bm_2β, rtol = 1.0e-4)

    # Purification approach: results at 2β, or T = 1.25
    alg = SimpleUpdate(; trunc = trunc_pepo, purified = true, bipartite)
    pepo, wts, info = time_evolve(pepo0, ham, dt, 2 * nstep, alg, wts0)
    env = converge_env(InfinitePEPS(pepo), 8)
    result_2β′ = measure_mag(pepo, env; purified = true)
    @info "⟨ρ|σ(x,z)|ρ⟩ at T = $(1 / (2β)): $(result_2β′)."
    @test 2 * β ≈ info.t
    @test isapprox(abs.(result_2β′), bm_2β, rtol = 1.0e-2)
end

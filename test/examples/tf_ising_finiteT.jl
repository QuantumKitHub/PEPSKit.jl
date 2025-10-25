using Test
using Random
using LinearAlgebra
using TensorKit
import MPSKitModels: σˣ, σᶻ
using PEPSKit

Random.seed!(10235876)

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
    trscheme1 = truncdim(4) & truncerr(1.0e-12)
    env0 = CTMRGEnv(randn, Float64, state, ℂ^4)
    env, = leading_boundary(env0, state; alg = :sequential, trscheme = trscheme1, tol = 1.0e-10)
    trscheme2 = truncdim(χ) & truncerr(1.0e-12)
    env, = leading_boundary(env, state; alg = :sequential, trscheme = trscheme2, tol = 1.0e-10)
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

trscheme_pepo = truncdim(8) & truncerr(1.0e-12)

dt, nstep = 1.0e-3, 400
β = dt * nstep

# when g = 2, β = 0.4 and 2β = 0.8 belong to two phases (without and with nonzero σᶻ)

# PEPO approach: results at β, or T = 2.5
alg = SimpleUpdate(; trscheme = trscheme_pepo, gate_bothsides = true)
pepo, wts, = time_evolve(pepo0, ham, dt, nstep, alg, wts0)
env = converge_env(InfinitePartitionFunction(pepo), 16)
result_β = measure_mag(pepo, env)
@info "Magnetization at T = $(1 / β)" result_β
@test isapprox(abs.(result_β), bm_β, rtol = 1.0e-2)

# continue to get results at 2β, or T = 1.25
pepo, wts, = time_evolve(pepo, ham, dt, nstep, alg, wts)
env = converge_env(InfinitePartitionFunction(pepo), 16)
result_2β = measure_mag(pepo, env)
@info "Magnetization at T = $(1 / (2β))" result_2β
@test isapprox(abs.(result_2β), bm_2β, rtol = 1.0e-4)

# Purification approach: results at 2β, or T = 1.25
alg = SimpleUpdate(; trscheme = trscheme_pepo, gate_bothsides = false)
pepo, = time_evolve(pepo0, ham, dt, 2 * nstep, alg, wts0)
env = converge_env(InfinitePEPS(pepo), 8)
result_2β′ = measure_mag(pepo, env; purified = true)
@info "Magnetization at T = $(1 / (2β)) (purification approach)" result_2β′
@test isapprox(abs.(result_2β′), bm_2β, rtol = 1.0e-2)

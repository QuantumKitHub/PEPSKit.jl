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

function converge_env(ψ, χ::Int)
    trscheme1 = truncdim(4) & truncerr(1.0e-12)
    env0 = CTMRGEnv(randn, Float64, ψ, ℂ^4)
    env, = leading_boundary(env0, ψ; alg = :sequential, trscheme = trscheme1, tol = 1.0e-10)
    trscheme2 = truncdim(χ) & truncerr(1.0e-12)
    env, = leading_boundary(env, ψ; alg = :sequential, trscheme = trscheme2, tol = 1.0e-10)
    return env
end

function measure_mag(ψ::InfinitePEPO, env::CTMRGEnv; purified::Bool = false)
    r, c = 1, 1
    lattice = physicalspace(ψ)
    Mx = LocalOperator(lattice, ((r, c),) => σˣ(Float64, Trivial))
    Mz = LocalOperator(lattice, ((r, c),) => σᶻ(Float64, Trivial))
    if purified
        magx = expectation_value(ψ, Mx, ψ, env)
        magz = expectation_value(ψ, Mz, ψ, env)
    else
        magx = expectation_value(ψ, Mx, env)
        magz = expectation_value(ψ, Mz, env)
    end
    return [magx, magz]
end

Nr, Nc = 2, 2
H = tfising_model(Float64, InfiniteSquare(Nr, Nc); J = 1.0, g = 2.0)
ψ0 = PEPSKit.infinite_temperature_density_matrix(H)
wts0 = SUWeight(ψ0)

trscheme_ψ = truncdim(8) & truncerr(1.0e-12)

dt, nstep = 1.0e-3, 400
β = dt * nstep

# when g = 2, β = 0.4 and 2β = 0.8 belong to two phases (without and with nonzero σᶻ)

# PEPO approach: results at β, or T = 2.5
alg = SimpleUpdate(;
    ψ0, env0 = wts0, H, dt, nstep,
    trscheme = trscheme_ψ, gate_bothsides = true
)
ψ, wts, = time_evolve(alg)
env = converge_env(InfinitePartitionFunction(ψ), 16)
result_β = measure_mag(ψ, env)
@info "Magnetization at T = $(1 / β)" result_β
@test isapprox(abs.(result_β), bm_β, rtol = 1.0e-2)

# continue to get results at 2β, or T = 1.25
alg = SimpleUpdate(;
    ψ0 = ψ, env0 = wts, H, dt, nstep,
    trscheme = trscheme_ψ, gate_bothsides = true
)
ψ, wts, = time_evolve(alg)
env = converge_env(InfinitePartitionFunction(ψ), 16)
result_2β = measure_mag(ψ, env)
@info "Magnetization at T = $(1 / (2β))" result_2β
@test isapprox(abs.(result_2β), bm_2β, rtol = 1.0e-4)

# Purification approach: results at 2β, or T = 1.25
alg = SimpleUpdate(;
    ψ0, env0 = wts0, H, dt, nstep = 2 * nstep,
    trscheme = trscheme_ψ, gate_bothsides = false
)
ψ, = time_evolve(alg)
env = converge_env(InfinitePEPS(ψ), 8)
result_2β′ = measure_mag(ψ, env; purified = true)
@info "Magnetization at T = $(1 / (2β)) (purification approach)" result_2β′
@test isapprox(abs.(result_2β′), bm_2β, rtol = 1.0e-2)

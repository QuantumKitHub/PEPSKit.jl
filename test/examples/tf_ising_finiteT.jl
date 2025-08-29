using Test
using Random
using LinearAlgebra
using TensorKit
using MPSKitModels
using PEPSKit
using PEPSKit: LocalOperator

Random.seed!(10235876)
σx = σˣ(Float64, Trivial)
σz = σᶻ(Float64, Trivial)

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

function measure_mag(pepo::InfinitePEPO, pf::InfinitePartitionFunction, env::CTMRGEnv)
    r, c = 1, 1
    @tensor M[w s; n e] := σx[p2; p1] * (pepo.A[r, c, 1])[p1 p2; n e s w]
    magx = expectation_value(pf, (r, c) => M, env)
    @tensor M[w s; n e] := σz[p2; p1] * (pepo.A[r, c, 1])[p1 p2; n e s w]
    magz = expectation_value(pf, (r, c) => M, env)
    return [magx, magz]
end

function measure_mag(peps::InfinitePEPS, env::CTMRGEnv)
    lattice = collect(space(t, 1) for t in peps.A)
    O = LocalOperator(lattice, ((1, 1),) => attach_ancilla(σx))
    magx = expectation_value(peps, O, env)
    O = LocalOperator(lattice, ((1, 1),) => attach_ancilla(σz))
    magz = expectation_value(peps, O, env)
    return [magx, magz]
end

Nr, Nc = 2, 2
pepo0 = trivial_InfinitePEPO(Float64, ℂ^2, (Nr, Nc, 1))
wts0 = SUWeight(pepo0)

trscheme_pepo = truncdim(8) & truncerr(1.0e-12)

ham = tfising_model(Float64, InfiniteSquare(Nr, Nc); J = 1.0, g = 2.0)
dt, maxiter = 1.0e-3, 400
β = dt * maxiter
alg = SimpleUpdate(dt, 0.0, maxiter, trscheme_pepo)

# when g = 2, β = 0.4 and 2β = 0.8 belong to two phases (without and with nonzero σᶻ)

# PEPO approach
## results at β, or T = 2.5
pepo, wts, = simpleupdate(pepo0, ham, alg, wts0; gate_side = :both)
pf = InfinitePartitionFunction(pepo)
env = converge_env(pf, 16)
result_β = measure_mag(pepo, pf, env)
@info "Magnetization at T = $(1 / β)" result_β
@test isapprox(abs.(result_β), bm_β, rtol = 1.0e-2)

## results at 2β, or T = 1.25
pepo, wts, = simpleupdate(pepo, ham, alg, wts; gate_side = :both)
pf = InfinitePartitionFunction(pepo)
env = converge_env(pf, 16)
result_2β = measure_mag(pepo, pf, env)
@info "Magnetization at T = $(1 / (2β))" result_2β
@test isapprox(abs.(result_2β), bm_2β, rtol = 1.0e-4)

# purification approach (should match 2β result)
pepo, = simpleupdate(pepo0, ham, alg, wts0; gate_side = :codomain)
peps = InfinitePEPS(pepo)
env = converge_env(peps, 8)
result_2β′ = measure_mag(peps, env)
@info "Magnetization at T = $(1 / (2β)) (purification approach)" result_2β′
@test isapprox(abs.(result_2β′), bm_2β, rtol = 1.0e-2)

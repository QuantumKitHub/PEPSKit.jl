using Test
using Random
using TensorKit
using PEPSKit

function converge_env(state, ctm_alg)
    env0 = CTMRGEnv(ones, scalartype(state), state, oneunit(spacetype(state)))
    env, = leading_boundary(env0, state, ctm_alg)
    return env
end

Random.seed!(1457860)
Nr, Nc = 2, 2
H = j1_j2_model(
    Float64, U1Irrep, InfiniteSquare(Nr, Nc);
    J1 = 1.0, J2 = 0.0, sublattice = false
)
Pspace = U1Space(1 // 2 => 1, -1 // 2 => 1)
Vspace = U1Space(0 => 2, 1 // 2 => 1, -1 // 2 => 1)
ψ0 = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc))
trunc = truncerror(; atol = 1.0e-10) & truncrank(4)
ctm_alg = SequentialCTMRG(; tol = 1.0e-10, verbosity = 2, trunc = truncerror(; atol = 1.0e-10) & truncrank(16))

# prepare simple update state
su_alg = SimpleUpdate(; trunc)
evolver = TimeEvolver(ψ0, H, 0.01, 5000, su_alg, SUWeight(ψ0))
ψ0, = time_evolve(evolver, H; tol = 1.0e-8, check_interval = 500)
env0 = converge_env(ψ0, ctm_alg)
e0 = expectation_value(ψ0, H, env0) / (Nr * Nc)
@info "Simple update energy = $(e0)."

# continue with NTU
@testset "NTU for ground state" begin
    ntu_alg = NeighbourUpdate(; bondenv_alg = NNpEnv())
    evolver = TimeEvolver(ψ0, H, 0.01, 100, ntu_alg)
    ψ, info = time_evolve(evolver, H, env0, ctm_alg)
    env = info.env
    e = expectation_value(ψ, H, env) / (Nr * Nc)
    @info "NTU energy = $(e)"
    @test e < e0
end

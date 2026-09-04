using TensorKit
using PEPSKit
using Test
using TensorKitTensors.SpinOperators: S_exchange

const CI = CartesianIndex

@testset "correlator_approx for physical state" begin
    Nr, Nc = 2, 2
    lattice = InfiniteSquare(Nr, Nc)
    sym = Trivial
    ham = j1_j2_model(Float64, sym, lattice; J1 = 1.0, J2 = 0.0, sublattice = false)
    op = S_exchange(Float64, sym)
    lattice = physicalspace(ham)

    ρ = PEPSKit.infinite_temperature_density_matrix(ham)
    state_trunc = truncrank(4) & truncerror(; atol = 1.0e-12)
    su_alg = SimpleUpdate(; trunc = state_trunc, purified = false)
    ρ, = time_evolve(ρ, ham, 1.0e-2, 50, su_alg, SUWeight(ρ))

    network = InfinitePartitionFunction(ρ)
    env = initialize_ctmrg_environment(network, ProductStateInitialization())
    env_trunc = truncrank(8) & truncerror(; atol = 1.0e-12)
    env, = leading_boundary(env, network; alg = :SequentialCTMRG, trunc = env_trunc)

    bonds = [
        (CI(1, 1), CI(1, 2)),
        (CI(1, 1), CI(2, 0)), (CI(1, 1), CI(2, 1)), (CI(1, 1), CI(2, 2)),
        (CI(1, 1), CI(3, 2)),
    ]
    cor_exact = map(bonds) do bond
        O = LocalOperator(lattice, bond => op)
        return expectation_value(ρ, O, env)
    end
    cor_trunc = correlator_approx(ρ, op, bonds, env; trunc = env_trunc, maxiter = 1)
    @info "Exact:" cor_exact
    @info "Approx:" cor_trunc
    @test cor_trunc ≈ cor_exact rtol = 1.0e-3
end

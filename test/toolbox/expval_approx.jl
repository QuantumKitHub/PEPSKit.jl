using TensorKit
using PEPSKit
using MPSKit
using Test
using Random

@testset "Approximate single-layer PEPO expectation values" begin
    Random.seed!(1234)

    d = ℂ^2
    D = ℂ^2
    χ = ℂ^3
    ρ = InfinitePEPO(d, D; unitcell = (2, 2, 1))
    env = CTMRGEnv(InfinitePartitionFunction(ρ), χ)
    trunc = notrunc()

    # U-shaped path, passing twice between two rows
    sites = CartesianIndex.([(2, 1), (1, 1), (1, 2), (2, 2)])
    V = d^length(sites)
    op = rand(ComplexF64, V, V)
    mpo = PEPSKit.gate_to_mpo(op; trunc)
    observable = MPOObservable(sites, mpo)
    exact = expectation_value(
        ρ, LocalOperator(physicalspace(ρ), sites => op), env
    )

    for direction in (:rows, :columns)
        @test expectation_value_approx(
            ρ, observable, env; trunc, maxiter = 0, direction
        ) ≈ exact
    end
end

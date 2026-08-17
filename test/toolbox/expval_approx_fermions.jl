using TensorKit
using PEPSKit
using MPSKit
using Test
using Random

const CI = CartesianIndex

@testset "Approximate fermionic single-layer PEPO expectation values" begin
    Random.seed!(1234)

    S = Vect[FermionParity]
    d = S(0 => 1, 1 => 1)
    D = S(0 => 1, 1 => 2)
    χ = S(0 => 2, 1 => 2)
    Nspaces = fill(D, 2, 2)
    Espaces = fill(D, 2, 2)
    Pspaces = fill(d, size(Nspaces))
    ρ = InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(randn, ComplexF64, InfinitePartitionFunction(ρ), χ)
    trunc = notrunc()

    op = randn(ComplexF64, d ⊗ d → d ⊗ d)
    mpo = PEPSKit.gate_to_mpo(op; trunc)
    geometries = (
        "horizontal" => [CI(1, 1), CI(1, 2)],
        "vertical" => [CI(1, 1), CI(2, 1)],
        "turned" => [CI(1, 1), CI(2, 2)],
        "reversed turned" => [CI(2, 2), CI(1, 1)],
    )

    for (geometry, sites) in geometries
        observable = MPOObservable(sites, mpo)
        exact = expectation_value(
            ρ, LocalOperator(physicalspace(ρ), sites => op), env
        )

        @testset "$geometry path" begin
            for direction in (:rows, :columns)
                @test expectation_value_approx(
                    ρ, observable, env; trunc, maxiter = 0, direction
                ) ≈ exact
            end
        end
    end
end

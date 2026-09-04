using TensorKit
using PEPSKit
using MPSKit
using Test
using Random

const CI = CartesianIndex

spaces = Dict(
    U1Irrep => (
        U1Space(1 => 2, -1 => 1),
        U1Space(1 => 1, 0 => 1, -1 => 2),
        U1Space(1 => 1, 0 => 1, -1 => 2),
    ),
    FermionParity => (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
)

sites_list = (
    [CI(1, 1), CI(1, 2)], # horizontal
    [CI(1, 1), CI(2, 1)], # vertical
    [CI(1, 1), CI(2, 2)], # turned
    [CI(2, 2), CI(1, 1)], # reversed turned
    [CI(2, 1), CI(1, 1), CI(1, 2), CI(2, 2)], # U-shaped
)

@testset "Single-layer PEPO ($S)" for S in keys(spaces)
    Random.seed!(1234)

    d, D, χ = spaces[S]
    ρ = InfinitePEPO(d, D; unitcell = (2, 2, 1))
    env = CTMRGEnv(InfinitePartitionFunction(ρ), χ)
    trunc = notrunc()

    for sites in sites_list
        n = length(sites)
        op = randn(ComplexF64, d^n → d^n)
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
end

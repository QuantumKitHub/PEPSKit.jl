using TensorKit
using PEPSKit
using Test
using TestExtras

ds = Dict(Trivial => ℂ^2, U1Irrep => U1Space(i => d for (i, d) in zip(-1:1, (1, 1, 2))), FermionParity => Vect[FermionParity](0 => 2, 1 => 1))
Ds = Dict(Trivial => ℂ^3, U1Irrep => U1Space(i => D for (i, D) in zip(-1:1, (1, 2, 2))), FermionParity => Vect[FermionParity](0 => 3, 1 => 2))
χs = Dict(Trivial => ℂ^4, U1Irrep => U1Space(i => χ for (i, χ) in zip(-2:2, (1, 3, 2))), FermionParity => Vect[FermionParity](0 => 3, 1 => 2))

@testset "Double-layer densitymatrix contractions ($I)" for I in keys(ds)
    d = ds[I]
    D = Ds[I]
    χ = χs[I]
    ρ = InfinitePEPO(d, D)

    ρ_peps = @constinferred InfinitePEPS(ρ)
    env = CTMRGEnv(ρ_peps, χ)

    O = rand(d, d)
    F = isomorphism(fuse(d ⊗ d'), d ⊗ d')
    @tensor O_doubled[-1; -2] := F[-1; 1 2] * O[1; 3] * twist(F', 2)[3 2; -2]

    # Single site
    O_singlesite = LocalOperator(reshape(physicalspace(ρ), 1, 1), ((1, 1),) => O)
    E1 = expectation_value(ρ, O_singlesite, ρ, env)
    O_doubled_singlesite = LocalOperator(physicalspace(ρ_peps), ((1, 1),) => O_doubled)
    E2 = expectation_value(ρ_peps, O_doubled_singlesite, ρ_peps, env)
    @test E1 ≈ E2

    # two sites
    for inds in zip(
            [(1, 1), (1, 1), (1, 1), (1, 2), (1, 1)],
            [(2, 1), (1, 2), (2, 2), (2, 1), (3, 1)]
        )
        O_twosite = LocalOperator(reshape(physicalspace(ρ), 1, 1), inds => O ⊗ O)
        E1 = expectation_value(ρ, O_twosite, ρ, env)
        O_doubled_twosite = LocalOperator(physicalspace(ρ_peps), inds => O_doubled ⊗ O_doubled)
        E2 = expectation_value(ρ_peps, O_doubled_twosite, ρ_peps, env)
        @test E1 ≈ E2
    end
end

using Test
using Random
using TensorKit
using PEPSKit

elt = Float64
ham = j1_j2_model(elt, U1Irrep, InfiniteSquare(2, 2); J1 = 1.0, J2 = 0.5, sublattice = false)
Vphy = physicalspace(ham)
Vns = [
    U1Space(0 => 1, 1 => 2, -1 => 1) U1Space(0 => 1, 1 => 1, -1 => 2)';
    U1Space(0 => 1, 1 => 2, -1 => 1)' U1Space(0 => 1, 1 => 1, -1 => 2)
]
Ves1 = [
    U1Space(0 => 1, 1 => 1, -1 => 2)' U1Space(1 / 2 => 2, -1 / 2 => 1, 3 / 2 => 1);
    U1Space(0 => 1, 1 => 1, -1 => 2) U1Space(1 / 2 => 1, -1 / 2 => 2, -3 / 2 => 1)'
]
Ves2 = [
    U1Space(0 => 1, 1 => 2, -1 => 1)' U1Space(0 => 1, 1 => 1, -1 => 2);
    U1Space(0 => 1, 1 => 2, -1 => 1) U1Space(0 => 1, 1 => 1, -1 => 2)'
]
Venv = U1Space(0 => 2, 1 => 1, -1 => 1)
states = [
    InfinitePEPS(randn, elt, Vphy, Vns, Ves1),
    InfinitePEPO(randn, elt, Vphy, Vns, Ves2),
]

@testset "Simple update on $(typeof(state0).name.wrapper)" for state0 in states
    # converted internally to SiteDependentTruncation
    alg = SimpleUpdate(; trunc = FixedSpaceTruncation())
    wts0 = SUWeight(state0)
    state, wts, = time_evolve(state0, ham, 0.1, 1, alg, wts0)
    for (t, t0) in zip(state.A, state0.A)
        @test space(t) == space(t0)
    end
end

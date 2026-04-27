using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: _is_bipartite, _get_fixedspacetrunc

elt = Float64
Nr, Nc = 2, 2
Vps = fill(U1Space(1 / 2 => 1, -1 / 2 => 1), (Nr, Nc))
Vns = [
    U1Space(0 => 1, 1 => 2, -1 => 1) U1Space(0 => 1, 1 => 2, -1 => 1)';
    U1Space(0 => 1, 1 => 2, -1 => 1)' U1Space(0 => 1, 1 => 2, -1 => 1)
]
Ves1 = [
    U1Space(1 / 2 => 1, -1 / 2 => 2, -3 / 2 => 1)' U1Space(0 => 1, 1 => 1, -1 => 2);
    U1Space(0 => 1, 1 => 1, -1 => 2) U1Space(1 / 2 => 1, -1 / 2 => 2, -3 / 2 => 1)'
]
Ves2 = [
    U1Space(0 => 1, 1 => 2, -1 => 1)' U1Space(0 => 1, 1 => 1, -1 => 2);
    U1Space(0 => 1, 1 => 1, -1 => 2) U1Space(0 => 1, 1 => 2, -1 => 1)'
]
Venv = U1Space(0 => 2, 1 => 1, -1 => 1)
states = (
    InfinitePEPS(randn, elt, Vps, Vns, Ves1),
    InfinitePEPO(randn, elt, Vps, Vns, Ves2),
)

@testset "Rotation of SiteDependentTruncation" begin
    state = states[1]
    for f in (rotl90, rotr90, rot180)
        trunc1 = f(_get_fixedspacetrunc(state))
        trunc2 = _get_fixedspacetrunc(f(state))
        @test all(
            t1.space == t2.space for (t1, t2) in zip(trunc1.truncs, trunc2.truncs)
        )
    end
end

@testset "Simple update on $(typeof(state0).name.wrapper), bipartite = $(bipartite)" for
    (state0, bipartite) in Iterators.product(states, (true, false))
    J2 = 0.5
    if bipartite
        state0[2, 1] = copy(state0[1, 2])
        state0[2, 2] = copy(state0[1, 1])
        J2 = 0.0
    end
    ham = j1_j2_model(elt, U1Irrep, InfiniteSquare(Nr, Nc); J1 = 1.0, J2, sublattice = false)
    # converted internally to SiteDependentTruncation
    alg = SimpleUpdate(; trunc = FixedSpaceTruncation(), bipartite)
    wts0 = SUWeight(state0)
    state, wts, = time_evolve(state0, ham, 0.1, 1, alg, wts0)
    for (t, t0) in zip(state.A, state0.A)
        @test space(t) == space(t0)
    end
    for (wt, wt0) in zip(wts.data, wts0.data)
        @test space(wt) == space(wt0)
    end
    if bipartite
        @test _is_bipartite(state)
        @test _is_bipartite(wts)
    end
end

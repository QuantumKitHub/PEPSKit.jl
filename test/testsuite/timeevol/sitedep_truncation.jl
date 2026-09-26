using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: _is_bipartite, _get_fixedspacetrunc, NORTH, EAST
using Adapt

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

function timeevol_sitedep_rotation(AT)
    Random.seed!(48736)
    return @testset "Rotation of SiteDependentTruncation ($AT)" begin
        state = adapt(AT, InfinitePEPS(randn, elt, Vps, Vns, Ves1))
        for f in (rotl90, rotr90, rot180)
            trunc1 = f(_get_fixedspacetrunc(state))
            trunc2 = _get_fixedspacetrunc(f(state))
            @test all(
                t1.space == t2.space for (t1, t2) in zip(trunc1.truncs, trunc2.truncs)
            )
        end
    end
end

function timeevol_sitedep_su(AT)
    Random.seed!(48736)
    states = (
        adapt(AT, InfinitePEPS(randn, elt, Vps, Vns, Ves1)),
        adapt(AT, InfinitePEPO(randn, elt, Vps, Vns, Ves2)),
    )
    return @testset "Simple update on $(typeof(state0).name.wrapper), bipartite = $(bipartite) ($AT)" for
        (state0, bipartite) in Iterators.product(states, (true, false))
        J2 = 0.5
        if bipartite
            state0[2, 1] = copy(state0[1, 2])
            state0[2, 2] = copy(state0[1, 1])
            J2 = 0.0
        end
        ham = adapt(AT, j1_j2_model(elt, U1Irrep, InfiniteSquare(Nr, Nc); J1 = 1.0, J2, sublattice = false))
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
end

"Check that neighbourhood updates preserve site-dependent virtual spaces."
function timeevol_sitedep_ntu(AT)
    Random.seed!(48736)
    states = (
        adapt(AT, InfinitePEPS(randn, elt, Vps, Vns, Ves1)),
        adapt(AT, InfinitePEPO(randn, elt, Vps, Vns, Ves2)),
    )
    @testset "NTU on $(typeof(state0).name.wrapper), bipartite = $(bipartite)" for
        (state0, bipartite) in Iterators.product(states, (false, true))
        J2 = 0.5
        if bipartite
            state0[2, 1] = copy(state0[1, 2])
            state0[2, 2] = copy(state0[1, 1])
            J2 = 0.0
        end
        ham = adapt(AT, j1_j2_model(elt, U1Irrep, InfiniteSquare(Nr, Nc); J1 = 1.0, J2, sublattice = false))
        # converted internally to SiteDependentTruncation
        opt_alg = ALSTruncation(; trunc = FixedSpaceTruncation())
        alg = NeighbourUpdate(; opt_alg, bondenv_alg = NNEnv(), bipartite)
        state, info = time_evolve(TimeEvolver(state0, ham, 0.1, 1, alg))
        for (t, t0) in zip(state.A, state0.A)
            @test space(t) == space(t0)
        end
        for (r, c) in Iterators.product(1:Nr, 1:Nc)
            @test space(info.wts[1, r, c], 1) == domain(state[r, c], EAST)
            @test space(info.wts[2, r, c], 1) == domain(state[r, c], NORTH)
        end
        if bipartite
            @test _is_bipartite(state)
            @test _is_bipartite(info.wts)
        end
    end
    return nothing
end

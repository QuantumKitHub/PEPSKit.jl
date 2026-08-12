using PEPSKit
using TensorKit
using Test
using Random

const CI = CartesianIndex

function _check_observable_bonds(observable::MPOObservable)
    @test length(observable.sites) == length(observable.mpo)
    path_positions = indexin(observable.sites, observable.path)
    @test all(!isnothing, path_positions)
    @test issorted(path_positions)
    @test first(path_positions) == 1
    @test last(path_positions) == length(observable.path)
    for k in 1:(length(observable.mpo) - 1)
        @test PEPSKit._mpo_right_stringspace(observable.mpo[k])' ==
            space(observable.mpo[k + 1], 1)
    end
    return nothing
end

Random.seed!(1234)
d = ℂ^2
lattice = fill(d, 2, 2)

@testset "Manhattan paths and directions" begin
    # first move horizontally
    @test PEPSKit._l_path(CI(2, 1), CI(2, 4)) ==
        CI.([(2, 1), (2, 2), (2, 3), (2, 4)])
    @test PEPSKit._l_path(CI(1, 3), CI(4, 3)) ==
        CI.([(1, 3), (2, 3), (3, 3), (4, 3)])
    @test PEPSKit._l_path(CI(3, 4), CI(1, 1)) ==
        CI.([(3, 4), (3, 3), (3, 2), (3, 1), (2, 1), (1, 1)])
    # nearest neighbor directions
    origin = CI(2, 2)
    @test PEPSKit._step_direction(origin, CI(1, 2)) === :north
    @test PEPSKit._step_direction(origin, CI(2, 3)) === :east
    @test PEPSKit._step_direction(origin, CI(3, 2)) === :south
    @test PEPSKit._step_direction(origin, CI(2, 1)) === :west
end

@testset "Routed MPO tensors" begin
    op2 = rand(ComplexF64, d^2, d^2)
    observable2 = MPOObservable([CI(1, 1), CI(3, 3)], op2, lattice)
    @test observable2.path ==
        CI.([(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)])
    @test observable2.sites == CI.([(1, 1), (3, 3)])
    _check_observable_bonds(observable2)

    original2 = PEPSKit.gate_to_mpo(op2)
    @test first(observable2.mpo) ≈ first(original2)
    @test last(observable2.mpo) ≈ last(original2)
    @test length(observable2.mpo) == 2

    sites4 = [CI(4, 2), CI(1, 3), CI(3, 4), CI(1, 1)]
    op4 = rand(ComplexF64, d^4, d^4)
    observable4 = MPOObservable(sites4, op4, lattice)
    @test observable4.path == CI.(
        [
            (1, 1), (1, 2), (2, 2), (3, 2), (4, 2), (4, 3),
            (3, 3), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4),
        ]
    )
    _check_observable_bonds(observable4)

    plaquette_sites = CI.([(2, 1), (1, 1), (1, 2), (2, 2)])
    plaquette_mpo = PEPSKit.gate_to_mpo(rand(ComplexF64, d^4, d^4))
    plaquette = MPOObservable(plaquette_sites, plaquette_sites, plaquette_mpo)
    @test plaquette.sites == plaquette_sites
    @test plaquette.path == plaquette_sites
    _check_observable_bonds(plaquette)
end

@testset "Explicit MPOs and validation" begin
    op = rand(ComplexF64, d^2, d^2)
    mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())

    observable = MPOObservable([(2, 2), (1, 1)], mpo)
    @test observable.sites == CI.([(2, 2), (1, 1)])
    @test observable.path == CI.([(2, 2), (2, 1), (1, 1)])
    @test observable.mpo !== mpo
    _check_observable_bonds(observable)

    # Invalid MPO with virtual space mismatch
    rank_one_mpo = PEPSKit.gate_to_mpo(op; trunc = truncrank(1))
    @test_throws SpaceMismatch MPOObservable(
        [CI(1, 1), CI(1, 2)], [first(mpo), last(rank_one_mpo)]
    )

    # the observable must act on two diffrent sites
    @test_throws ArgumentError MPOObservable([CI(1, 1), CI(1, 1)], mpo)
    @test_throws ArgumentError MPOObservable([CI(1, 1), CI(1, 1)], op, lattice)
    # Routing between consecutive tensors must not cross a later operator site.
    @test_throws ArgumentError MPOObservable(
        [CI(1, 1), CI(1, 3), CI(1, 2)],
        PEPSKit.gate_to_mpo(rand(ComplexF64, d^3, d^3)),
    )
end

using PEPSKit
using TensorKit
using Test
using Adapt
using Random

const directions = (:north, :east, :south, :west)

spaces = Dict(
    U1Irrep => (
        Rep[U₁](0 => 1, 1 => 1),
        Rep[U₁](0 => 1, 1 => 1, -1 => 1),
        Rep[U₁](1 => 1),
    ),
    FermionParity => (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](1 => 1),
    ),
)

"""
Check fermionic fuser flips and twists for each MPO routing direction.
"""
function toolbox_mpo_routing_fusers(AT)
    return @testset "Fuser flips and twists ($AT)" begin
        d = Vect[FermionParity](0 => 1, 1 => 1)
        D = Vect[FermionParity](0 => 1, 1 => 1)
        ρ = adapt(AT, InfinitePEPO(d, D; unitcell = (2, 2, 1)))
        A = ρ[1, 1, 1]
        Bh = ρ[1, 2, 1]
        Bv = ρ[2, 1, 1]
        A′ = PEPSKit.twistdual(A, 2)
        Bh′ = PEPSKit.twistdual(Bh, 2)
        Bv′ = PEPSKit.twistdual(Bv, 2)

        op = adapt(AT, randn(ComplexF64, d^2 → d^2))
        mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())

        first_tensor = PEPSKit.mpo_path_first(A, first(mpo), Val(:east))
        last_tensor = PEPSKit.mpo_path_last(Bh, last(mpo), Val(:west))
        @tensor exact[W1 S1 S2; N1 N2 E2] := op[po1 po2; pi1 pi2] *
            A′[pi1 po1; N1 x S1 W1] * Bh′[pi2 po2; N2 E2 S2 x]
        @tensor routed[W1 S1 S2; N1 N2 E2] :=
            first_tensor[W1 S1; N1 x] * last_tensor[x S2; N2 E2]
        @test routed ≈ exact

        first_tensor = PEPSKit.mpo_path_first(Bh, first(mpo), Val(:west))
        last_tensor = PEPSKit.mpo_path_last(A, last(mpo), Val(:east))
        @tensor exact[W2 S1 S2; N1 E1 N2] := op[po1 po2; pi1 pi2] *
            Bh′[pi1 po1; N1 E1 S1 x] * A′[pi2 po2; N2 x S2 W2]
        @tensor routed[W2 S1 S2; N1 E1 N2] :=
            first_tensor[x S1; N1 E1] * last_tensor[W2 S2; N2 x]
        @test routed ≈ exact

        first_tensor = PEPSKit.mpo_path_first(A, first(mpo), Val(:south))
        last_tensor = PEPSKit.mpo_path_last(Bv, last(mpo), Val(:north))
        @tensor exact[W1 W2 S2; N1 E1 E2] := op[po1 po2; pi1 pi2] *
            A′[pi1 po1; N1 E1 x W1] * Bv′[pi2 po2; x E2 S2 W2]
        @tensor routed[W1 W2 S2; N1 E1 E2] :=
            first_tensor[W1 x; N1 E1] * last_tensor[W2 S2; x E2]
        @test routed ≈ exact

        first_tensor = PEPSKit.mpo_path_first(Bv, first(mpo), Val(:north))
        last_tensor = PEPSKit.mpo_path_last(A, last(mpo), Val(:south))
        @tensor exact[W1 S1 W2; E1 N2 E2] := op[po1 po2; pi1 pi2] *
            Bv′[pi1 po1; x E1 S1 W1] * A′[pi2 po2; N2 E2 x W2]
        @tensor routed[W1 S1 W2; E1 N2 E2] :=
            first_tensor[W1 S1; x E1] * last_tensor[W2 x; N2 E2]
        @test routed ≈ exact
    end
end

"""
Check MPO endpoint shapes and string-routing identities for each symmetry.
"""
function toolbox_mpo_routing_identities(AT)
    return @testset "Routing identities ($S) ($AT)" for S in keys(spaces)
        Random.seed!(1234)
        d, D, stringspace = spaces[S]
        ρ = adapt(AT, InfinitePEPO(d, D; unitcell = (1, 1, 1)))
        op = adapt(AT, rand(ComplexF64, d^2 → d^2))
        mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())

        A = ρ[1, 1, 1]
        for direction in directions
            first_tensor = PEPSKit.mpo_path_first(A, first(mpo), Val(direction))
            last_tensor = PEPSKit.mpo_path_last(A, last(mpo), Val(direction))
            @test (numout(first_tensor), numin(first_tensor)) == (2, 2)
            @test (numout(last_tensor), numin(last_tensor)) == (2, 2)
        end

        middle = adapt(AT, TensorMap(TensorKit.BraidingTensor{ComplexF64}(d, stringspace)))
        for incoming in directions, outgoing in directions
            incoming == outgoing && continue
            tensor = PEPSKit.mpo_path_middle(A, middle, Val((incoming, outgoing)))
            string_tensor = PEPSKit.mpo_path_string(
                A, stringspace, Val((incoming, outgoing))
            )
            @test (numout(tensor), numin(tensor)) == (2, 2)
            @test string_tensor ≈ tensor
        end
    end
end

"""
Check routing choices, dense leg permutations, and periodic string insertion.
"""
function toolbox_mpo_routing_paths(AT)
    return @testset "MPO paths ($AT)" begin
        Random.seed!(1234)
        CI = CartesianIndex

        # The first horizontal L crosses a later operator site, so routing must try the vertical L.
        # Conversely, a collinear out-of-order MPO cannot be connected by either supported L path.
        fallback_sites = CI.([(1, 1), (3, 3), (1, 2)])
        @test PEPSKit._ordered_mpo_path(fallback_sites) ==
            CI.([(1, 1), (2, 1), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (1, 2)])
        @test_throws r"routing this MPO ordering is not implemented" PEPSKit._ordered_mpo_path(
            CI.([(1, 1), (1, 3), (1, 2)])
        )

        # Reconstructing on unequal physical spaces catches mismatches between the snake order and either group of permuted physical legs.
        lattice = [ℂ^2 ℂ^3; ℂ^1 ℂ^2]
        sites = CI.([(2, 2), (1, 1), (1, 2), (2, 1)])
        physical = foldl(⊗, lattice[sites])
        op = adapt(AT, randn(ComplexF64, physical ← physical))
        path, expanded = PEPSKit._route_mpo_term(sites, op, lattice)
        @test path == sites[[2, 4, 1, 3]]
        @tensor reconstructed[p1 p2 p3 p4; q1 q2 q3 q4] :=
            expanded[1][p1; q1 a] * expanded[2][a p2; q2 b] *
            expanded[3][b p3; q3 c] * expanded[4][c p4; q4]
        @test reconstructed ≈ permute(op, ((2, 4, 1, 3), (6, 8, 5, 7)))

        # Intermediate sites have different physical spaces from the endpoints and lie outside the unit cell.
        # Each inserted braid must therefore use periodic lookup and the MPO's storage type.
        @testset "Periodic strings ($S)" for S in keys(spaces)
            physical, _, _ = spaces[S]
            intermediate = physical ⊕ physical
            lattice = [physical intermediate; intermediate physical]
            op = adapt(AT, randn(ComplexF64, physical^2 ← physical^2))
            mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())
            path, expanded = PEPSKit._route_mpo_term(CI.([(0, 0), (2, 2)]), mpo, lattice)
            for k in 2:(length(path) - 1)
                site = path[k]
                V = lattice[mod1(site[1], 2), mod1(site[2], 2)]
                braid = adapt(AT, TensorMap(TensorKit.BraidingTensor{ComplexF64}(V, space(mpo[2], 1))))
                @test expanded[k] ≈ braid
            end
            @test all(t -> storagetype(t) == storagetype(mpo[1]), expanded)
        end
    end
end

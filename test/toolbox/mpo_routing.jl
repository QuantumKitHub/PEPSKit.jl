using PEPSKit
using TensorKit
using Test
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

@testset "Fuser flips and twists" begin
    d = Vect[FermionParity](0 => 1, 1 => 1)
    D = Vect[FermionParity](0 => 1, 1 => 1)
    ρ = InfinitePEPO(d, D; unitcell = (2, 2, 1))
    A = ρ[1, 1, 1]
    Bh = ρ[1, 2, 1]
    Bv = ρ[2, 1, 1]
    A′ = PEPSKit.twistdual(A, 2)
    Bh′ = PEPSKit.twistdual(Bh, 2)
    Bv′ = PEPSKit.twistdual(Bv, 2)

    op = randn(ComplexF64, d^2 → d^2)
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

@testset "Routing identities ($S)" for S in keys(spaces)
    Random.seed!(1234)
    d, D, stringspace = spaces[S]
    ρ = InfinitePEPO(d, D; unitcell = (1, 1, 1))
    op = rand(ComplexF64, d^2 → d^2)
    mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())

    A = ρ[1, 1, 1]
    for direction in directions
        first_tensor = PEPSKit.mpo_path_first(A, first(mpo), Val(direction))
        last_tensor = PEPSKit.mpo_path_last(A, last(mpo), Val(direction))
        @test (numout(first_tensor), numin(first_tensor)) == (2, 2)
        @test (numout(last_tensor), numin(last_tensor)) == (2, 2)
    end

    middle = TensorMap(TensorKit.BraidingTensor{ComplexF64}(d, stringspace))
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

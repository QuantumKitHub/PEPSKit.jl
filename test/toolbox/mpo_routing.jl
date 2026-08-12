using PEPSKit
using TensorKit
using Test
using Random

const directions = (:north, :east, :south, :west)

@testset "MPO path routing on PEPO" begin
    Random.seed!(1234)

    d = Rep[U₁](0 => 1, 1 => 1)
    D = Rep[U₁](0 => 1, 1 => 1, -1 => 1)
    ρ = InfinitePEPO(d, D; unitcell = (1, 1, 1))
    A = ρ[1, 1, 1]

    op = rand(ComplexF64, d^2 → d^2)
    mpo = PEPSKit.gate_to_mpo(op; trunc = notrunc())
    for direction in directions
        first_tensor = PEPSKit.mpo_path_first(A, first(mpo), Val(direction))
        last_tensor = PEPSKit.mpo_path_last(A, last(mpo), Val(direction))
        @test (numout(first_tensor), numin(first_tensor)) == (2, 2)
        @test (numout(last_tensor), numin(last_tensor)) == (2, 2)
    end

    stringspace = Rep[U₁](1 => 1)
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

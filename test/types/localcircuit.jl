using TensorKit
using PEPSKit
using PEPSKit: LocalCircuit
using Test

@testset "Rotation" begin
    op = LocalCircuit(
        [ℂ^1 ℂ^2 ℂ^3; ℂ^4 ℂ^5 ℂ^6],
        (
            ((1, 1), (1, 2)) => randn(ℂ^1, ℂ^1) ⊗ randn(ℂ^2, ℂ^2),
            ((2, 1), (1, 1)) => randn(ℂ^4, ℂ^4) ⊗ randn(ℂ^1, ℂ^1),
            ((1, 2), (2, 3)) => randn(ℂ^2, ℂ^2) ⊗ randn(ℂ^6, ℂ^6),
            ((1, 3), (2, 2)) => randn(ℂ^3, ℂ^3) ⊗ randn(ℂ^5, ℂ^5),
        )...
    )
    @test rot180(rot180(op)) == op
    @test rotl90(rotl90(op)) == rot180(op) == rotr90(rotr90(op))
    @test physicalspace(rotl90(op)) == rotl90(physicalspace(op))
    @test physicalspace(rotr90(op)) == rotr90(physicalspace(op))
    @test physicalspace(rot180(op)) == rot180(physicalspace(op))
end

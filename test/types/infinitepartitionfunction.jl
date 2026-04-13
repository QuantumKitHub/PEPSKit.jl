using Test
using TensorKit
using PEPSKit
using PEPSKit: PFTensor

T = ComplexF64
Ss = [Trivial, U1Irrep]
Vs = [ComplexSpace(3), U1Space(0 => 2, -1 => 1, 1 => 1)]

sizes = [(1, 1), (3, 3)]

@testset "$(sz) InfinitePartitionFunction with $(Ss[i]) symmetry" for (i, sz) in
    Iterators.product(eachindex(Ss), sizes)

    S = Ss[i]
    V = Vs[i]

    pf = InfinitePartitionFunction(randn, T, V; unitcell = sz)

    @test scalartype(pf) == T
    @test eltype(pf) <: PFTensor{typeof(V)}
    @test spacetype(pf) == typeof(V)

    @test (rotl90 ∘ rotl90)(pf) ≈ rot180(pf)
    @test (rotr90 ∘ rotr90 ∘ rotr90)(pf) ≈ rotl90(pf)
    @test length(pf) == prod(sz)
end

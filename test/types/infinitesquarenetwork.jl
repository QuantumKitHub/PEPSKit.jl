using Test
using TensorKit
using PEPSKit
using PEPSKit: PEPSTensor, PFTensor, PEPSSandwich, PEPOSandwich

T = ComplexF64
Ss = [Trivial, U1Irrep]
Ps = [ComplexSpace(2), U1Space(0 => 1, 1 => 1)]
Vs = [ComplexSpace(3), U1Space(0 => 2, -1 => 1, 1 => 1)]

sizes = [(1, 1), (3, 3)]

@testset "$(sz) InfiniteSquareNetwork with $(Ss[i]) symmetry" for (i, sz) in
    Iterators.product(eachindex(Ss), sizes)

    S = Ss[i]
    P = Ps[i]
    V = Vs[i]

    peps_tensor = PEPSTensor(randn, T, P, V)
    pf_tensor = PFTensor(randn, T, V)
    pepo_tensor = randn(T, P ⊗ P' ← V ⊗ V ⊗ V' ⊗ V')

    peps = InfinitePEPS(peps_tensor; unitcell = sz)
    pf = InfinitePartitionFunction(pf_tensor; unitcell = sz)
    pepo = InfinitePEPO(pepo_tensor; unitcell = (sz..., 2))

    peps_n = InfiniteSquareNetwork(peps)
    pf_n = InfiniteSquareNetwork(pf)
    pepo_n = InfiniteSquareNetwork(peps, pepo)

    @test scalartype(peps_n) == T
    @test eltype(peps_n) == PEPSSandwich{typeof(peps_tensor)}
    @test spacetype(peps_n) == typeof(P)
    @test sectortype(peps_n) == S

    @test scalartype(pf_n) == T
    @test eltype(pf_n) == typeof(pf_tensor)
    @test spacetype(pf_n) == typeof(V)
    @test sectortype(pf_n) == S

    @test scalartype(pepo_n) == T
    @test eltype(pepo_n) == PEPOSandwich{2, typeof(peps_tensor), typeof(pepo_tensor)}
    @test spacetype(pepo_n) == typeof(P)
    @test sectortype(pepo_n) == S

    @test peps_n + peps_n ≈ 2 * peps_n
    @test repeat(InfiniteSquareNetwork(InfinitePEPS(peps_tensor)), sz...) == peps_n
    @test (rotl90 ∘ rotl90)(peps_n) ≈ rot180(peps_n)
end

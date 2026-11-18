using Test, Random
using TensorKit
using PEPSKit

nondual(V) = isdual(V) ? flip(V) : V

@testset "SiteDependentTruncation of the environment bonds" begin
    Random.seed!(20261102)
    PA = U1Space(-1 => 1, 0 => 1)
    PB = U1Space(0 => 1, 1 => 1)
    Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
    Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)
    peps = InfinitePEPS(randn, ComplexF64, [PA PB; PB PA], [Vpeps Vpeps'; Vpeps' Vpeps], [Vpeps Vpeps'; Vpeps' Vpeps])
    env0 = CTMRGEnv(randn, ComplexF64, peps, Venv)

    # a different space on every bond
    smaller(V, k) = U1Space(c => max(dim(V, c) - (k % 3), 1) for c in sectors(V))
    targets = [smaller(nondual(space(env0.edges[I], 1)), sum(Tuple(I))) for I in CartesianIndices(env0.edges)]
    trunc = SiteDependentTruncation(map(truncspace, targets))
    @testset "$projector_alg" for projector_alg in (:HalfInfiniteProjector, :FullInfiniteProjector)
        alg = SimultaneousCTMRG(; projector_alg, trunc)
        env, = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(peps), env0, alg)
        @test map(e -> nondual(space(e, 1)), env.edges) == targets
    end

    # no environment coordinates in SequentialCTMRG, no environment bonds in a PEPS-bond truncation
    strategy(alg, coordinate...) = PEPSKit.truncation_strategy(alg.projector_alg, env0.edges[1], coordinate...)
    @test_throws ArgumentError strategy(SequentialCTMRG(; trunc))
    @test_throws ArgumentError strategy(SimultaneousCTMRG(; trunc = SiteDependentTruncation(fill(truncrank(4), 2, 2, 2))), (1, 1, 1))
end

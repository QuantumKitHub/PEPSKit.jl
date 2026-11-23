using Test, Random
using TensorKit
using PEPSKit

nondual(V) = isdual(V) ? flip(V) : V
edgespaces(env) = map(e -> nondual(space(e, 1)), env.edges)

# a different space on every bond
smaller(V::ComplexSpace, k) = ℂ^(dim(V) - (k % 3))
smaller(V, k) = U1Space(c => max(dim(V, c) - (k % 3), 1) for c in sectors(V))

Random.seed!(20261102)
PA = U1Space(-1 => 1, 0 => 1)
PB = U1Space(0 => 1, 1 => 1)
Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)
peps_u1 = InfinitePEPS(randn, ComplexF64, [PA PB; PB PA], [Vpeps Vpeps'; Vpeps' Vpeps], [Vpeps Vpeps'; Vpeps' Vpeps])
peps_rect = InfinitePEPS(randn, ComplexF64, fill(ℂ^2, 2, 3), [ℂ^2 ℂ^3 ℂ^2; ℂ^3 ℂ^2 ℂ^3], [ℂ^3 ℂ^2 ℂ^2; ℂ^2 ℂ^3 ℂ^3])

@testset "SiteDependentTruncation of the environment bonds, $name" for (name, peps, χenv) in
    (("U(1), 2 × 2", peps_u1, Venv), ("trivial, 2 × 3", peps_rect, ℂ^8))
    env0 = CTMRGEnv(randn, ComplexF64, peps, χenv)
    targets = [smaller(V, sum(Tuple(I))) for (I, V) in pairs(edgespaces(env0))]
    trunc = SiteDependentTruncation(map(truncspace, targets))

    @testset "$ctmrg_alg, $projector_alg" for ctmrg_alg in (SimultaneousCTMRG, SequentialCTMRG),
            projector_alg in (:HalfInfiniteProjector, :FullInfiniteProjector)
        alg = ctmrg_alg(; projector_alg, trunc)
        env, = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(peps), env0, alg)
        @test edgespaces(env) == targets
    end

    # environment bond truncations rotate along with the environment
    @testset "$rot" for rot in (rotl90, rotr90, rot180)
        @test map(t -> t.space, rot(trunc).truncs) == edgespaces(rot(env0))
    end

    # FixedSpaceTruncation is a truncation to the current spaces
    projector_alg = PEPSKit.fix_spaces(SimultaneousCTMRG().projector_alg, env0)
    @test map(t -> t.space, projector_alg.trunc.truncs) == edgespaces(env0)
end

@testset "Unsupported truncations" begin
    # FixedSpaceTruncation is replaced by the spaces of the initial environment in `leading_boundary`
    @test_throws ArgumentError PEPSKit.truncation_strategy(SimultaneousCTMRG().projector_alg, (1, 1, 1))
    # no environment coordinates in C4vCTMRG, no environment bonds in a PEPS-bond truncation
    trunc = SiteDependentTruncation(fill(truncrank(4), 4, 1, 1))
    alg = C4vCTMRG(; projector_alg = :C4vEighProjector, trunc)
    @test_throws ArgumentError PEPSKit.truncation_strategy(alg.projector_alg)
    trunc = SiteDependentTruncation(fill(truncrank(4), 2, 2, 2))
    alg = SimultaneousCTMRG(; trunc)
    @test_throws ArgumentError PEPSKit.truncation_strategy(alg.projector_alg, (1, 1, 1))
end

using Test
using PEPSKit
using PEPSKit: herm_depth, herm_width
using TensorKit

function _test_elementwise_equal(peps1::InfinitePEPS, peps2::InfinitePEPS)
    return @test all(zip(peps1.A, peps2.A)) do (p1, p2)
        p1.data == p2.data
    end
end

@testset "RotateReflect" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_rotatereflect = symmetrize!(deepcopy(peps), RotateReflect())
    _test_elementwise_equal(peps_full, rotl90(peps_full))
    _test_elementwise_equal(peps_full, rot180(peps_full))
    _test_elementwise_equal(peps_full, rotr90(peps_full))
    _test_elementwise_equal(
        peps_full, InfinitePEPS(reverse(map(copy ∘ herm_depth, peps_full.A); dims=1))
    )
    _test_elementwise_equal(
        peps_full, InfinitePEPS(reverse(map(copy ∘ herm_width, peps_full.A); dims=2))
    )
end

@testset "ReflectDepth" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_depth = symmetrize!(deepcopy(peps), ReflectDepth())
    _test_elementwise_equal(
        peps_depth, InfinitePEPS(reverse(map(copy ∘ herm_depth, peps_depth.A); dims=1))
    )
end

@testset "ReflectWidth" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_width = symmetrize!(deepcopy(peps), ReflectWidth())
    _test_elementwise_equal(
        peps_width, InfinitePEPS(reverse(map(copy ∘ herm_width, peps_width.A); dims=2))
    )
end

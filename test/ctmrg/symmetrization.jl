using Test
using PEPSKit
using PEPSKit: herm_depth, herm_width, _fit_spaces
using TensorKit

function PEPSKit._fit_spaces(data_peps::InfinitePEPS, space_peps::InfinitePEPS)
    fitted_tensors = map(zip(data_peps.A, space_peps.A)) do (pd, ps)
        PEPSKit._fit_spaces(pd, ps)
    end
    return InfinitePEPS(fitted_tensors)
end

@testset "RotateReflect" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_full = symmetrize!(deepcopy(peps), RotateReflect())
    @test peps_full ≈ _fit_spaces(rotl90(peps_full), peps_full)
    @test peps_full ≈ _fit_spaces(rot180(peps_full), peps_full)
    @test peps_full ≈ _fit_spaces(rotr90(peps_full), peps_full)

    peps_reflect_depth = _fit_spaces(
        InfinitePEPS(reverse(map(herm_depth, peps_full.A); dims=1)), peps_full
    )
    @test peps_full ≈ peps_reflect_depth

    peps_reflect_width = _fit_spaces(
        InfinitePEPS(reverse(map(herm_width, peps_full.A); dims=2)), peps_full
    )
    @test peps_full ≈ peps_reflect_width
end

@testset "ReflectDepth" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_depth = symmetrize!(deepcopy(peps), ReflectDepth())
    peps_reflect = _fit_spaces(
        InfinitePEPS(reverse(map(herm_depth, peps_depth.A); dims=1)), peps_depth
    )
    @test peps_depth ≈ peps_reflect
end

@testset "ReflectWidth" for unitcell in [(1, 1), (2, 2), (3, 3)]
    peps = InfinitePEPS(2, 2; unitcell)

    peps_width = symmetrize!(deepcopy(peps), ReflectWidth())
    peps_reflect = _fit_spaces(
        InfinitePEPS(reverse(map(herm_width, peps_width.A); dims=2)), peps_width
    )
    @test peps_width ≈ peps_reflect
end

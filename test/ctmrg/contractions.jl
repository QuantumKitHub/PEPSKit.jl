using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: eachcoordinate
using PEPSKit: EnlargedCorner, simultaneous_projectors
using PEPSKit: renormalize_northwest_corner, renormalize_northeast_corner, renormalize_southeast_corner, renormalize_southwest_corner

# settings
Random.seed!(91283219347)
stype = ComplexF64
ctm_alg = SimultaneousCTMRG(; projector_alg = :halfinfinite)

function test_peps_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(randn, stype, peps, chis_north, chis_east, chis_south, chis_west)

    n = InfiniteSquareNetwork(peps)

    return test_contractions(n, env)
end

function test_pf_contractions(
        Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    pf = InfinitePartitionFunction(randn, stype, Nspaces, Espaces)
    env = CTMRGEnv(randn, stype, pf, chis_north, chis_east, chis_south, chis_west)
    n = InfiniteSquareNetwork(pf)

    return test_contractions(n, env)
end

function test_pepo_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    pepo = InfinitePEPO(randn, stype, Pspaces, Pspaces, Pspaces)
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
    n = InfiniteSquareNetwork(peps, pepo)
    env = CTMRGEnv(randn, stype, n, chis_north, chis_east, chis_south, chis_west)

    return test_contractions(n, env)
end

function test_contractions(n::InfiniteSquareNetwork, env::CTMRGEnv)
    coordinates = eachcoordinate(n)
    dirs_and_coordinates = eachcoordinate(n, 1:4)

    # initialize dense and sparse enlarged corners
    sparse_enlarged_corners = map(dirs_and_coordinates) do co
        return EnlargedCorner(n, env, co)
    end
    dense_enlarged_corners = map(TensorMap, sparse_enlarged_corners)

    # compute projectors (doesn't matter how)
    (P_left, P_right), info = simultaneous_projectors(
        dense_enlarged_corners, env, ctm_alg.projector_alg
    )

    # test corner renormalization
    return foreach(coordinates) do (r, c)
        for renormalize_f in (
                renormalize_northwest_corner, renormalize_northeast_corner,
                renormalize_southeast_corner, renormalize_southwest_corner,
            )
            C_sparse = renormalize_f((r, c), sparse_enlarged_corners, P_left, P_right)
            C_dense = renormalize_f((r, c), dense_enlarged_corners, P_left, P_right)
            @test C_sparse ≈ C_dense
        end
    end

    # TODO: test all other uncovered contracitions
end

@testset "Random Cartesian spaces" begin
    unitcell = (3, 3)

    Pspaces = ComplexSpace.(rand(2:3, unitcell...))
    Nspaces = ComplexSpace.(rand(2:4, unitcell...))
    Espaces = ComplexSpace.(rand(2:4, unitcell...))
    chis_north = ComplexSpace.(rand(5:10, unitcell...))
    chis_east = ComplexSpace.(rand(5:10, unitcell...))
    chis_south = ComplexSpace.(rand(5:10, unitcell...))
    chis_west = ComplexSpace.(rand(5:10, unitcell...))

    test_peps_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    test_pf_contractions(
        Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    test_pepo_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
end

@testset "Specific U1 spaces" begin
    unitcell = (2, 2)

    PA = U1Space(-1 => 1, 0 => 1)
    PB = U1Space(0 => 1, 1 => 1)
    Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
    Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)

    Pspaces = [PA PB; PB PA]
    Nspaces = [Vpeps Vpeps'; Vpeps' Vpeps]
    chis = [Venv Venv; Venv Venv]

    test_peps_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
    test_pf_contractions(Nspaces, Nspaces, chis, chis, chis, chis)
    test_pepo_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)

    # 4x4 unit cell with all 32 inequivalent bonds
    #
    #    10     4     7    32
    #     |     |     |     |
    #  3--A--1--B--5--C--8--D--3
    #     |     |     |     |
    #     2     6     9    11
    #     |     |     |     |
    # 14--E-12--F-15--G-17--H-14
    #     |     |     |     |
    #    13    16    18    19
    #     |     |     |     |
    # 22--I-20--J-23--K-25--L-22
    #     |     |     |     |
    #    21    24    26    27
    #     |     |     |     |
    # 29--M-28--N-30--O-31--P-29
    #     |     |     |     |
    #    10     4     7    32

    phys_space = Vect[U1Irrep](1 => 1, -1 => 1)
    corner_space = Vect[U1Irrep](0 => 1, 1 => 1, -1 => 1)
    vspaces = map(i -> Vect[U1Irrep](0 => 1 + i % 4, 1 => i ÷ 4 % 4, -2 => i ÷ 16), 1:32)
    @test length(Set(vspaces)) == 32

    Espaces = [
        vspaces[1] vspaces[5] vspaces[8] vspaces[3]
        vspaces[12] vspaces[15] vspaces[17] vspaces[14]
        vspaces[20] vspaces[23] vspaces[25] vspaces[22]
        vspaces[28] vspaces[30] vspaces[31] vspaces[29]
    ]

    Nspaces = [
        vspaces[10] vspaces[4] vspaces[7] vspaces[32]
        vspaces[2] vspaces[6] vspaces[9] vspaces[11]
        vspaces[13] vspaces[16] vspaces[18] vspaces[19]
        vspaces[21] vspaces[24] vspaces[26] vspaces[27]
    ]
    Pspaces = fill(phys_space, (4, 4))
    chis = fill(corner_space, (4, 4))

    test_peps_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
    test_pf_contractions(Nspaces, Nspaces, chis, chis, chis, chis)
    test_pepo_contractions(Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
end

@testset "Random fermionic spaces" begin
    unitcell = (3, 3)

    S = Vect[FermionParity]
    pdims = rand(2:3, unitcell..., 2)
    vdims = rand(2:4, unitcell..., 2)
    edims = rand(5:10, unitcell..., 2)

    function _construct_space(ds::Array{<:Int, 3})
        V = map(Iterators.product(axes(ds)[1:2]...)) do (r, c)
            return S(0 => ds[r, c, 1], 1 => ds[r, c, 2])
        end
        return V
    end

    Pspaces = _construct_space(pdims)
    Nspaces = _construct_space(vdims)
    Espaces = _construct_space(vdims)
    chis_north = _construct_space(edims)
    chis_east = _construct_space(edims)
    chis_south = _construct_space(edims)
    chis_west = _construct_space(edims)

    test_peps_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    test_pf_contractions(
        Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
    test_pepo_contractions(
        Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west,
    )
end

using Test
using Random
using PEPSKit
using PEPSKit: _prev, _next, ctmrg_iteration
using TensorKit

# settings
Random.seed!(91283219347)
stype = ComplexF64
ctm_algs = [
    SequentialCTMRG(; projector_alg=:halfinfinite),
    SequentialCTMRG(; projector_alg=:fullinfinite),
    SimultaneousCTMRG(; projector_alg=:halfinfinite),
    SimultaneousCTMRG(; projector_alg=:fullinfinite),
]

function test_unitcell(
    ctm_alg,
    unitcell,
    Pspaces,
    Nspaces,
    Espaces,
    chis_north,
    chis_east,
    chis_south,
    chis_west,
)
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(randn, stype, peps, chis_north, chis_east, chis_south, chis_west)

    # apply one CTMRG iteration with fixeds
    env′, = ctmrg_iteration(InfiniteSquareNetwork(peps), env, ctm_alg)

    # compute random expecation value to test matching bonds
    random_op = LocalOperator(
        PEPSKit._to_space.(Pspaces),
        [
            (c,) => randn(
                scalartype(peps),
                PEPSKit._to_space(Pspaces[c]),
                PEPSKit._to_space(Pspaces[c]),
            ) for c in CartesianIndices(unitcell)
        ]...,
    )
    @test expectation_value(peps, random_op, env) isa Number
    @test expectation_value(peps, random_op, env′) isa Number
end

function random_dualize!(M::AbstractMatrix{<:ElementarySpace})
    mask = rand([true, false], size(M))
    M[mask] .= adjoint.(M[mask])
    return M
end

@testset "Integer space specifiers with $ctm_alg" for ctm_alg in ctm_algs
    unitcell = (3, 3)

    Pspaces = rand(2:3, unitcell...)
    Nspaces = rand(2:4, unitcell...)
    Espaces = rand(2:4, unitcell...)
    chis_north = rand(5:10, unitcell...)
    chis_east = rand(5:10, unitcell...)
    chis_south = rand(5:10, unitcell...)
    chis_west = rand(5:10, unitcell...)

    test_unitcell(
        ctm_alg,
        unitcell,
        Pspaces,
        Nspaces,
        Espaces,
        chis_north,
        chis_east,
        chis_south,
        chis_west,
    )
end

@testset "Random Cartesian spaces with $ctm_alg" for ctm_alg in ctm_algs
    unitcell = (3, 3)

    Pspaces = random_dualize!(ComplexSpace.(rand(2:3, unitcell...)))
    Nspaces = random_dualize!(ComplexSpace.(rand(2:4, unitcell...)))
    Espaces = random_dualize!(ComplexSpace.(rand(2:4, unitcell...)))
    chis_north = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_east = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_south = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_west = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))

    test_unitcell(
        ctm_alg,
        unitcell,
        Pspaces,
        Nspaces,
        Espaces,
        chis_north,
        chis_east,
        chis_south,
        chis_west,
    )
end

@testset "Specific U1 spaces with $ctm_alg" for ctm_alg in ctm_algs
    unitcell = (2, 2)

    PA = U1Space(-1 => 1, 0 => 1)
    PB = U1Space(0 => 1, 1 => 1)
    Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
    Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)

    Pspaces = [PA PB; PB PA]
    Nspaces = [Vpeps Vpeps'; Vpeps' Vpeps]
    chis = [Venv Venv; Venv Venv]

    test_unitcell(ctm_alg, unitcell, Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)

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

    test_unitcell(ctm_alg, unitcell, Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
end

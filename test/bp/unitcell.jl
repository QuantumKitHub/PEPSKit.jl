using Test
using Random
using PEPSKit
using PEPSKit: _prev, _next, BPEnv, bp_iteration, gauge_fix, _fix_svd_algorithm, BeliefPropagation, bp_iteration
using TensorKit

# settings
Random.seed!(91283219347)
stype = ComplexF64

function test_unitcell(alg, unitcell, Pspaces, Nspaces, Espaces,)
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
    env  = BPEnv(randn, stype, peps)

    # apply one BP iteration with fixeds
    env′ = bp_iteration(InfiniteSquareNetwork(peps), env, alg)
    env″ = bp_iteration(InfiniteSquareNetwork(peps), env′, alg) # another iteration to fix spaces

    # compute random expecation value to test matching bonds
    random_op = LocalOperator(
        Pspaces,
        [
            (c,) => randn(
                    scalartype(peps),
                    Pspaces[c], Pspaces[c],
                ) for c in CartesianIndices(unitcell)
        ]...,
    )
    @test expectation_value(peps, random_op, env) isa Number
    @test expectation_value(peps, random_op, env′) isa Number

    # test if gauge fixing routines run through
    _, signs = gauge_fix(env′, env″)
    @test signs isa Array
    return
end

@testset "Random Cartesian spaces with BP" begin
    unitcell = (3, 3)

    Pspaces = ComplexSpace.(rand(2:3, unitcell...))
    Nspaces = ComplexSpace.(rand(2:4, unitcell...))
    Espaces = ComplexSpace.(rand(2:4, unitcell...))

    alg = BeliefPropagation()
    test_unitcell(alg, unitcell, Pspaces, Nspaces, Espaces)
end

@testset "Specific U1 spaces with BP" begin
    unitcell = (2, 2)

    PA = U1Space(-1 => 1, 0 => 1)
    PB = U1Space(0 => 1, 1 => 1)
    Vpeps = U1Space(-1 => 2, 0 => 1, 1 => 2)
    Venv = U1Space(-2 => 2, -1 => 3, 0 => 4, 1 => 3, 2 => 2)

    Pspaces = [PA PB; PB PA]
    Nspaces = [Vpeps Vpeps'; Vpeps' Vpeps]

    alg = BeliefPropagation()
    test_unitcell(alg, unitcell, Pspaces, Nspaces, Nspaces)

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

    test_unitcell(alg, unitcell, Pspaces, Nspaces, Nspaces)
end

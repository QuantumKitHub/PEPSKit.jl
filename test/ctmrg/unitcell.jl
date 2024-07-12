using Test
using Random
using PEPSKit
using PEPSKit: _prev, _next, ctmrg_iter
using TensorKit

# settings
Random.seed!(91283219347)
stype = ComplexF64
ctm_alg = CTMRG()

function test_unitcell(
    unitcell, Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west
)
    peps = InfinitePEPS(randn, stype, Pspaces, Nspaces, Espaces)
    env = CTMRGEnv(randn, stype, peps, chis_north, chis_east, chis_south, chis_west)

    # apply one CTMRG iteration with fixeds
    env′, = ctmrg_iter(peps, env, ctm_alg)

    # compute random expecation value to test matching bonds
    random_op = LocalOperator(
        PEPSKit._to_space.(Pspaces),
        [
            (c,) => TensorMap(
                randn,
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

@testset "Integer space specifiers" begin
    unitcell = (3, 3)

    Pspaces = rand(2:3, unitcell...)
    Nspaces = rand(2:4, unitcell...)
    Espaces = rand(2:4, unitcell...)
    chis_north = rand(5:10, unitcell...)
    chis_east = rand(5:10, unitcell...)
    chis_south = rand(5:10, unitcell...)
    chis_west = rand(5:10, unitcell...)

    test_unitcell(
        unitcell, Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west
    )
end

@testset "Random Cartesian spaces" begin
    unitcell = (3, 3)

    Pspaces = random_dualize!(ComplexSpace.(rand(2:3, unitcell...)))
    Nspaces = random_dualize!(ComplexSpace.(rand(2:4, unitcell...)))
    Espaces = random_dualize!(ComplexSpace.(rand(2:4, unitcell...)))
    chis_north = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_east = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_south = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))
    chis_west = random_dualize!(ComplexSpace.(rand(5:10, unitcell...)))

    test_unitcell(
        unitcell, Pspaces, Nspaces, Espaces, chis_north, chis_east, chis_south, chis_west
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

    test_unitcell(unitcell, Pspaces, Nspaces, Nspaces, chis, chis, chis, chis)
end

using TensorKit
using PEPSKit
using Test

@testset "Standardize virtual-space dualness" begin
    d = U1Space(0 => 1)
    D = U1Space(0 => 1)
    χ = U1Space(0 => 1)
    Pspaces = fill(d, 2, 2, 1)

    patterns = (
        (fill(D', 2, 2, 1), fill(D', 2, 2, 1)),
        (
            reshape([D, D', D', D], 2, 2, 1),
            reshape([D', D, D, D'], 2, 2, 1),
        ),
    )
    for (Nspaces, Espaces) in patterns
        ρ = InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
        env = CTMRGEnv(randn, ComplexF64, InfinitePartitionFunction(ρ), χ)
        standardized_ρ, standardized_env = PEPSKit.standardize_dualness(ρ, env)

        for row in axes(ρ, 1), col in axes(ρ, 2)
            A = standardized_ρ[row, col, 1]
            edge_coordinates = (
                PEPSKit.NORTH => (row - 1, col),
                PEPSKit.EAST => (row, col + 1),
                PEPSKit.SOUTH => (row + 1, col),
                PEPSKit.WEST => (row, col - 1),
            )
            for (direction, coordinates) in edge_coordinates
                V = PEPSKit.virtualspace(A, direction)
                E = PEPSKit.edge(standardized_env, direction, coordinates...)
                @test isdual(V) == (direction in (PEPSKit.NORTH, PEPSKit.EAST))
                @test V == space(E, 2)'
            end
        end
    end
end

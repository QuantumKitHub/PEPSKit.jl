using TensorKit
using PEPSKit
using MPSKit
using Test
using Random

const CI = CartesianIndex

"""
Contract `⟨op⟩` on each bond in `bonds` independently without caching
"""
function _shared_window_reference(
        op::AbstractTensorMap, bonds::Vector{NTuple{2, CI{2}}}, ρ, env
    )
    lattice = physicalspace(ρ)
    observables = [MPOObservable(collect(pair), op, lattice) for pair in bonds]
    rowrange, colrange = PEPSKit._window_ranges(bonds)
    alg = PEPSKit.WindowApprox(Zipup(; trunc = notrunc()), nothing)
    norm = PEPSKit._contract_window_rows(
        ρ, nothing, env, rowrange, colrange, alg
    )
    return map(observables) do observable
        numerator = PEPSKit._contract_window_rows(
            ρ, observable, env, rowrange, colrange, alg
        )
        return numerator / norm
    end
end

bonds = [
    # source at (1, 1), bond sites in order
    ## target in the same row as source
    (CI(1, 1), CI(1, 2)),
    ## target in a later row, on both sides of or just below the source
    (CI(1, 1), CI(2, 0)), (CI(1, 1), CI(2, 1)), (CI(1, 1), CI(2, 2)),
    # source still at (1, 1), but bond sites need swapping
    (CI(2, 2), CI(1, 1)),
    # another source at (1, 2), bond sites in order
    (CI(1, 2), CI(2, 0)), (CI(1, 2), CI(2, 2)),
]

@testset "Two-site source grouping" begin
    groups = PEPSKit._twosite_source_groups(bonds)
    @test length(groups) == 3 &&
        groups[(CI(1, 1), false)][CI(2, 2)] == 4 &&
        haskey(groups, (CI(1, 1), true))
end

@testset "Single-layer PEPO" begin
    Random.seed!(1234)

    d = ℂ^2
    D = ℂ^3
    χ = ℂ^4
    ρ = InfinitePEPO(d, D; unitcell = (2, 2, 1))
    lattice = physicalspace(ρ)
    env = CTMRGEnv(InfinitePartitionFunction(ρ), χ)
    trunc = notrunc()

    O² = rand(ComplexF64, d^2, d^2)
    id² = isomorphism(d, d) ⊗ isomorphism(d, d)

    # bonds to be measured should be unique
    @test_throws ArgumentError correlator_approx(ρ, O², fill(bonds[3], 2), env)

    exact_same_row = expectation_value(ρ, LocalOperator(lattice, bonds[1] => O²), env)
    @test correlator_approx(
        ρ, O², bonds[1], env; trunc, maxiter = 0, direction = :rows
    ) ≈ exact_same_row

    exact_reversed = expectation_value(ρ, LocalOperator(lattice, bonds[5] => O²), env)
    @test correlator_approx(
        ρ, O², bonds[5], env; trunc, maxiter = 0, direction = :columns
    ) ≈ exact_reversed

    vals_ref = _shared_window_reference(O², bonds, ρ, env)
    vals_rows = correlator_approx(
        ρ, O², bonds, env; trunc, maxiter = 0, direction = :rows
    )
    @test vals_rows ≈ vals_ref
    vals_columns = correlator_approx(
        ρ, O², bonds, env; trunc, maxiter = 0, direction = :columns
    )
    @test vals_columns ≈ vals_rows
    vals_auto = correlator_approx(
        ρ, O², bonds, env; trunc, maxiter = 0, direction = :auto
    )
    @test vals_auto ≈ vals_columns

    @test correlator_approx(ρ, id², bonds, env; trunc, direction = :rows) ≈
        ones(length(bonds))

    alg = PEPSKit.WindowApprox(Zipup(; trunc), nothing)
    cache = PEPSKit._window_row_cache(ρ, env, 1:2, 1:2, alg)
    @test all(eachindex(cache.north_prefixes)) do k
        dot(cache.south_suffixes[k], cache.north_prefixes[k]) ≈ cache.norm
    end
end

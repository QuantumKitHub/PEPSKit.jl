using TensorKit
using PEPSKit
using MPSKit
using Test
using Random

const CI = CartesianIndex

"""
Contract `⟨op⟩` between `i` and each site in `js` independently without caching, using one shared window.
"""
function _shared_window_reference(op::AbstractTensorMap, i::CI{2}, js, ρ, env)
    lattice = physicalspace(ρ)
    observables = [MPOObservable([i, j], op, lattice) for j in js]
    rowrange, colrange = PEPSKit._window_ranges([i; js])
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

# Exercise both U(1) symmetry and fermionic signs with small physical, virtual, and boundary spaces.
spaces = Dict(
    U1Irrep => (
        U1Space(1 => 2, -1 => 1),
        U1Space(0 => 1, 1 => 1),
        U1Space(0 => 1, 1 => 1),
    ),
    FermionParity => (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 1),
    )
)

@testset "Single-layer PEPO ($S)" for S in keys(spaces)
    # Use a reproducible random PEPO on a rectangular unit cell and disable boundary truncation.
    Random.seed!(1234)
    d, D, χ = spaces[S]
    ρ = InfinitePEPO(d, D; unitcell = (2, 3, 1))
    env = CTMRGEnv(InfinitePartitionFunction(ρ), χ)
    trunc = notrunc()
    i = CI(-1, 0)
    # Unsorted targets include both same-row neighbors of i and targets to the east, west, and directly south on row 1.
    # Row 0 has no measurement targets but is still contracted, testing MPO-string propagation through a row without targets.
    js = [CI(1, 1), CI(-1, -1), CI(1, 0), CI(-1, 1), CI(1, -1)]

    # The normalized expectation of the identity must be one at every target.
    id² = isomorphism(d, d) ⊗ isomorphism(d, d)
    @test correlator_approx(ρ, id², i, js, env; trunc, maxiter = 0) ≈ ones(length(js))

    # This target distribution only permits row sweeps.
    # Compare against independent contractions in target order.
    O² = rand(ComplexF64, d^2, d^2)
    vals_ref = _shared_window_reference(O², i, js, ρ, env)
    vals_rows = correlator_approx(ρ, O², i, js, env; trunc, maxiter = 0)
    @test vals_rows ≈ vals_ref

    # CartesianIndices are flattened exactly as in correlator.
    grid = CartesianIndices((1:1, -1:1))
    @test correlator_approx(ρ, O², i, grid, env; trunc, maxiter = 0) ≈
        vals_ref[[5, 3, 1]]

    # The rotated distribution only permits column sweeps.
    # Check rotation and automatic selection together.
    cell = size(ρ)[1:2]
    ic = PEPSKit.siterotr90(i, cell)
    jcs = PEPSKit.siterotr90.(js, Ref(cell))
    ρc, envc = rotr90(ρ), rotr90(env)
    @test correlator_approx(ρc, O², ic, jcs, envc; trunc, maxiter = 0) ≈ vals_rows

    # Test auto sweep direction choice in for southwest targets.
    selection_trunc = truncrank(2)
    selection_alg = PEPSKit.WindowApprox(Zipup(; trunc = selection_trunc), nothing)
    for (offset, direction) in ((CI(1, -2), :rows), (CI(2, -1), :columns), (CI(1, -1), :rows))
        j = i + offset
        expected = only(PEPSKit._correlator_approx(ρ, O², i, [j], env, selection_alg, direction))
        @test correlator_approx(ρ, O², i, j, env; trunc = selection_trunc, maxiter = 0) == expected
    end

    # Check the cached normalization against a full contraction without observables.
    alg = PEPSKit.WindowApprox(Zipup(; trunc), nothing)
    rowrange, colrange = PEPSKit._window_ranges([i; js])
    cache = PEPSKit._window_row_cache(ρ, env, rowrange, colrange, alg)
    @test cache.norm ≈ PEPSKit._contract_window_rows(ρ, nothing, env, rowrange, colrange, alg)

    # Verify that south_boundaries[k] closes the north state after row k, catching indexing errors.
    north = cache.north_boundary
    for k in eachindex(cache.row_mpos)
        north = PEPSKit._approximate(cache.row_mpos[k], north, alg)
        @test dot(cache.south_boundaries[k], north) ≈ cache.norm
    end

    # Replacing a site must modify the returned MPO while leaving the cached row intact.
    W = first(cache.row_mpos)
    W_before = deepcopy(parent(W))
    replacement = 2 * parent(W)[2]
    modified = PEPSKit._row_mpo_with_site(cache, replacement, first(rowrange), first(colrange))
    @test parent(W) == W_before
    @test parent(modified)[2] == replacement

    # Check that the planar row-MPO adjoint agrees with the full tensor adjoint.
    @test convert(TensorMap, PEPSKit._adjoint_mpo(W)) ≈ convert(TensorMap, W)'
end

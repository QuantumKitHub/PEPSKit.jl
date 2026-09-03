using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: TensorProductTerm, add_term!, _scale_local_term
import TensorKitTensors.SpinOperators as SO

# `LocalOperator` terms given as an explicit tensor product of single-site operators, one
# per site acted on, rather than as the tensor product formed up front.
#
# The transverse-field Ising model is the natural test case: with trivial symmetry every set
# of sites carries exactly one term, so the whole Hamiltonian is a sum of tensor products,
# a two-factor product per nearest neighbor bond and a one-factor product per site. This
# requires trivial symmetry, since a single-site σᶻ is not symmetric under the Z₂ symmetry
# of the model.

"""
Transverse-field Ising Hamiltonian built from tensor product terms, matching the conventions
of [`transverse_field_ising`](@ref).
"""
function transverse_field_ising_tensorproduct(
        T::Type{<:Number}, lattice::InfiniteSquare; J = 1.0, g = 1.0
    )
    Z, X = SO.σᶻ(T, Trivial), SO.σˣ(T, Trivial)
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => [-J * Z, copy(Z)] for neighbor in nearest_neighbours(lattice))...,
        ([idx] => [(-J * g) * X] for idx in vertices(lattice))...,
    )
end
transverse_field_ising_tensorproduct(lattice::InfiniteSquare; kwargs...) =
    transverse_field_ising_tensorproduct(ComplexF64, lattice; kwargs...)

const Dbond, χenv, g = 2, 8, 3.1

@testset "Tensor product terms reproduce the two-site Ising Hamiltonian ($uc)" for uc in
    ((1, 1), (2, 2))
    Random.seed!(2985721)
    H = transverse_field_ising(InfiniteSquare(uc...); g)
    H_prod = transverse_field_ising_tensorproduct(InfiniteSquare(uc...); g)

    # one term per site set: a two-factor product per bond, a one-factor product per site
    @test length(H_prod.terms) == length(H.terms)
    @test all(t -> t isa TensorProductTerm, values(H_prod.terms))
    @test sort(length.(collect(values(H_prod.terms)))) ==
        sort(vcat(fill(1, prod(uc)), fill(2, 2 * prod(uc))))

    peps = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond); unitcell = uc)
    env, = leading_boundary(CTMRGEnv(peps, ComplexSpace(χenv)), peps; tol = 1.0e-8, verbosity = 0)

    @test expectation_value(peps, H_prod, env) ≈ expectation_value(peps, H, env) rtol = 1.0e-9

    # scaling a product term scales the term, not each of its factors
    α = 0.37
    @test expectation_value(peps, α * H_prod, env) ≈ α * expectation_value(peps, H_prod, env) rtol =
        1.0e-9
end

@testset "Tensor product term bookkeeping" begin
    lattice = fill(ComplexSpace(2), 1, 1)
    X, Z = SO.σˣ(ComplexF64, Trivial), SO.σᶻ(ComplexF64, Trivial)

    # a sum of tensor products is not a tensor product, so terms may not be accumulated
    O = LocalOperator(lattice, [(1, 1), (1, 2)] => [X, X])
    @test_throws ArgumentError add_term!(O, [(1, 1), (1, 2)], [Z, Z])

    # factors are reordered along with their sites
    O1 = LocalOperator(lattice, [(1, 1), (1, 2)] => [X, Z])
    O2 = LocalOperator(lattice, [(1, 2), (1, 1)] => [Z, X])
    @test only(keys(O1.terms)) == only(keys(O2.terms))
    @test all(map(≈, only(values(O1.terms)), only(values(O2.terms))))

    # each factor must be a single-site operator, and match the physical space
    @test_throws ArgumentError LocalOperator(lattice, [(1, 1), (1, 2)] => [X])  # arity
    @test_throws ArgumentError LocalOperator(lattice, [(1, 1)] => [X ⊗ X])         # not single-site
    @test_throws SpaceMismatch LocalOperator(
        lattice, [(1, 1)] => [SO.S_x(ComplexF64, Trivial; spin = 1)]
    )

    # scaling touches exactly one factor
    scaled = _scale_local_term([X, Z], 3.0)
    @test scaled[1] ≈ 3.0 * X
    @test scaled[2] ≈ Z

    # the real part of a product is the product of the real parts of its factors
    O3 = LocalOperator(lattice, [(1, 1), (1, 2)] => [(2.0 + 3.0im) * Z, copy(Z)])
    re = only(values(real(O3).terms))
    @test re[1] ≈ real((2.0 + 3.0im) * Z)
    @test re[2] ≈ real(Z)
end

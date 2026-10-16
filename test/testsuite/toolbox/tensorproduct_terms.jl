using Test
using Random
using TensorKit
using PEPSKit, Adapt
using PEPSKit: add_term!
import TensorKitTensors.SpinOperators as SO

const Dbond, χenv, g = 2, 8, 3.1
const unitcells = ((1, 1), (2, 2))

"""
Build the transverse-field Ising Hamiltonian from real single-site factors.
"""
function transverse_field_ising_tensorproduct(AT, lattice::InfiniteSquare; g = 1.0)
    Z, X = adapt.(Ref(AT), (SO.σᶻ(Float64, Trivial), SO.σˣ(Float64, Trivial)))
    spaces = fill(domain(X)[1], (lattice.Nrows, lattice.Ncols))
    return LocalOperator(
        spaces,
        (neighbor => [-Z, Z] for neighbor in nearest_neighbours(lattice))...,
        ([idx] => [-g * X] for idx in vertices(lattice))...,
    )
end

function toolbox_tensorproduct_ising(AT)
    return @testset "Tensor product Ising Hamiltonian ($uc) ($AT)" for uc in unitcells
        Random.seed!(2985721)
        H = transverse_field_ising(InfiniteSquare(uc...); g)
        H_prod = transverse_field_ising_tensorproduct(AT, InfiniteSquare(uc...); g)
        peps = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond); unitcell = uc))
        env, = leading_boundary(
            CTMRGEnv(peps, ComplexSpace(χenv)), peps; tol = 1.0e-8, verbosity = 0
        )
        E = expectation_value(peps, H, env)
        @test expectation_value(peps, H_prod, env) ≈ E rtol = 1.0e-9
        # Real factors must retain tensor-product dispatch after complex scaling.
        α = 2 + 3im
        @test expectation_value(peps, α * H_prod, env) ≈ α * E rtol = 1.0e-9
    end
end

function toolbox_tensorproduct_bookkeeping(AT)
    return @testset "Tensor product bookkeeping ($AT)" begin
        lattice = fill(ComplexSpace(2), 1, 1)
        X, Z = adapt.(Ref(AT), (SO.σˣ(Float64, Trivial), SO.σᶻ(Float64, Trivial)))
        sites = CartesianIndex.([(3, 4), (3, 5)])
        factors = [X, Z]
        product = LocalOperator(lattice, sites => factors)
        @test sites == CartesianIndex.([(3, 4), (3, 5)])
        reverse!(sites)
        reverse!(factors)
        @test only(product.terms) == (CartesianIndex.([(1, 1), (1, 2)]) => [X, Z])
        @test product == LocalOperator(lattice, [(1, 2), (1, 1)] => [Z, X])

        snapshot = deepcopy(product)
        scaled = (2 + 3im) * product
        @test product == snapshot
        @test scalartype(scaled) == ComplexF64
        @test_throws ArgumentError add_term!(product, [(1, 1), (1, 2)], [Z, X])
        @test_throws ArgumentError product + product

        # Factorwise real/imaginary parts are incorrect even for tensor products.
        imaginary = LocalOperator(lattice, [(1, 1), (1, 2)] => [im * Z, im * Z])
        @test_throws ArgumentError real(imaginary)
        @test_throws ArgumentError imag(imaginary)
        @test_throws ArgumentError LocalOperator(lattice, CartesianIndex{2}[] => typeof(X)[])
        @test_throws ArgumentError LocalOperator(lattice, [(1, 1), (1, 2)] => [X])
        @test_throws SpaceMismatch LocalOperator(
            lattice, [(1, 1)] => [SO.S_x(Float64, Trivial; spin = 1)]
        )
    end
end

using Test
using Random
using TensorKit
using PEPSKit
using PEPSKit: MPOTerm, TensorProductTerm, gate_to_mpo, add_term!
import TensorKitTensors.SpinOperators as SO

# `LocalOperator` terms given as a matrix product operator, one tensor per site acted on,
# rather than as the dense rank-2N tensor. `gate_to_mpo` turns a dense operator into one.
#
# A tensor product term is the trivial-bond case, and is the narrower type, so both are
# vectors of tensors and dispatch separates them by whether the factors are rank-2.

"""
Largest bond dimension across the cuts of an MPO term, i.e. the Schmidt rank of the operator
it represents. The outgoing bond of a factor is its last domain index, except for a rank-2
factor, which carries no bond at all.
"""
function mpobond(term)
    length(term) == 1 && return 1
    return maximum(1:(length(term) - 1)) do i
        W = term[i]
        numind(W) == 2 && return 1
        return dim(space(W, numind(W)))
    end
end

const Dbond, χenv = 2, 8

X, Z = SO.σˣ(ComplexF64, Trivial), SO.σᶻ(ComplexF64, Trivial)
const lattice11 = fill(ComplexSpace(2), 1, 1)

@testset "MPO factors follow the standard convention" begin
    # rank-2 at the ends of the chain, rank-4 in the bulk
    W1, W2 = gate_to_mpo(X ⊗ X + Z ⊗ Z)
    @test (numout(W1), numin(W1)) == (1, 2)
    @test (numout(W2), numin(W2)) == (2, 1)

    Ws = gate_to_mpo(X ⊗ X ⊗ X + Z ⊗ Z ⊗ Z)
    @test length(Ws) == 3
    @test (numout(Ws[1]), numin(Ws[1])) == (1, 2)
    @test (numout(Ws[2]), numin(Ws[2])) == (2, 2)
    @test (numout(Ws[3]), numin(Ws[3])) == (2, 1)

    # the bond is the Schmidt rank across each cut
    @test mpobond(gate_to_mpo(X ⊗ X)) == 1                    # a pure product
    @test mpobond(gate_to_mpo(X ⊗ X + Z ⊗ Z)) == 2
    @test mpobond(gate_to_mpo(X ⊗ X ⊗ X + Z ⊗ Z ⊗ Z)) == 2
    @test mpobond([X]) == 1                                    # a lone site has no bond

    # Heisenberg XYZ has Schmidt rank 3 at every spin
    for spin in (1 // 2, 1, 3 // 2)
        term = SO.S_x_S_x(ComplexF64, Trivial; spin) +
            SO.S_y_S_y(ComplexF64, Trivial; spin) +
            SO.S_z_S_z(ComplexF64, Trivial; spin)
        @test mpobond(gate_to_mpo(term)) == 3
    end
end

@testset "Dispatch separates products from MPOs" begin
    # a vector of rank-2 operators is both, and the narrower type wins where it matters
    @test [X, Z] isa TensorProductTerm
    @test [X, Z] isa MPOTerm

    # anything carrying a bond is only an MPO term
    for O in (X ⊗ Z, X ⊗ X + Z ⊗ Z, X ⊗ X ⊗ X + Z ⊗ Z ⊗ Z)
        Ws = gate_to_mpo(O)
        @test Ws isa MPOTerm
        @test !(Ws isa TensorProductTerm)
    end
end

@testset "MPO terms reproduce dense terms ($(name))" for (name, inds, O) in (
        ("NN horizontal", [(1, 1), (1, 2)], -(X ⊗ X) + Z ⊗ Z),
        ("NN vertical", [(1, 1), (2, 1)], -(X ⊗ X) + Z ⊗ Z),
        ("NNN diagonal", [(1, 1), (2, 2)], X ⊗ Z),
        ("3 site line", [(1, 1), (1, 2), (1, 3)], X ⊗ X ⊗ X + Z ⊗ Z ⊗ Z),
    )
    Random.seed!(2985721)
    peps = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    env = CTMRGEnv(randn, ComplexF64, peps, ComplexSpace(χenv))

    H_dense = LocalOperator(lattice11, inds => O)
    H_mpo = LocalOperator(lattice11, inds => gate_to_mpo(O))
    @test expectation_value(peps, H_mpo, env) ≈ expectation_value(peps, H_dense, env) rtol =
        1.0e-9
end

@testset "MPO terms reproduce the Heisenberg XYZ Hamiltonian" begin
    Random.seed!(2985721)
    H = heisenberg_XYZ(InfiniteSquare())
    H_mpo = LocalOperator(
        physicalspace(H), (inds => gate_to_mpo(op) for (inds, op) in H.terms)...
    )
    @test all(t -> t isa MPOTerm, values(H_mpo.terms))
    @test all(t -> mpobond(t) == 3, values(H_mpo.terms))

    peps = InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond))
    env, = leading_boundary(
        CTMRGEnv(peps, ComplexSpace(χenv)), peps; tol = 1.0e-8, verbosity = 0
    )
    E = expectation_value(peps, H, env)
    @test expectation_value(peps, H_mpo, env) ≈ E rtol = 1.0e-9

    # scaling an MPO term scales the term, not each of its factors
    α = 0.37
    @test expectation_value(peps, α * H_mpo, env) ≈ α * E rtol = 1.0e-9

    # the real part is taken factor by factor
    H_real = real(H_mpo)
    @test all(t -> t isa MPOTerm, values(H_real.terms))
    @test all(all(W -> norm(imag(W)) < 1.0e-14, t) for t in values(H_real.terms))
end

@testset "MPO term bookkeeping" begin
    # the factors are ordered along the chain, so the sites may not be permuted
    @test_throws ArgumentError LocalOperator(
        lattice11, [(1, 2), (1, 1)] => gate_to_mpo(X ⊗ Z)
    )

    # a mismatch between sites and factors is caught
    @test_throws ArgumentError LocalOperator(
        lattice11, [(1, 1), (1, 2), (1, 3)] => gate_to_mpo(X ⊗ Z)
    )

    # factor ranks must match their position in the chain
    @test_throws ArgumentError LocalOperator(lattice11, [(1, 1)] => [X ⊗ X])
    @test_throws ArgumentError LocalOperator(lattice11, [(1, 1)] => gate_to_mpo(X ⊗ Z))

    # an MPO is not summed in place, since the bond dimension of a sum is not the same
    O = LocalOperator(lattice11, [(1, 1), (1, 2)] => gate_to_mpo(X ⊗ Z))
    @test_throws ArgumentError add_term!(
        O, [CartesianIndex(1, 1), CartesianIndex(1, 2)], gate_to_mpo(Z ⊗ X)
    )

    # physical spaces are checked
    @test_throws SpaceMismatch LocalOperator(
        lattice11, [(1, 1), (1, 2)] => gate_to_mpo(
            SO.S_x_S_x(ComplexF64, Trivial; spin = 1)
        )
    )
end

using Test
using Random
using TensorKit
using PEPSKit, Adapt
using PEPSKit: gate_to_mpo, add_term!
import TensorKitTensors.SpinOperators as SO

const Dbond, χenv = 2, 8

const X = SO.σˣ(Float64, Trivial)
const Z = SO.σᶻ(Float64, Trivial)
const lattice11 = fill(ComplexSpace(2), 1, 1)

const dense_terms = (
    ("NN horizontal", [(1, 1), (1, 2)], -(X ⊗ X) + Z ⊗ Z),
    ("NN vertical", [(1, 1), (2, 1)], -(X ⊗ X) + Z ⊗ Z),
    ("NNN diagonal", [(1, 1), (2, 2)], X ⊗ Z),
    ("3 site line", [(1, 1), (1, 2), (1, 3)], X ⊗ X ⊗ X + Z ⊗ Z ⊗ Z),
    # Asymmetric operators expose accidental reordering of sites or factors.
    ("NNN anti-diagonal", [(1, 2), (2, 1)], X ⊗ Z),
    ("reversed sites", [(1, 2), (1, 1)], X ⊗ Z),
)

function toolbox_mpo_terms_dense(AT)
    Random.seed!(2985721)
    return @testset "MPO contractions ($AT)" begin
        peps = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(Dbond)))
        env = CTMRGEnv(randn, ComplexF64, peps, ComplexSpace(χenv))
        @testset "$name" for (name, inds, O) in dense_terms
            O = adapt(AT, O)
            H_dense = LocalOperator(lattice11, inds => O)
            H_mpo = LocalOperator(lattice11, inds => gate_to_mpo(O))
            @test expectation_value(peps, H_mpo, env) ≈ expectation_value(peps, H_dense, env) rtol = 1.0e-9
        end

        # Complex scaling must promote real factors without scaling the coefficient twice.
        _, inds, O = first(dense_terms)
        O = adapt(AT, O)
        H_mpo = LocalOperator(lattice11, inds => gate_to_mpo(O))
        α = 2 + 3im
        @test expectation_value(peps, α * H_mpo, env) ≈
            α * expectation_value(peps, LocalOperator(lattice11, inds => O), env) rtol = 1.0e-9

        # The BP fallback must agree with the existing dense BP contractions.
        bp_env = BPEnv(peps)
        x, z = adapt.(Ref(AT), (X, Z))
        @testset "BP ($name)" for (name, sites, dense, factors) in (
                ("one-site product", [(1, 1)], x, [x]),
                ("two-site product", [(1, 1), (1, 2)], x ⊗ z, [x, z]),
                ("MPO", inds, O, gate_to_mpo(O)),
            )
            H_dense = LocalOperator(lattice11, sites => dense)
            H_factors = LocalOperator(lattice11, sites => factors)
            @test expectation_value(peps, H_factors, bp_env) ≈
                expectation_value(peps, H_dense, bp_env) rtol = 1.0e-9
        end
    end
end

function toolbox_mpo_pepo(AT)
    return @testset "Factored PEPO observables ($p, purified=$purified) ($AT)" for p in (ℂ^2, Vect[FermionParity](0 => 1, 1 => 1)), purified in (false, true)
        Random.seed!(425)
        rho = adapt(AT, InfinitePEPO(p, p; unitcell = (1, 1, 1)))
        env = CTMRGEnv(purified ? InfinitePEPS(rho) : InfinitePartitionFunction(rho), p)
        # The purified form supplies rho as both bra and ket.
        args = purified ? (rho, env) : (env,)
        a, b = ntuple(_ -> adapt(AT, randn(ComplexF64, p ← p)), 2)
        gate = adapt(AT, randn(ComplexF64, p ⊗ p ← p ⊗ p))
        @testset "$name" for (name, sites, dense, factors) in (
                ("one-site product", [(1, 1)], a, [a]),
                ("two-site product", [(1, 1), (1, 2)], a ⊗ b, [a, b]),
                ("MPO", [(1, 1), (1, 2)], gate, gate_to_mpo(gate; trunc = notrunc())),
            )
            H_dense = LocalOperator(physicalspace(rho), sites => dense)
            H_factors = LocalOperator(physicalspace(rho), sites => factors)
            @test expectation_value(rho, H_factors, args...) ≈
                expectation_value(rho, H_dense, args...) rtol = 1.0e-9
        end
    end
end

function toolbox_mpo_bookkeeping(AT)
    Random.seed!(2985721)
    return @testset "MPO bookkeeping ($AT)" begin
        d = ℂ^2
        lattice = fill(d, 2, 2)
        dense = adapt(AT, randn(Float64, d ⊗ d ← d ⊗ d))
        mpo = gate_to_mpo(dense; trunc = notrunc())
        sites = CartesianIndex.([(3, 4), (3, 3)])
        original_sites, original_mpo = copy(sites), copy(mpo)
        operator = LocalOperator(lattice, sites => mpo)
        @test sites == original_sites
        reverse!(sites)
        reverse!(mpo)
        @test only(operator.terms) == (CartesianIndex.([(1, 2), (1, 1)]) => original_mpo)

        snapshot = deepcopy(operator)
        scaled = (2 + 3im) * operator
        @test operator == snapshot
        onsite = adapt(AT, randn(ComplexF64, d ← d))
        @test scalartype(operator + LocalOperator(lattice, ((2, 1),) => onsite)) == ComplexF64
        @test scalartype(scaled) == ComplexF64
        @test_throws ArgumentError real(scaled)
        @test_throws ArgumentError imag(scaled)

        # Both insertion and addition must reject accumulation involving an MPO.
        inds = CartesianIndex.([(1, 1), (1, 2)])
        dense_operator = LocalOperator(lattice, inds => dense)
        mpo_operator = LocalOperator(lattice, inds => original_mpo)
        for (left, right) in ((mpo_operator, mpo_operator), (mpo_operator, dense_operator), (dense_operator, mpo_operator))
            @test_throws ArgumentError add_term!(deepcopy(left), inds, only(values(right.terms)))
            @test_throws ArgumentError left + right
        end
        @test isempty(LocalOperator(lattice, inds => [zero(first(original_mpo)), last(original_mpo)]).terms)

        # Unequal physical spaces expose factor reordering during coordinate transformations.
        lattice = [ℂ^2 ℂ^3 ℂ^4; ℂ^5 ℂ^6 ℂ^7]
        dense = adapt(AT, randn(Float64, ℂ^6 ⊗ ℂ^2 ← ℂ^6 ⊗ ℂ^2))
        operator = LocalOperator(lattice, ((2, 2), (1, 1)) => gate_to_mpo(dense; trunc = notrunc()))
        @test rotr90(rotl90(operator)) == operator
        sites, term = only(operator.terms)
        @test repeat(operator, 2, 1).terms == Dict(sites => term, (sites .+ CartesianIndex(2, 0)) => term)
    end
end

function toolbox_mpo_validation(AT)
    Random.seed!(2985721)
    return @testset "MPO validation ($AT)" begin
        d, b1, b2 = ℂ^2, ℂ^3, ℂ^4
        lattice = fill(d, 2, 2)
        first_tensor = adapt(AT, randn(Float64, d ← d ⊗ b1))
        last_tensor = adapt(AT, randn(Float64, b1 ⊗ d ← d))
        mpo = [first_tensor, last_tensor]
        sites = ((1, 1), (1, 2))

        @test_throws ArgumentError LocalOperator(lattice, CartesianIndex{2}[] => AbstractTensorMap[])
        @test_throws ArgumentError LocalOperator(lattice, ((1, 1),) => mpo)
        @test_throws ArgumentError LocalOperator(lattice, ((1, 1), (1, 1)) => mpo)
        @test_throws ArgumentError LocalOperator(lattice, sites => reverse(mpo))

        # Bond matching and the two physical legs are independent constraints.
        wrong_bond = adapt(AT, randn(Float64, b2 ⊗ d ← d))
        wrong_input = adapt(AT, randn(Float64, b1 ⊗ d ← ℂ^3))
        wrong_output = adapt(AT, randn(Float64, b1 ⊗ ℂ^3 ← d))
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_bond])
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_input])
        @test_throws SpaceMismatch LocalOperator(lattice, sites => [first_tensor, wrong_output])
    end
end

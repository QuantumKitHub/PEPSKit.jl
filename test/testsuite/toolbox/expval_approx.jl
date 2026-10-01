using TensorKit
using PEPSKit
using MPSKit
using Test
using Adapt
using Random

const CI = CartesianIndex

spaces = Dict(
    U1Irrep => (
        U1Space(1 => 2, -1 => 1),
        U1Space(1 => 1, 0 => 1, -1 => 2),
        U1Space(1 => 1, 0 => 1, -1 => 2),
    ),
    FermionParity => (
        Vect[FermionParity](0 => 1, 1 => 1),
        Vect[FermionParity](0 => 1, 1 => 2),
        Vect[FermionParity](0 => 2, 1 => 2),
    ),
)

sites_list = (
    [CI(1, 1), CI(1, 2)], # horizontal
    [CI(1, 1), CI(2, 1)], # vertical
    [CI(1, 1), CI(2, 2)], # turned
    [CI(2, 2), CI(1, 1)], # reversed turned
    [CI(2, 1), CI(1, 1), CI(1, 2), CI(2, 2)], # U-shaped
)

"""
Check approximate expectation values against exact single-layer PEPO contractions.
"""
function toolbox_expval_approx(AT)
    return @testset "Single-layer PEPO ($S) ($AT)" for S in keys(spaces)
        Random.seed!(1234)

        d, D, χ = spaces[S]
        ρ = adapt(AT, InfinitePEPO(d, D; unitcell = (2, 2, 1)))
        env = CTMRGEnv(InfinitePartitionFunction(ρ), χ)
        lattice = physicalspace(ρ)
        trunc = notrunc()

        # Dense terms may change MPO order when forming a snake; explicit MPOs must keep theirs.
        # Exact contractions expose permutation or braiding errors in both paths and sweep orientations.
        for sites in sites_list
            n = length(sites)
            op = adapt(AT, randn(ComplexF64, d^n → d^n))
            mpo = PEPSKit.gate_to_mpo(op; trunc)
            dense = LocalOperator(lattice, sites => op)
            factorized = LocalOperator(lattice, sites => mpo)
            exact = expectation_value(ρ, dense, env)
            for observable in (dense, factorized), direction in (:rows, :columns)
                @test expectation_value_approx(
                    ρ, observable, env; trunc, maxiter = 0, direction
                ) ≈ exact
            end
            # Square windows must default to row sweeps.
            if sites == sites_list[3]
                auto = expectation_value_approx(ρ, dense, env; trunc = truncrank(2), maxiter = 0)
                rows = expectation_value_approx(ρ, dense, env; trunc = truncrank(2), maxiter = 0, direction = :rows)
                @test auto == rows
            end
        end
    end
end

"""
Check single-site dispatch, sparse snakes, and sums of independently normalized terms.
"""
function toolbox_expval_approx_localoperator(AT)
    return @testset "LocalOperator interface ($AT)" begin
        Random.seed!(4321)
        d = Vect[FermionParity](0 => 1, 1 => 1)
        ρ = adapt(AT, InfinitePEPO(d, d; unitcell = (2, 2, 1)))
        env = CTMRGEnv(InfinitePartitionFunction(ρ), d)
        lattice = physicalspace(ρ)
        trunc = notrunc()

        # Single-site terms bypass MPO decomposition, which requires at least two sites.
        op1 = adapt(AT, randn(ComplexF64, d → d))
        one_site = LocalOperator(lattice, [CI(1, 1)] => op1)
        one_factor = LocalOperator(lattice, [CI(1, 1)] => [op1])
        for observable in (one_site, one_factor)
            @test expectation_value_approx(ρ, observable, env; trunc, maxiter = 0) ≈
                expectation_value(ρ, one_site, env)
        end

        # A greedy horizontal-first route revisits a site for this support.
        # The dense snake must connect outside the column's endpoints and preserve the operator.
        op3 = adapt(AT, randn(ComplexF64, d^3 → d^3))
        snake = LocalOperator(lattice, [CI(1, 0), CI(0, 1), CI(2, 1)] => op3)
        @test expectation_value_approx(ρ, snake, env; trunc, maxiter = 0) ≈
            expectation_value(ρ, snake, env)

        # These terms use different normalization windows; the gapped MPO also inserts a string.
        # Non-unit-cell coordinates and both sweeps exercise translation and rotation of the expanded path.
        op2 = adapt(AT, randn(ComplexF64, d^2 → d^2))
        sites2 = [CI(-1, 0), CI(-1, 2)]
        mixed = LocalOperator(lattice, [CI(1, 1)] => op1, sites2 => PEPSKit.gate_to_mpo(op2; trunc))
        dense_sum = LocalOperator(lattice, [CI(1, 1)] => op1, sites2 => op2)
        for direction in (:rows, :columns)
            @test expectation_value_approx(ρ, mixed, env; trunc, maxiter = 0, direction) ≈
                expectation_value(ρ, dense_sum, env)
        end

        # Complex scaling leaves real factors in the MPO; expansion must retain a valid mixed-scalar container.
        real_op = real(op2)
        real_mpo = LocalOperator(lattice, sites2 => PEPSKit.gate_to_mpo(real_op; trunc))
        coefficient = 2 + 3im
        @test expectation_value_approx(ρ, coefficient * real_mpo, env; trunc, maxiter = 0) ≈
            coefficient * expectation_value(ρ, LocalOperator(lattice, sites2 => real_op), env)

        # Constructors discard zero terms, so an empty sum must produce a scalar zero without contracting.
        @test iszero(expectation_value_approx(ρ, LocalOperator(lattice), env))
        # Reject a mismatched unit cell even when the individual operator's physical space matches.
        wrong_lattice = LocalOperator(fill(d, 1, 1), [CI(1, 1)] => op1)
        @test_throws ArgumentError expectation_value_approx(ρ, wrong_lattice, env)
    end
end

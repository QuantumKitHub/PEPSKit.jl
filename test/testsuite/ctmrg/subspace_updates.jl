using Test
using Random
using LinearAlgebra
using MatrixAlgebraKit
using TensorKit
using PEPSKit
using Adapt
using MatrixAlgebraKit: TruncatedAlgorithm

"""Count the orthogonalizations performed by a production range update."""
struct SubspaceCountingOrth{A} <: MatrixAlgebraKit.AbstractAlgorithm
    alg::A
    qr::Base.RefValue{Int}
    lq::Base.RefValue{Int}
end

"""Count QR calls while using the ordinary forward factorization."""
function MatrixAlgebraKit.left_orth!(t::AbstractTensorMap, a::SubspaceCountingOrth)
    a.qr[] += 1
    return left_orth!(t, a.alg)
end

"""Count LQ calls while using the ordinary forward factorization."""
function MatrixAlgebraKit.right_orth!(t::AbstractTensorMap, a::SubspaceCountingOrth)
    a.lq[] += 1
    return right_orth!(t, a.alg)
end

"""Evaluate the defining recurrences using an explicit full environment."""
function expected_subspace_ranges(M, X, q)
    Q = first(left_orth(M * X))
    for _ in 1:q
        X = first(left_orth(M' * Q))
        Q = first(left_orth(M * X))
    end
    return first(left_orth(M' * Q)), Q
end

"""Check recurrences for the subspace iteration projector."""
function ctmrg_subspace_updates_recurrences(AT)
    return @testset "Range recurrences and factorization counts" begin
        @testset "$T, q=$q" for T in (Float64, ComplexF64), q in (0, 1)
            Random.seed!(264)
            L, R = adapt(AT, randn(T, ℂ^9 ← ℂ^9)), adapt(AT, randn(T, ℂ^9 ← ℂ^7))
            X = first(left_orth(adapt(AT, randn(T, ℂ^7 ← ℂ^4))))
            counter = SubspaceCountingOrth(OrthAdjoint().fwd_alg, Ref(0), Ref(0))
            alg = SubspaceIterationProjector(;
                subspace = ℂ^4, n_subspace_iter = q,
                rotate_basis = false, orth_alg = OrthAdjoint(; fwd_alg = counter)
            )
            before = copy(X)
            trunc = truncspace(ℂ^2)
            result = PEPSKit._subspace_iteration_decomposition(L, R, alg, (; X, trunc))
            expected_count = 2(q + 1)
            @test counter.qr[] + counter.lq[] == expected_count
            @test counter.lq[] == 1
            @test X == before
            Xref, Yref = expected_subspace_ranges(L * R, X, q)
            @test result.X * result.X' ≈ Xref * Xref' atol = 1.0e-10
            Ynew = PEPSKit.rangefinder_update(X, L, R, alg.subspace, alg).Y
            @test Ynew * Ynew' ≈ Yref * Yref' atol = 1.0e-10
            @test Ynew' * Ynew ≈ id(T, domain(Ynew)) atol = 1.0e-10
            core = Yref' * L * R * Xref
            u, s, v, = svd_trunc(core, TruncatedAlgorithm(DefaultAlgorithm(), trunc))
            expected = Yref * u * s * v * Xref'
            actual = result.U * result.S * result.V
            @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10

        end
    end
end

"""Check complete sectors for the subspace iteration projector."""
function ctmrg_subspace_updates_complete_sectors(AT)
    return @testset "Complete and rectangular sectors" begin
        @testset "$m ← $n, q=$q" for (m, n) in ((7, 5), (5, 7), (5, 5)), q in (0, 1)
            Random.seed!(265)
            M = adapt(AT, randn(ComplexF64, ℂ^m ← ℂ^n))
            alg = SubspaceIterationProjector(; subspace = ℂ^20, n_subspace_iter = q, rotate_basis = false)
            trunc = truncspace(ℂ^2)
            result = PEPSKit._subspace_iteration_decomposition(
                M, id(ℂ^n), alg,
                (; X = nothing, trunc)
            )
            u, s, v, = svd_trunc(M, TruncatedAlgorithm(DefaultAlgorithm(), trunc))
            actual, expected = result.U * result.S * result.V, u * s * v
            @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10
            @test result.X' * result.X ≈ id(ComplexF64, domain(result.X)) atol = 1.0e-10

        end
        @testset "$S" for (S, charges) in (
                (U1Irrep, (0, 1)),
                (SU2Irrep, (0, 1 // 2)),
                (A4Irrep, (0, 3)),
                (FermionParity ⊠ U1Irrep, ((0, 0), (1, 1))),
                (FibonacciAnyon, (:I, :τ)),
            )
            W = Vect[S](charges[1] => 3, charges[2] => 7)
            K = Vect[S](charges[1] => 3, charges[2] => 4)
            χ = Vect[S](charges[1] => 2, charges[2] => 2)
            Random.seed!(266)
            A = adapt(AT, zeros(ComplexF64, W ← W))
            for (_, b) in blocks(A)
                b[diagind(b)] .= 0.1 .^ (0:(size(b, 1) - 1))
            end
            A = first(left_orth(adapt(AT, randn(ComplexF64, space(A))))) * A * last(right_orth(adapt(AT, randn(ComplexF64, space(A)))))
            alg = SubspaceIterationProjector(; subspace = K)
            state = (; X = nothing, trunc = truncspace(χ))
            local result
            for _ in 1:5
                result = PEPSKit._subspace_iteration_decomposition(A, id(ComplexF64, W), alg, state)
                state = (; result.X, state.trunc)
            end
            u, s, v, = svd_trunc(A, TruncatedAlgorithm(DefaultAlgorithm(), state.trunc))
            actual, expected = result.U * result.S * result.V, u * s * v
            @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-8
            @test domain(result.X) == ProductSpace(K)
            @test result.U' * result.U ≈ id(ComplexF64, χ) atol = 1.0e-10
        end
    end
end

"""Check recovery for the subspace iteration projector."""
function ctmrg_subspace_updates_recovery(AT)
    return @testset "Cache scalar recovery" begin
        @testset "$flavor" for flavor in (SimultaneousCTMRG, SequentialCTMRG)
            Random.seed!(268)
            ψ = adapt(AT, InfinitePEPS(ℂ^2, ℂ^2))
            network, env = InfiniteSquareNetwork(ψ), CTMRGEnv(ψ, ℂ^3)
            alg = flavor(0.0, 1, 1, 0, SubspaceIterationProjector(; subspace = ℂ^5))
            real_ψ = adapt(AT, InfinitePEPS(randn, Float64, ℂ^2, ℂ^2))
            real_network, real_env = InfiniteSquareNetwork(real_ψ), CTMRGEnv(real_ψ, ℂ^3)
            real_cache = PEPSKit.initialize_projector_cache(real_network, real_env, alg)
            _, _, reset = PEPSKit.ctmrg_iteration(network, env, alg, real_cache)
            @test scalartype(first(reset.X)) == ComplexF64
        end
    end
end

"""Check initialization for the subspace iteration projector."""
function ctmrg_subspace_updates_initialization(AT)
    return @testset "Seeded initialization" begin
        a, b = [SubspaceIterationProjector(; subspace = ℂ^4, rng = Random.Xoshiro(269)) for _ in 1:2]
        first_bases = PEPSKit.rangefinder_initialize(ComplexF64, ProductSpace(ℂ^6), a.subspace, a)
        second_bases = PEPSKit.rangefinder_initialize(ComplexF64, ProductSpace(ℂ^6), b.subspace, b)
        @test first_bases == second_bases
        @test first_bases' * first_bases ≈ id(ComplexF64, ℂ^4) atol = 1.0e-12
    end
end

"""Check subspace range updates for the requested storage type."""
function ctmrg_subspace_updates(AT)
    return @testset "Subspace range updates ($AT)" begin
        ctmrg_subspace_updates_recurrences(AT)
        ctmrg_subspace_updates_complete_sectors(AT)
        ctmrg_subspace_updates_recovery(AT)
        ctmrg_subspace_updates_initialization(AT)
    end
end

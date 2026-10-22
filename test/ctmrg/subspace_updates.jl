using Test, PEPSKit, TensorKit, MatrixAlgebraKit, LinearAlgebra, Random
using MatrixAlgebraKit: TruncatedAlgorithm
BLAS.set_num_threads(1)
PEPSKit.set_scheduler!(:serial)

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
function expected_subspace_ranges(M, X, Y, update, q)
    if update == :independent
        for _ in 0:q
            X = first(left_orth(M' * M * X))
            Y = first(left_orth(M * M' * Y))
        end
    else
        for _ in 1:q
            X = first(left_orth(M' * M * X))
        end
        Q = first(left_orth(M * X))
        X = first(left_orth(M' * Q))
        Y = update == :alternating_refined ? first(left_orth(M * X)) : Q
    end
    return X, Y
end

@testset "Unified range recurrences and factorization counts" begin
    for T in (Float64, ComplexF64), update in (:independent, :alternating, :alternating_refined), q in (0, 1)
        Random.seed!(264)
        L, R = randn(T, ℂ^9 ← ℂ^9), randn(T, ℂ^9 ← ℂ^7)
        X, Y = first(left_orth(randn(T, ℂ^7 ← ℂ^4))), last(right_orth(randn(T, ℂ^4 ← ℂ^9)))'
        counter = SubspaceCountingOrth(OrthAdjoint().fwd_alg, Ref(0), Ref(0))
        alg = SubspaceIterationProjector(;
            subspace = ℂ^4, update, n_subspace_iter = q,
            rotate_basis = false, orth_alg = OrthAdjoint(; fwd_alg = counter)
        )
        before = deepcopy((X, Y))
        trunc = truncspace(ℂ^2)
        result = PEPSKit._subspace_iteration_decomposition(L, R, alg, (; X, Y, trunc, recycle = false))
        expected_count = update == :alternating_refined ? 2(q + 1) + 1 : 2(q + 1)
        @test counter.qr[] + counter.lq[] == expected_count
        if update == :alternating
            @test counter.lq[] == 1
        end
        @test (X, Y) == before
        Xref, Yref = expected_subspace_ranges(L * R, X, Y, update, q)
        @test result.X * result.X' ≈ Xref * Xref' atol = 1.0e-10
        @test result.Y * result.Y' ≈ Yref * Yref' atol = 1.0e-10
        @test domain(result.Y) == domain(result.X)
        @test result.Y' * result.Y ≈ id(T, domain(result.Y)) atol = 1.0e-10
        core = Yref' * L * R * Xref
        u, s, v, = svd_trunc(core, TruncatedAlgorithm(DefaultAlgorithm(), trunc))
        expected = Yref * u * s * v * Xref'
        actual = result.U * result.S * result.V
        @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10
        counter.qr[] = counter.lq[] = 0
        frozen = PEPSKit._subspace_iteration_decomposition(
            L, R, alg,
            (; X = result.X, Y = result.Y, trunc, recycle = true)
        )
        @test frozen.recycled && counter.qr[] + counter.lq[] == 0
    end
end

@testset "Unified complete and rectangular sectors" begin
    for update in (:independent, :alternating, :alternating_refined), (m, n) in ((7, 5), (5, 7), (5, 5)), q in (0, 1)
        Random.seed!(265)
        M = randn(ComplexF64, ℂ^m ← ℂ^n)
        alg = SubspaceIterationProjector(; subspace = ℂ^20, update, n_subspace_iter = q, rotate_basis = false)
        trunc = truncspace(ℂ^2)
        result = PEPSKit._subspace_iteration_decomposition(
            M, id(ℂ^n), alg,
            (; X = nothing, Y = nothing, trunc, recycle = false)
        )
        u, s, v, = svd_trunc(M, TruncatedAlgorithm(DefaultAlgorithm(), trunc))
        actual, expected = result.U * result.S * result.V, u * s * v
        @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10
        @test result.X' * result.X ≈ id(ComplexF64, domain(result.X)) atol = 1.0e-10
        repeated = PEPSKit._subspace_iteration_decomposition(
            M, id(ℂ^n), alg,
            (; result.X, result.Y, trunc, recycle = true)
        )
        @test repeated.recycled == (m == n)
    end
    for S in (U1Irrep, SU2Irrep, A4Irrep, FermionParity ⊠ U1Irrep, FibonacciAnyon), update in (:independent, :alternating, :alternating_refined)
        charges = S == U1Irrep ? (0, 1) : S == SU2Irrep ? (0, 1 // 2) : S == A4Irrep ? (0, 3) : S == FibonacciAnyon ? (:I, :τ) : ((0, 0), (1, 1))
        W = Vect[S](charges[1] => 3, charges[2] => 7)
        K = Vect[S](charges[1] => 3, charges[2] => 4)
        χ = Vect[S](charges[1] => 2, charges[2] => 2)
        Random.seed!(266)
        A = zeros(ComplexF64, W ← W)
        for (_, b) in blocks(A)
            b[diagind(b)] .= 0.1 .^ (0:(size(b, 1) - 1))
        end
        A = first(left_orth(randn(ComplexF64, space(A)))) * A * last(right_orth(randn(ComplexF64, space(A))))
        alg = SubspaceIterationProjector(; subspace = K, update)
        state = (; X = nothing, Y = nothing, trunc = truncspace(χ), recycle = false)
        local result
        for _ in 1:5
            result = PEPSKit._subspace_iteration_decomposition(A, id(ComplexF64, W), alg, state)
            state = (; result.X, result.Y, state.trunc, recycle = false)
        end
        u, s, v, = svd_trunc(A, TruncatedAlgorithm(DefaultAlgorithm(), state.trunc))
        actual, expected = result.U * result.S * result.V, u * s * v
        @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-8
        @test domain(result.X) == ProductSpace(K)
        @test result.U' * result.U ≈ id(ComplexF64, χ) atol = 1.0e-10
    end
end

@testset "Unified optional cache grids and freezing" begin
    for update in (:independent, :alternating, :alternating_refined), flavor in (SimultaneousCTMRG, SequentialCTMRG), tol in (0.0, 2.0)
        Random.seed!(267)
        ψ = InfinitePEPS(ℂ^2, ℂ^2; unitcell = (2, 1))
        network, env = InfiniteSquareNetwork(ψ), CTMRGEnv(ψ, ℂ^3)
        p = SubspaceIterationProjector(; subspace = ℂ^6, update, subspace_tol = tol, min_subspace_iters = 2)
        alg = flavor(0.0, 3, 3, 0, p)
        cache = PEPSKit.initialize_projector_cache(network, env, alg)
        initial = cache
        original = deepcopy(cache)
        @test isnothing(cache.Y) == (tol == 0 && update in (:alternating, :alternating_refined))
        @test isnothing(cache.U) == isnothing(cache.V) == (tol == 0)
        rotated = cache
        for _ in 1:4
            rotated = rotl90(rotated)
        end
        @test rotated.X == cache.X && rotated.Y == cache.Y
        local info
        for _ in 1:3
            env, info, cache = PEPSKit.ctmrg_iteration(network, env, alg, cache)
        end
        @test (initial.X, initial.Y, initial.U, initial.V) ==
            (original.X, original.Y, original.U, original.V)
        @test info.contraction_metrics.recycled == (tol > 0)
        @test cache.recycle == (tol > 0)
        @test isfinite(norm(env))
        disabled = PEPSKit.@set p.subspace_tol = 0.0
        thawed, error = PEPSKit._update_subspace_cache(cache, cache.X, cache.Y, cache.U, cache.V, disabled, true)
        @test !thawed.recycle && error == Inf
        @test isnothing(thawed.U) && isnothing(thawed.V)
        larger = CTMRGEnv(ψ, ℂ^4)
        _, _, reset = PEPSKit.ctmrg_iteration(network, larger, alg, cache)
        @test space(reset.X[1]) != space(cache.X[1])
        @test cache.iteration == 3 && initial.iteration == 0
    end
end

@testset "Unified cache configuration and scalar recovery" begin
    for flavor in (SimultaneousCTMRG, SequentialCTMRG)
        Random.seed!(268)
        ψ = InfinitePEPS(ℂ^2, ℂ^2)
        network, env = InfiniteSquareNetwork(ψ), CTMRGEnv(ψ, ℂ^3)
        alternating_alg = flavor(0.0, 1, 1, 0, SubspaceIterationProjector(; subspace = ℂ^5, update = :alternating, subspace_tol = 0))
        independent_alg = flavor(0.0, 1, 1, 0, SubspaceIterationProjector(; subspace = ℂ^5))
        cache = PEPSKit.initialize_projector_cache(network, env, alternating_alg)
        _, _, reset = PEPSKit.ctmrg_iteration(network, env, independent_alg, cache)
        @test !isnothing(reset.Y) && !isnothing(reset.U) && !isnothing(reset.V)
        real_ψ = InfinitePEPS(randn, Float64, ℂ^2, ℂ^2)
        real_network, real_env = InfiniteSquareNetwork(real_ψ), CTMRGEnv(real_ψ, ℂ^3)
        real_cache = PEPSKit.initialize_projector_cache(real_network, real_env, alternating_alg)
        _, _, reset = PEPSKit.ctmrg_iteration(network, env, alternating_alg, real_cache)
        @test scalartype(first(reset.X)) == ComplexF64
    end
end

@testset "Unified seeded initialization" begin
    for update in (:independent, :alternating, :alternating_refined)
        a, b = [SubspaceIterationProjector(; subspace = ℂ^4, update, subspace_tol = 0, rng = Random.Xoshiro(269)) for _ in 1:2]
        first_bases = PEPSKit.rangefinder_initialize(ComplexF64, ProductSpace(ℂ^7), ProductSpace(ℂ^6), a.subspace, a)
        second_bases = PEPSKit.rangefinder_initialize(ComplexF64, ProductSpace(ℂ^7), ProductSpace(ℂ^6), b.subspace, b)
        @test first_bases == second_bases
        @test first_bases[1]' * first_bases[1] ≈ id(ComplexF64, ℂ^4) atol = 1.0e-12
        if !isnothing(first_bases[2])
            @test first_bases[2]' * first_bases[2] ≈ id(ComplexF64, ℂ^4) atol = 1.0e-12
        end
    end
end

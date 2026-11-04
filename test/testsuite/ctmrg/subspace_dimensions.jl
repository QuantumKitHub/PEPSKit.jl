using Test
using Random
using LinearAlgebra
using MatrixAlgebraKit
using MatrixAlgebraKit: TruncatedAlgorithm
using TensorKit
using PEPSKit
using Adapt

"""Check integer budgets, new-charge discovery, and fixed target multiplicities."""
function ctmrg_subspace_dimensions_selection(AT)
    return @testset "Integer subspace allocation" begin
        W = U1Space(0 => 3, 1 => 3, 2 => 3)
        A, changed = (zeros(ComplexF64, W ← W) for _ in 1:2)
        for (hot, M) in ((2, A), (1, changed)), (c, b) in blocks(M)
            b[diagind(b)] .= c == U1Irrep(hot) ? [9, 8, 7] : c == U1Irrep(0) ? [6, 5, 4] : [3, 2, 1]
        end
        A, changed, trunc = adapt(AT, A), adapt(AT, changed), truncrank(4)
        @testset "$update" for update in (:independent, :alternating, :alternating_refined)
            Random.seed!(273)
            alg = SubspaceIterationProjector(; subspace = 6, trunc, update)
            first_step = PEPSKit._subspace_iteration_decomposition(A, id(W), alg, (; X = nothing, Y = nothing, trunc))
            @test fuse(domain(first_step.X)) == U1Space(0 => 3, 2 => 3)
            incoming = (; first_step.X, first_step.Y, trunc)
            result = PEPSKit._subspace_iteration_decomposition(changed, id(W), alg, (; incoming..., refresh = true))
            @test fuse(domain(result.X)) == U1Space(0 => 3, 1 => 3)
            @test result.X' * result.X ≈ id(ComplexF64, domain(result.X))
            u, s, v, = svd_trunc(changed, TruncatedAlgorithm(DefaultAlgorithm(), trunc))
            actual, expected = result.U * result.S * result.V, u * s * v
            @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10
            fixed = truncspace(U1Space(0 => 2, 1 => 2))
            result = PEPSKit._subspace_iteration_decomposition(A, id(W), alg, (; X = nothing, Y = nothing, trunc = fixed))
            @test fuse(domain(result.X)) == U1Space(0 => 2, 1 => 2, 2 => 2)
        end
        @testset "$S" for (S, charges) in ((SU2Irrep, (0, 1 // 2)), (A4Irrep, (0, 3)), (FibonacciAnyon, (:I, :τ)))
            W = Vect[S](charges[1] => 3, charges[2] => 2)
            M = zeros(ComplexF64, W ← W)
            for (c, b) in blocks(M)
                b[diagind(b)] .= (c == S(charges[1]) ? 3 : 10) .- (0:(size(b, 1) - 1))
            end
            M = adapt(AT, M)
            alg = SubspaceIterationProjector(; subspace = 6, trunc = truncrank(3))
            result = PEPSKit._subspace_iteration_decomposition(M, id(W), alg, (; X = nothing, Y = nothing, trunc = alg.trunc))
            _, s, _, = svd_trunc(M, TruncatedAlgorithm(DefaultAlgorithm(), alg.trunc))
            @test space(result.S, 1) == space(s, 1)
            @test dim(domain(result.X)) <= 6
        end
    end
end

"""Compare rank-changing sweeps with full SVD and check refresh and fixed-point policies."""
function ctmrg_subspace_dimensions_sweeps(AT)
    return @testset "Adaptive CTMRG sweeps" begin
        @testset "$flavor, $update, probe=$probe" for flavor in (SimultaneousCTMRG, SequentialCTMRG), update in (:independent, :alternating, :alternating_refined), probe in (nothing, 1)
            Random.seed!(274)
            P, V = U1Space(-1 => 1, 0 => 1, 1 => 1), U1Space(0 => 1, 1 => 1)
            ψ = adapt(AT, InfinitePEPS(P, V; unitcell = (2, 1)))
            network = InfiniteSquareNetwork(ψ)
            env = CTMRGEnv(ψ, U1Space(0 => 2))
            projector = SubspaceIterationProjector(; subspace = 4, trunc = truncrank(2), update, charge_refresh = 1, charge_probe = probe)
            alg = flavor(0.0, 2, 2, 0, projector)
            reference = flavor(; projector_alg = :FullInfiniteProjector, trunc = truncrank(2), verbosity = 0)
            cache = PEPSKit.initialize_projector_cache(network, env, alg)
            env_ref = env
            for _ in 1:2
                env, _, cache = PEPSKit.ctmrg_iteration(network, env, alg, cache)
                env_ref, = PEPSKit.ctmrg_iteration(network, env_ref, reference)
            end
            @test norm(ψ, env) ≈ norm(ψ, env_ref) rtol = 1.0e-7
            @test all(X -> dim(domain(X)) <= 4, cache.X)
            fixed = PEPSKit._set_fixed_truncation(alg).projector_alg
            @test fixed.trunc isa FixedSpaceTruncation && !PEPSKit._subspace_charge_refresh(cache, fixed)
        end
        periodic = SubspaceIterationProjector(; subspace = 24, charge_refresh = 10)
        cache = PEPSKit.SubspaceIterationCache(3, nothing, nothing)
        @test !PEPSKit._projector_convergence_ready(cache, periodic)
        @test PEPSKit._projector_convergence_ready(PEPSKit.@set(cache.iteration = 11), periodic)
        @test PEPSKit._subspace_charge_refresh(PEPSKit.@set(cache.iteration = 10), periodic)
    end
end

"""Discover absent charges incrementally and recover broad probing when requested."""
function ctmrg_subspace_dimensions_hybrid(AT)
    return @testset "Hybrid charge probes" begin
        @test isnothing(SubspaceIterationProjector(; subspace = 6).charge_probe)
        for probe in (0, -1, true, 1.5)
            @test_throws ArgumentError SubspaceIterationProjector(; subspace = 6, charge_probe = probe)
        end
        W = U1Space(0 => 3, 1 => 3, 2 => 3)
        A = zeros(ComplexF64, W ← W)
        for (c, b) in blocks(A)
            b[diagind(b)] .= c == U1Irrep(1) ? [100, 90, 80] : c == U1Irrep(0) ? [6, 5, 4] : [3, 2, 1]
        end
        A = adapt(AT, A)
        initial = U1Space(0 => 3, 2 => 3)
        @testset "$update" for update in (:independent, :alternating, :alternating_refined)
            alg = SubspaceIterationProjector(; subspace = 6, trunc = truncrank(4), charge_probe = 1, update)
            Random.seed!(277)
            X, Y = PEPSKit.rangefinder_initialize(A, id(W), initial, alg)
            incoming = (; X, Y, trunc = alg.trunc, refresh = true)
            K, refresh = PEPSKit._adaptive_subspace_space(A, id(W), alg, incoming)
            @test refresh && K == U1Space(0 => 3, 1 => 1, 2 => 3)
            broad = PEPSKit.@set alg.charge_probe = nothing
            large = PEPSKit.@set alg.charge_probe = 6
            @test first(PEPSKit._adaptive_subspace_space(A, id(W), broad, incoming)) == W
            @test first(PEPSKit._adaptive_subspace_space(A, id(W), large, incoming)) == W
            @test first(PEPSKit._adaptive_subspace_space(A, id(W), alg, (; incoming..., X = nothing))) == W
            @test first(PEPSKit._adaptive_subspace_space(A, id(W), alg, (; incoming..., X = real(X)))) == W
            local result
            for n in 1:3
                probe, = PEPSKit._adaptive_subspace_space(A, id(W), alg, incoming)
                expanded = PEPSKit._resize_subspace_basis(incoming.X, codomain(A), probe, alg, ComplexF64)
                @test expanded' * expanded ≈ id(ComplexF64, domain(expanded)) atol = 1.0e-12
                result = PEPSKit._subspace_iteration_decomposition(A, id(W), alg, incoming)
                selected = fuse(domain(result.X))
                @test dim(selected, U1Irrep(1)) == n
                @test dim(selected) <= 6
                incoming = (; result.X, result.Y, trunc = alg.trunc, refresh = true)
            end
            @test fuse(domain(result.X)) == U1Space(0 => 3, 1 => 3)
            u, s, v, = svd_trunc(A, TruncatedAlgorithm(DefaultAlgorithm(), alg.trunc))
            actual, expected = result.U * result.S * result.V, u * s * v
            @test actual / norm(actual) ≈ expected / norm(expected) atol = 1.0e-10
            fixed = truncspace(U1Space(0 => 2, 1 => 2))
            @test first(PEPSKit._adaptive_subspace_space(A, id(W), alg, (; incoming..., X, Y, trunc = fixed))) == W
        end
        W = SU2Space(0 => 6, 1 // 2 => 4)
        alg = SubspaceIterationProjector(; subspace = 6, charge_probe = 1)
        current = SU2Space(0 => 2, 1 // 2 => 1)
        @test PEPSKit._subspace_probe_space(W, alg, truncrank(3), current) == SU2Space(0 => 3, 1 // 2 => 2)
    end
end

"""Run focused checks for automatically allocated integer trial dimensions."""
function ctmrg_subspace_dimensions(AT)
    return @testset "Subspace dimensions ($AT)" begin
        ctmrg_subspace_dimensions_hybrid(AT)
        ctmrg_subspace_dimensions_selection(AT)
        ctmrg_subspace_dimensions_sweeps(AT)
    end
end

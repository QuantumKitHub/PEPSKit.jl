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
        @testset "$flavor, $update" for flavor in (SimultaneousCTMRG, SequentialCTMRG), update in (:independent, :alternating, :alternating_refined)
            Random.seed!(274)
            P, V = U1Space(-1 => 1, 0 => 1, 1 => 1), U1Space(0 => 1, 1 => 1)
            ψ = adapt(AT, InfinitePEPS(P, V; unitcell = (2, 1)))
            network = InfiniteSquareNetwork(ψ)
            env = CTMRGEnv(ψ, U1Space(0 => 2))
            projector = SubspaceIterationProjector(; subspace = 4, trunc = truncrank(2), update, charge_refresh = 1)
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

"""Run focused checks for automatically allocated integer trial dimensions."""
function ctmrg_subspace_dimensions(AT)
    return @testset "Subspace dimensions ($AT)" begin
        ctmrg_subspace_dimensions_selection(AT)
        ctmrg_subspace_dimensions_sweeps(AT)
    end
end

using Test, PEPSKit, TensorKit, MatrixAlgebraKit, LinearAlgebra, Random, Zygote
using ChainRulesCore: ignore_derivatives
BLAS.set_num_threads(1)
PEPSKit.set_scheduler!(:serial)

"""Estimate a directional derivative using extrapolated central differences."""
function subspace_update_difference(f, x, direction; h = 1.0e-6)
    central(δ) = (f(x + δ * direction) - f(x - δ * direction)) / (2δ)
    return (4central(h) - central(2h)) / 3
end

@testset "Unified two-step matrix pullbacks" begin
    for T in (Float64, ComplexF64), update in (:independent, :alternating, :alternating_refined), q in (0, 1), rotate in (false, true)
        Random.seed!(84264)
        W, K, χ = ℂ^7, ℂ^4, ℂ^2
        A, direction, target = randn(T, W ← W), randn(T, W ← W), randn(T, W ← W)
        direction /= norm(direction)
        alg = SubspaceIterationProjector(;
            subspace = K, update, n_subspace_iter = q,
            rotate_basis = rotate, subspace_tol = 0
        )
        X, Y = PEPSKit.rangefinder_initialize(A, id(T, W), K, alg)
        initial = (; X, Y, trunc = truncspace(χ), recycle = false)
        function loss(t)
            step = PEPSKit._subspace_iteration_decomposition(t, id(T, W), alg, initial)
            result = PEPSKit._subspace_iteration_decomposition(
                t, id(T, W), alg,
                (; step.X, step.Y, initial.trunc, recycle = false)
            )
            return real(dot(target, result.U * result.S * result.V))
        end
        gradient = only(Zygote.gradient(loss, A))
        @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
        target_U = randn(T, W ← χ)
        phase_loss(t) = real(dot(target_U, PEPSKit._subspace_iteration_decomposition(t, id(T, W), alg, initial).U))
        phase_gradient = only(Zygote.gradient(phase_loss, A))
        @test real(dot(phase_gradient, direction)) ≈ subspace_update_difference(phase_loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
    end
end

@testset "Unified symmetric recycled pullbacks" begin
    F = FermionParity ⊠ U1Irrep
    spaces = (
        (SU2Space(0 => 3, 1 // 2 => 7), SU2Space(0 => 3, 1 // 2 => 4), SU2Space(0 => 2, 1 // 2 => 2)),
        (Vect[F]((0, 0) => 3, (1, 1) => 7), Vect[F]((0, 0) => 3, (1, 1) => 4), Vect[F]((0, 0) => 2, (1, 1) => 2)),
    )
    for (W, K, χ) in spaces, update in (:independent, :alternating, :alternating_refined)
        Random.seed!(84265)
        A, direction, target = randn(ComplexF64, W ← W), randn(ComplexF64, W ← W), randn(ComplexF64, W ← W)
        direction /= norm(direction)
        alg = SubspaceIterationProjector(; subspace = K, update, n_subspace_iter = 1, subspace_tol = 0)
        X, Y = PEPSKit.rangefinder_initialize(A, id(ComplexF64, W), K, alg)
        initial = (; X, Y, trunc = truncspace(χ), recycle = false)
        function loss(t)
            step = PEPSKit._subspace_iteration_decomposition(t, id(ComplexF64, W), alg, initial)
            result = PEPSKit._subspace_iteration_decomposition(
                t, id(ComplexF64, W), alg,
                (; step.X, step.Y, initial.trunc, recycle = false)
            )
            return real(dot(target, result.U * result.S * result.V))
        end
        gradient = only(Zygote.gradient(loss, A))
        @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
    end
end

"""Contract two production sweeps with the first sweep's cache retained in the graph."""
function subspace_two_sweep_energy(ψ, env, alg, cache, H, frozen)
    network = InfiniteSquareNetwork(ψ)
    env1, _, cache1 = PEPSKit.ctmrg_iteration(network, env, alg, cache)
    cache1 = PEPSKit.SubspaceIterationCache(cache1.iteration, frozen, cache1.X, cache1.Y, cache1.U, cache1.V)
    env2, info, _ = PEPSKit.ctmrg_iteration(network, env1, alg, cache1)
    return cost_function(ψ, env2, H), info.contraction_metrics.recycled
end

@testset "Unified active and frozen production pullbacks" begin
    settings = ((:independent, 0), (:alternating_refined, 0), (:alternating, 0), (:alternating, 1))
    for (update, q) in settings, flavor in (SimultaneousCTMRG, SequentialCTMRG), rotate in (false, true), frozen in (false, true)
        Random.seed!(84266)
        ψ = InfinitePEPS(ℂ^2, ℂ^2); ψ /= norm(ψ)
        direction = InfinitePEPS(ℂ^2, ℂ^2); direction /= norm(direction)
        env = CTMRGEnv(ψ, ℂ^3)
        H = heisenberg_XYZ(InfiniteSquare())
        projector = SubspaceIterationProjector(;
            subspace = ℂ^4, update, n_subspace_iter = q,
            rotate_basis = rotate, subspace_tol = frozen ? 1.0e-3 : 0.0
        )
        alg = flavor(0.0, 2, 2, 0, projector)
        cache = PEPSKit.initialize_projector_cache(InfiniteSquareNetwork(ψ), env, alg)
        loss(x) = first(subspace_two_sweep_energy(x, env, alg, cache, H, frozen))
        gradient = only(Zygote.gradient(loss, ψ))
        _, applied = subspace_two_sweep_energy(ψ, env, alg, cache, H, frozen)
        @test applied == frozen
        @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, ψ, direction; h = 1.0e-7) rtol = 3.0e-5 atol = 3.0e-7
    end
end

@testset "Unified exact-rank fixed-point gradients" begin
    for update in (:independent, :alternating, :alternating_refined), flavor in (SimultaneousCTMRG, SequentialCTMRG)
        Random.seed!(84267)
        ψ = InfinitePEPS(ℂ^2, ℂ^2); ψ /= norm(ψ)
        direction = InfinitePEPS(ℂ^2, ℂ^2); direction /= norm(direction)
        env = CTMRGEnv(ψ, ℂ^2)
        H = heisenberg_XYZ(InfiniteSquare())
        alg = flavor(;
            tol = 1.0e-11, miniter = 20, maxiter = 200, verbosity = 0,
            projector_alg = (;
                alg = :SubspaceIterationProjector, subspace = ℂ^8, update,
                n_subspace_iter = update == :alternating ? 1 : 0, subspace_tol = 0,
            )
        )
        mode = FixedPointGradient(; solver_alg = (; alg = :GMRES, tol = 1.0e-10, maxiter = 100))
        function loss(x)
            ignore_derivatives(() -> Random.seed!(84268))
            boundary, = PEPSKit.hook_pullback(leading_boundary, env, x, alg; alg_rrule = mode)
            return cost_function(x, boundary, H)
        end
        _, info = leading_boundary(env, ψ, alg)
        @test info.converged
        gradient = only(Zygote.gradient(loss, ψ))
        @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, ψ, direction; h = 1.0e-4) rtol = 1.0e-4 atol = 1.0e-7
    end
end

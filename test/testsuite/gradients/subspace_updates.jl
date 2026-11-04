using Test
using Random
using LinearAlgebra
using MatrixAlgebraKit
using TensorKit
using Zygote
using PEPSKit
using Adapt
using ChainRulesCore: ignore_derivatives

"""Estimate a directional derivative using extrapolated central differences."""
function subspace_update_difference(f, x, direction; h = 1.0e-6)
    central(δ) = (f(x + δ * direction) - f(x - δ * direction)) / (2δ)
    return (4central(h) - central(2h)) / 3
end

"""Contract two production sweeps with the first sweep's cache retained in the graph."""
function subspace_two_sweep_energy(ψ, env, alg, cache, H)
    network = InfiniteSquareNetwork(ψ)
    env1, _, cache1 = PEPSKit.ctmrg_iteration(network, env, alg, cache)
    env2, info, _ = PEPSKit.ctmrg_iteration(network, env1, alg, cache1)
    return cost_function(ψ, env2, H)
end

"""Check matrix for the subspace iteration projector."""
function gradients_subspace_updates_matrix(AT)
    return @testset "Two-step matrix pullbacks" begin
        @testset "$T, $update, q=$q, rotate=$rotate" for T in (Float64, ComplexF64), update in (:independent, :alternating, :alternating_refined), q in (0, 1), rotate in (false, true)
            Random.seed!(84264)
            W, K, χ = ℂ^7, ℂ^4, ℂ^2
            A, direction, target = adapt(AT, randn(T, W ← W)), adapt(AT, randn(T, W ← W)), adapt(AT, randn(T, W ← W))
            direction /= norm(direction)
            alg = SubspaceIterationProjector(;
                subspace = K, update, n_subspace_iter = q,
                rotate_basis = rotate,
            )
            X, Y = PEPSKit.rangefinder_initialize(A, id(T, W), K, alg)
            initial = (; X, Y, trunc = truncspace(χ))
            function loss(t)
                step = PEPSKit._subspace_iteration_decomposition(t, id(T, W), alg, initial)
                result = PEPSKit._subspace_iteration_decomposition(
                    t, id(T, W), alg,
                    (; step.X, step.Y, initial.trunc)
                )
                return real(dot(target, result.U * result.S * result.V))
            end
            gradient = only(Zygote.gradient(loss, A))
            @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
            target_U = adapt(AT, randn(T, W ← χ))
            phase_loss(t) = real(dot(target_U, PEPSKit._subspace_iteration_decomposition(t, id(T, W), alg, initial).U))
            phase_gradient = only(Zygote.gradient(phase_loss, A))
            @test real(dot(phase_gradient, direction)) ≈ subspace_update_difference(phase_loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
        end
    end
end

"""Check symmetric for the subspace iteration projector."""
function gradients_subspace_updates_symmetric(AT)
    return @testset "Symmetric recycled pullbacks" begin
        F = FermionParity ⊠ U1Irrep
        spaces = (
            (SU2Space(0 => 3, 1 // 2 => 7), SU2Space(0 => 3, 1 // 2 => 4), SU2Space(0 => 2, 1 // 2 => 2)),
            (Vect[F]((0, 0) => 3, (1, 1) => 7), Vect[F]((0, 0) => 3, (1, 1) => 4), Vect[F]((0, 0) => 2, (1, 1) => 2)),
        )
        @testset "$W, $update" for (W, K, χ) in spaces, update in (:independent, :alternating, :alternating_refined)
            Random.seed!(84265)
            A, direction, target = adapt(AT, randn(ComplexF64, W ← W)), adapt(AT, randn(ComplexF64, W ← W)), adapt(AT, randn(ComplexF64, W ← W))
            direction /= norm(direction)
            alg = SubspaceIterationProjector(; subspace = K, update, n_subspace_iter = 1)
            X, Y = PEPSKit.rangefinder_initialize(A, id(ComplexF64, W), K, alg)
            initial = (; X, Y, trunc = truncspace(χ))
            function loss(t)
                step = PEPSKit._subspace_iteration_decomposition(t, id(ComplexF64, W), alg, initial)
                result = PEPSKit._subspace_iteration_decomposition(
                    t, id(ComplexF64, W), alg,
                    (; step.X, step.Y, initial.trunc)
                )
                return real(dot(target, result.U * result.S * result.V))
            end
            gradient = only(Zygote.gradient(loss, A))
            @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, A, direction) rtol = 3.0e-5 atol = 3.0e-7
        end
    end
end

"""Check differentiation through consecutive production sweeps."""
function gradients_subspace_updates_sweeps(AT)
    return @testset "Two-sweep production pullbacks" begin
        settings = ((:independent, 0), (:alternating_refined, 0), (:alternating, 0), (:alternating, 1))
        @testset "$update, q=$q, $flavor, rotate=$rotate" for (update, q) in settings, flavor in (SimultaneousCTMRG, SequentialCTMRG), rotate in (false, true)
            Random.seed!(84266)
            ψ = adapt(AT, InfinitePEPS(ℂ^2, ℂ^2))
            ψ /= norm(ψ)
            direction = adapt(AT, InfinitePEPS(ℂ^2, ℂ^2))
            direction /= norm(direction)
            env = CTMRGEnv(ψ, ℂ^3)
            H = adapt(AT, heisenberg_XYZ(InfiniteSquare()))
            projector = SubspaceIterationProjector(;
                subspace = ℂ^4, update, n_subspace_iter = q,
                rotate_basis = rotate
            )
            alg = flavor(0.0, 2, 2, 0, projector)
            cache = PEPSKit.initialize_projector_cache(InfiniteSquareNetwork(ψ), env, alg)
            loss(x) = subspace_two_sweep_energy(x, env, alg, cache, H)
            gradient = only(Zygote.gradient(loss, ψ))
            @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, ψ, direction; h = 1.0e-7) rtol = 3.0e-5 atol = 3.0e-7
        end
    end
end

"""Check fixed point for the subspace iteration projector."""
function gradients_subspace_updates_fixed_point(AT)
    references = Dict()
    cases = [(update, flavor, unitcell, false) for update in (:independent, :alternating, :alternating_refined) for flavor in (SimultaneousCTMRG, SequentialCTMRG) for unitcell in ((1, 1), (2, 1))]
    # Complete-range regressions for the two updates with drifting bond gauges.
    append!(cases, ((update, SimultaneousCTMRG, (1, 1), true) for update in (:independent, :alternating)))
    return @testset "Subspace fixed-point gradients" begin
        @testset "$update, $flavor, $unitcell, complete=$complete" for (update, flavor, unitcell, complete) in cases
            Random.seed!(complete ? 5 : 84267)
            ψ = adapt(AT, InfinitePEPS(ℂ^2, ℂ^2; unitcell))
            ψ /= norm(ψ)
            direction = adapt(AT, InfinitePEPS(ℂ^2, ℂ^2; unitcell))
            direction /= norm(direction)
            env = CTMRGEnv(ψ, ℂ^4)
            H = adapt(AT, heisenberg_XYZ(InfiniteSquare(unitcell...)))
            alg = flavor(;
                tol = 1.0e-11, miniter = 20, maxiter = 500, verbosity = 0,
                projector_alg = (;
                    alg = :SubspaceIterationProjector, subspace = ℂ^(complete ? 16 : 6), update,
                    n_subspace_iter = update == :alternating ? 1 : 0,
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
            reference = get!(references, (flavor, unitcell, complete)) do
                full = flavor(; tol = 1.0e-12, maxiter = 500, verbosity = 0, projector_alg = :FullInfiniteProjector)
                f(x) = cost_function(x, first(PEPSKit.hook_pullback(leading_boundary, env, x, full; alg_rrule = mode)), H)
                only(Zygote.gradient(f, ψ))
            end
            @test norm(gradient - reference) / norm(reference) < 1.0e-7
            if !complete && update == :independent && flavor == SimultaneousCTMRG && unitcell == (1, 1)
                for solver in (:Arnoldi, :BiCGStab, :GeomSum, :ManualIter)
                    other = FixedPointGradient(; solver_alg = (; alg = solver, tol = 1.0e-10, maxiter = 200))
                    f(x) = cost_function(x, first(PEPSKit.hook_pullback(leading_boundary, env, x, alg; alg_rrule = other)), H)
                    @test norm(only(Zygote.gradient(f, ψ)) - reference) / norm(reference) < 1.0e-7
                end
            end
            if complete && update == :independent
                function unrolled(x)
                    ignore_derivatives(() -> Random.seed!(84268))
                    return cost_function(x, first(leading_boundary(env, x, alg)), H)
                end
                @test norm(only(Zygote.gradient(unrolled, ψ)) - reference) / norm(reference) < 1.0e-7
            end
            @test real(dot(gradient, direction)) ≈ subspace_update_difference(loss, ψ, direction; h = 1.0e-4) rtol = 1.0e-7 atol = 1.0e-9
        end
    end
end

"""Check subspace update gradients for the requested storage type."""
function gradients_subspace_updates(AT)
    return @testset "Subspace update gradients ($AT)" begin
        gradients_subspace_updates_matrix(AT)
        gradients_subspace_updates_symmetric(AT)
        gradients_subspace_updates_sweeps(AT)
        gradients_subspace_updates_fixed_point(AT)
    end
end

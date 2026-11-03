using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using Zygote
using Adapt
using ChainRulesCore: ignore_derivatives

"""Check cache feedback through active and charge-refresh sweeps."""
function gradients_subspace_dimensions(AT)
    return @testset "Adaptive subspace gradients ($AT)" begin
        @testset "$update, q=$q, $mode, probe=$probe" for update in (:independent, :alternating, :alternating_refined), q in (0, 1), mode in (:active, :refresh), probe in (nothing, 1)
            Random.seed!(275)
            W = U1Space(0 => 6, 1 => 6, 2 => 6)
            A, direction, target = (adapt(AT, randn(ComplexF64, W ← W)) for _ in 1:3)
            direction /= norm(direction)
            alg = SubspaceIterationProjector(; subspace = 8, trunc = truncrank(4), charge_probe = probe, update, n_subspace_iter = q, rotate_basis = q == 0)
            X, Y = PEPSKit.rangefinder_initialize(A, id(W), W, alg)
            function loss(t)
                ignore_derivatives(() -> Random.seed!(276))
                first_step = PEPSKit._subspace_iteration_decomposition(t, id(W), alg, (; X, Y, trunc = alg.trunc))
                result = PEPSKit._subspace_iteration_decomposition(t, id(W), alg, (; first_step.X, first_step.Y, trunc = alg.trunc, refresh = mode == :refresh))
                return real(dot(target, result.U * result.S * result.V))
            end
            gradient = only(Zygote.gradient(loss, A))
            h = 1.0e-6
            central(δ) = (loss(A + δ * direction) - loss(A - δ * direction)) / (2δ)
            finite = (4central(h) - central(2h)) / 3
            @test real(dot(gradient, direction)) ≈ finite rtol = 3.0e-5 atol = 3.0e-7
        end
    end
end

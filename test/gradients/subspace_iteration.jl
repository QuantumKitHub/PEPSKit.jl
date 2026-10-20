using Test
using Random
using LinearAlgebra
using MatrixAlgebraKit
using TensorKit
using Zygote
using PEPSKit
using ChainRulesCore: ignore_derivatives

"""Check a directional derivative with Richardson-extrapolated central differences."""
function si_directional_difference(f, x, direction; h = 1.0e-4)
    central(δ) = (f(x + δ * direction) - f(x - δ * direction)) / (2δ)
    return (4central(h) - central(2h)) / 3
end

"""Build a Hermitian nearest-neighbor operator compatible with the physical symmetry."""
function si_test_operator(T, P)
    h = randn(T, P ⊗ P ← P ⊗ P)
    h = h + h'
    h /= norm(h)
    return LocalOperator(fill(P, 1, 1), ((1, 1), (1, 2)) => h, ((1, 1), (2, 1)) => h)
end

"""Supply small PEPS spaces spanning trivial, Abelian, non-Abelian, and fermionic sectors."""
function si_energy_test_spaces()
    F = Vect[FermionParity ⊠ U1Irrep]((0, 0) => 1, (1, 1) => 1, (1, -1) => 1)
    A = Vect[A4Irrep](3 => 1)
    return (
        (ℂ^2, ℂ^2, ℂ^2),
        (U1Space(-1 => 1, 0 => 1, 1 => 1), U1Space(0 => 1, 1 => 1), U1Space(-1 => 1, 0 => 1, 1 => 1)),
        (SU2Space(0 => 1, 1 => 1), SU2Space(1 // 2 => 1), SU2Space(0 => 1, 1 => 1)),
        (F, F, F),
        (A, A, Vect[A4Irrep](0 => 1, 3 => 1)),
    )
end

@testset "SI right orthogonalization pullback" begin
    spaces = (
        (ℂ^3, ℂ^6),
        (U1Space(-1 // 2 => 2, 1 // 2 => 3), U1Space(-1 // 2 => 4, 1 // 2 => 5)),
        (SU2Space(0 => 2, 1 // 2 => 3), SU2Space(0 => 4, 1 // 2 => 5)),
        (
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 3),
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 4, (1, 1) => 5),
        ),
    )
    for T in (Float64, ComplexF64), (K, W) in spaces
        Random.seed!(731)
        A, direction = randn(T, K ← W), randn(T, K ← W)
        target = randn(T, W ← W)
        loss(t) = real(dot(target, last(right_orth(t, OrthAdjoint()))' * last(right_orth(t, OrthAdjoint()))))
        g = only(Zygote.gradient(loss, A))
        @test real(dot(g, direction)) ≈ si_directional_difference(loss, A, direction) rtol = 1.0e-6 atol = 1.0e-8
    end
end

@testset "SI recycled-basis pullback" begin
    spaces = (
        (ℂ^6, ℂ^4, ℂ^2),
        (
            U1Space(-1 // 2 => 4, 1 // 2 => 5, 3 // 2 => 3),
            U1Space(-1 // 2 => 3, 1 // 2 => 3, 3 // 2 => 2),
            U1Space(-1 // 2 => 2, 1 // 2 => 2, 3 // 2 => 1),
        ),
        (
            SU2Space(0 => 4, 1 // 2 => 5, 1 => 3),
            SU2Space(0 => 3, 1 // 2 => 3, 1 => 2), SU2Space(0 => 2, 1 // 2 => 2, 1 => 1),
        ),
        (
            Vect[A4Irrep](0 => 4, 1 => 5, 3 => 3),
            Vect[A4Irrep](0 => 3, 1 => 3, 3 => 2), Vect[A4Irrep](0 => 2, 1 => 2, 3 => 1),
        ),
        (
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 4, (1, 1) => 5, (1, -1) => 3),
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 3, (1, 1) => 3, (1, -1) => 2),
            Vect[FermionParity ⊠ U1Irrep]((0, 0) => 2, (1, 1) => 2, (1, -1) => 1),
        ),
        (
            Vect[FibonacciAnyon](:I => 4, :τ => 5),
            Vect[FibonacciAnyon](:I => 3, :τ => 3), Vect[FibonacciAnyon](:I => 2, :τ => 2),
        ),
    )
    for (W, K, χ) in spaces
        Random.seed!(732)
        L, R = randn(ComplexF64, W ← W), randn(ComplexF64, W ← W)
        direction, target = randn(ComplexF64, space(L)), randn(ComplexF64, space(L))
        alg = SubspaceIterationProjector(; subspace = K)
        X, Y = PEPSKit.rangefinder_initialize(L, R, alg.subspace, alg)
        inputs = (; X, Y, trunc = truncspace(χ), recycle = false)
        function loss(t)
            first_step = PEPSKit._subspace_iteration_decomposition(t, R, alg, inputs)
            next_inputs = (; first_step.X, first_step.Y, inputs.trunc, recycle = false)
            result = PEPSKit._subspace_iteration_decomposition(t, R, alg, next_inputs)
            return real(dot(target, result.U * result.S * result.V))
        end
        g = only(Zygote.gradient(loss, L))
        @test real(dot(g, direction)) ≈ si_directional_difference(loss, L, direction) rtol = 1.0e-5 atol = 1.0e-7
        # A phase-sensitive loss checks the gauge convention separately from reconstruction.
        target_U = randn(ComplexF64, codomain(L) ← χ)
        phase_loss(t) = real(dot(target_U, PEPSKit._subspace_iteration_decomposition(t, R, alg, inputs).U))
        gphase = only(Zygote.gradient(phase_loss, L))
        @test real(dot(gphase, direction)) ≈ si_directional_difference(phase_loss, L, direction) rtol = 1.0e-5 atol = 1.0e-7
    end
end

@testset "SI energy backpropagation" begin
    for T in (Float64, ComplexF64), flavor in (SimultaneousCTMRG, SequentialCTMRG), (P, V, χ) in si_energy_test_spaces()
        Random.seed!(733)
        K = infimum(fuse(χ ⊗ V ⊗ V'), χ ⊕ χ)
        psi = InfinitePEPS(randn, T, P, V)
        psi = psi / norm(psi)
        direction = InfinitePEPS(randn, T, P, V)
        direction = direction / norm(direction)
        env = CTMRGEnv(randn, T, psi, χ)
        H = si_test_operator(T, P)
        alg = flavor(;
            miniter = 2, maxiter = 2, tol = 0.0, verbosity = 0,
            projector_alg = (; alg = :SubspaceIterationProjector, subspace = K, subspace_tol = 0.0),
        )
        function loss(ψ)
            ignore_derivatives(() -> Random.seed!(734))
            return cost_function(ψ, first(leading_boundary(env, ψ, alg)), H)
        end
        g = only(Zygote.gradient(loss, psi))
        @test real(dot(g, direction)) ≈ si_directional_difference(loss, psi, direction; h = 1.0e-6) rtol = 1.0e-5 atol = 1.0e-7
    end
end

@testset "SI fixed-point gradient" begin
    for flavor in (SimultaneousCTMRG, SequentialCTMRG), (P, V, χ) in si_energy_test_spaces()
        Random.seed!(738)
        K = fuse(χ ⊗ V ⊗ V')
        psi = InfinitePEPS(P, V)
        psi = psi / norm(psi)
        direction = InfinitePEPS(P, V)
        direction = direction / norm(direction)
        env = CTMRGEnv(psi, χ)
        H = si_test_operator(ComplexF64, P)
        alg = flavor(;
            miniter = 20, maxiter = 200, tol = 1.0e-12, verbosity = 0,
            projector_alg = (; alg = :SubspaceIterationProjector, subspace = K, subspace_tol = 0.0),
        )
        gradmode = FixedPointGradient(; solver_alg = (; alg = :GMRES, tol = 1.0e-10, maxiter = 100))
        function loss(ψ)
            ignore_derivatives(() -> Random.seed!(736))
            env′, = PEPSKit.hook_pullback(leading_boundary, env, ψ, alg; alg_rrule = gradmode)
            return cost_function(ψ, env′, H)
        end
        Random.seed!(736)
        _, info = leading_boundary(env, psi, alg)
        @test info.converged
        g = only(Zygote.gradient(loss, psi))
        @test real(dot(g, direction)) ≈ si_directional_difference(loss, psi, direction) rtol = 1.0e-4 atol = 1.0e-7
    end
end

using Test, Random, LinearAlgebra, TensorKit, PEPSKit, Zygote, Adapt
using ChainRulesCore: ignore_derivatives
using MatrixAlgebraKit: TruncatedAlgorithm

function si_directional_difference(f, x, direction; h = 3.0e-5)
    central(δ) = (f(x + δ * direction) - f(x - δ * direction)) / (2δ)
    return (4central(h) - central(2h)) / 3
end

function gradients_subspace_iteration(AT)
    return @testset "SI decomposition gradients ($AT)" begin
        @testset "SVD pullback, $sector, $reverse" for sector in (Trivial, SU2Irrep, FermionParity ⊠ U1Irrep), reverse in (:GMRES, :Arnoldi, :TruncPullback)
            Random.seed!(84264)
            V = sector == Trivial ? ℂ^2 : sector == SU2Irrep ? SU2Space(0 => 1, 1 // 2 => 1) : Vect[sector]((0, 0) => 1, (1, 1) => 1)
            W = V ⊗ V'
            K = fuse(W)
            χ = typeof(K)(c => 1 for c in sectors(K))
            A, B, direction, target = (adapt(AT, randn(ComplexF64, W ← W)) for _ in 1:4)
            direction /= norm(direction)
            target_U = adapt(AT, randn(ComplexF64, W ← χ))
            svd = SVDAdjoint(; fwd_alg = SubspaceIteration(; subspace = K), rrule_alg = (; alg = reverse))
            svd = PEPSKit.@set svd.fwd_alg = TruncatedAlgorithm(svd.fwd_alg, truncspace(χ))
            dense = SVDAdjoint(; rrule_alg = (; alg = :TruncPullback))
            dense = PEPSKit.@set dense.fwd_alg = TruncatedAlgorithm(dense.fwd_alg, truncspace(χ))
            function loss(t, algorithm, factorized)
                M = factorized ? PEPSKit._factorized_environment(t, B) : PEPSKit.half_infinite_environment(t, B)
                u, s, v, = PEPSKit._projector_svd(M, algorithm)
                if !factorized
                    Q = ignore_derivatives(() -> isometry(ComplexF64, W ← K))
                    u, v = PEPSKit._subspace_svd_phase_correction(u, v, Q' * u)
                end
                return real(dot(target, normalize(u * s * v))) + 0.1real(dot(target_U, u))
            end
            f(t) = loss(t, svd, true)
            ref(t) = loss(t, dense, false)
            value, (gradient,) = Zygote.withgradient(f, A)
            reference = only(Zygote.gradient(ref, A))
            @test value ≈ f(A) atol = 1.0e-10
            @test f(A) ≈ ref(A) atol = 1.0e-10
            @test norm(gradient - reference) / norm(reference) < 1.0e-7
            @test real(dot(gradient, direction)) ≈ si_directional_difference(f, A, direction) rtol = 1.0e-6 atol = 1.0e-8
        end
        @testset "$projector, $flavor, fermionic=$fermionic" for projector in (:FullInfiniteProjector, :HalfInfiniteProjector), flavor in (SimultaneousCTMRG, SequentialCTMRG), fermionic in (false, true)
            Random.seed!(84267)
            V = fermionic ? Vect[FermionParity](0 => 1, 1 => 1) : ℂ^2
            χ = fermionic ? Vect[FermionParity](0 => 2, 1 => 2) : ℂ^4
            # Compressed Heisenberg trials; integer fermionic budgets also exercise allocation in AD.
            K = fermionic ? dim(fuse(χ ⊗ V ⊗ V')) : ℂ^(projector == :FullInfiniteProjector ? 8 : 10)
            ψ = adapt(AT, InfinitePEPS(V, V))
            ψ /= norm(ψ)
            env = CTMRGEnv(ψ, χ)
            direction = adapt(AT, InfinitePEPS(V, V)); direction /= norm(direction)
            H = adapt(AT, fermionic ? pwave_superconductor(InfiniteSquare()) : heisenberg_XYZ(InfiniteSquare()))
            alg = flavor(;
                tol = 1.0e-11, miniter = 20, maxiter = 500, verbosity = 0, projector_alg = projector,
                decomposition_alg = (; alg = :SubspaceIteration, subspace = K)
            )
            full = flavor(; tol = 1.0e-12, miniter = 20, maxiter = 500, verbosity = 0, projector_alg = projector)
            mode = FixedPointGradient(; solver_alg = (; alg = :GMRES, tol = 1.0e-10, maxiter = 200))
            function energy(p, boundary_alg)
                ignore_derivatives(() -> Random.seed!(84268))
                boundary, run_info = PEPSKit.hook_pullback(leading_boundary, env, p, boundary_alg; alg_rrule = mode)
                @assert run_info.converged
                return cost_function(p, boundary, H)
            end
            f(p) = energy(p, alg)
            ref(p) = energy(p, full)
            Random.seed!(84268)
            boundary, info = leading_boundary(env, ψ, alg)
            @test info.converged
            @test f(ψ) ≈ ref(ψ) atol = 1.0e-9
            gradient, reference = only(Zygote.gradient(f, ψ)), only(Zygote.gradient(ref, ψ))
            @test norm(gradient - reference) / norm(reference) < 1.0e-7
            @test real(dot(gradient, direction)) ≈ si_directional_difference(f, ψ, direction) rtol = 1.0e-6 atol = 1.0e-8
        end
    end
end

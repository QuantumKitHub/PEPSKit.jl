using Test, LinearAlgebra, KrylovKit, PEPSKit
using Random, TensorKit, Zygote, Adapt

"""Check split-specific solver defaults across both optimizer entry points and explicit overrides."""
function ctmrg_split_gradient_defaults(AT)
    return @testset "Split fixed-point gradient defaults ($AT)" begin
        Random.seed!(73418)
        psi = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2)))
        ordinary_env = CTMRGEnv(psi, ComplexSpace(2))
        split_env = SplitCTMRGEnv(ordinary_env)
        constructors = (
            kwargs -> PEPSOptimize(; boundary_alg = SplitCTMRG(), kwargs...),
            kwargs -> PEPSKit.select_algorithm(PEPSKit.fixedpoint, split_env; kwargs...),
        )
        for construct in constructors
            solver = PEPSKit.parent_alg(construct((;)).gradient_alg).solver_alg
            @test solver isa GMRES
            partial = (; solver_alg = (; tol = 1.0e-7, krylovdim = 17))
            solver = PEPSKit.parent_alg(construct((; gradient_alg = partial)).gradient_alg).solver_alg
            @test solver isa GMRES
            @test solver.tol == 1.0e-7
            @test solver.krylovdim == 17
            explicit = (; solver_alg = (; alg = :Arnoldi))
            solver = PEPSKit.parent_alg(construct((; gradient_alg = explicit)).gradient_alg).solver_alg
            @test solver isa Arnoldi
            requested = Arnoldi(; tol = 1.0e-7)
            explicit = (; solver_alg = requested)
            @test PEPSKit.parent_alg(construct((; gradient_alg = explicit)).gradient_alg).solver_alg === requested
            explicit = FixedPointGradient(; solver_alg = Arnoldi(; tol = 1.0e-7))
            @test PEPSKit.parent_alg(construct((; gradient_alg = explicit)).gradient_alg) === explicit
            @test construct((; gradient_alg = nothing)).gradient_alg === nothing
        end
        wrapped = PEPSKit.DynamicTol(SplitCTMRG(), 1.0e-12, 1.0e-4, 1.0e-3)
        @test PEPSKit.parent_alg(PEPSOptimize(; boundary_alg = wrapped).gradient_alg).solver_alg isa GMRES
        @test PEPSOptimize().gradient_alg.solver_alg isa Arnoldi
        ordinary = PEPSKit.select_algorithm(PEPSKit.fixedpoint, ordinary_env)
        @test PEPSKit.parent_alg(ordinary.gradient_alg).solver_alg isa Arnoldi
        @test FixedPointGradient().solver_alg isa Arnoldi
    end
end

"""Preserve forward convergence diagnostics when attaching a fixed-point pullback."""
function ctmrg_fixedpoint_forward_info(AT)
    return @testset "Fixed-point forward convergence information ($AT)" begin
        Random.seed!(73419)
        psi = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2)))
        for alg in (SequentialCTMRG(; tol = 0.0, maxiter = 1, verbosity = 0), SplitCTMRG(; tol = 0.0, maxiter = 1, verbosity = 0))
            initial = CTMRGEnv(psi, ComplexSpace(2))
            env = alg isa SplitCTMRG ? SplitCTMRGEnv(initial) : initial
            expected, expected_info = leading_boundary(env, psi, alg)
            gradalg = FixedPointGradient(; solver_alg = (; alg = :GMRES))
            (actual, info), _ = Zygote.pullback(psi) do A
                return PEPSKit.hook_pullback(leading_boundary, env, A, alg; alg_rrule = gradalg)
            end
            @test norm(actual - expected) ≤ 1.0e-12 * max(norm(expected), 1)
            @test !info.converged
            @test info.convergence_error ≈ expected_info.convergence_error
            @test info.contraction_metrics == expected_info.contraction_metrics
        end
    end
end

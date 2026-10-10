using Test
using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit
using Adapt

include("gradient_utility.jl")

## Test CTMRG gradients
# -------------------------------------------
# Every gradient algorithm is compared with a converged fixed-point reference gradient, which is
# checked against finite differences. Each option is covered once, not every combination.

ctmrg_tol = 1.0e-12
solver_tol = 1.0e-10
ctmrg_verbosity = 0
rtol = 1.0e-7 # gradient errors and finite differences
naive_miniter = 30 # CTMRG iterations naive AD differentiates through

models = [
    (
        name = "Heisenberg", H = heisenberg_XYZ(InfiniteSquare()),
        Pspace = ComplexSpace(2), Vspace = ComplexSpace(2), Espace = ComplexSpace(6),
        svd_rrule_algs = [:TruncPullback, :Arnoldi], naive = true,
    ),
    (
        name = "p-wave superconductor", H = pwave_superconductor(InfiniteSquare()),
        Pspace = Vect[FermionParity](0 => 1, 1 => 1), Vspace = Vect[FermionParity](0 => 1, 1 => 1),
        Espace = Vect[FermionParity](0 => 3, 1 => 3), svd_rrule_algs = [:Arnoldi], naive = false,
    ),
]
forward_algs = [
    (:SimultaneousCTMRG, :HalfInfiniteProjector),
    (:SimultaneousCTMRG, :FullInfiniteProjector),
    (:SequentialCTMRG, :HalfInfiniteProjector),
    (:SequentialCTMRG, :FullInfiniteProjector),
]

ctmrg_algorithm(ctmrg_alg, projector_alg, rrule_alg = :FullPullback; kwargs...) = PEPSKit.CTMRGAlgorithm(;
    alg = ctmrg_alg, projector_alg, decomposition_alg = SVDAdjoint(; rrule_alg = (; alg = rrule_alg)),
    tol = ctmrg_tol, verbosity = ctmrg_verbosity, kwargs...,
)
gradient_alg(alg, solver) = PEPSKit.GradientAlgorithm(; alg, solver_alg = (; alg = solver, tol = solver_tol))

# only the simultaneous scheme with the half-infinite projector exposes the SVD it needs
implicit_allowed(ctmrg_alg, projector_alg) =
    ctmrg_alg == :SimultaneousCTMRG && projector_alg == :HalfInfiniteProjector

# (label, SVD rrule, gradient algorithm or nothing for naive AD)
function gradient_cases(model, ctmrg_alg, projector_alg)
    cases = Any[("fixed point, $r", r, gradient_alg(:FixedPointGradient, :GMRES)) for r in model.svd_rrule_algs]
    if model.naive && (ctmrg_alg, projector_alg) != (:SequentialCTMRG, :FullInfiniteProjector)
        push!(cases, ("naive AD", :FullPullback, nothing))
    end
    if implicit_allowed(ctmrg_alg, projector_alg)
        push!(cases, ("implicit, GMRES", :FullPullback, gradient_alg(:ImplicitGradient, :GMRES)))
    end
    if (ctmrg_alg, projector_alg) == first(forward_algs)
        for solver in (:GeomSum, :ManualIter, :BiCGStab, :Arnoldi)
            push!(cases, ("fixed point, $solver", :FullPullback, gradient_alg(:FixedPointGradient, solver)))
        end
        push!(cases, ("implicit, BiCGStab", :FullPullback, gradient_alg(:ImplicitGradient, :BiCGStab)))
    end
    return cases
end

function test_ctmrg_gradients(AT, model, ctmrg_alg, projector_alg, cases; D = nothing, χ = nothing)
    Vspace = isnothing(D) ? model.Vspace : ComplexSpace(D)
    Espace = isnothing(χ) ? model.Espace : ComplexSpace(χ)
    Random.seed!(42039482030)
    dir = adapt(AT, InfinitePEPS(model.Pspace, Vspace))
    psi = adapt(AT, InfinitePEPS(model.Pspace, Vspace))
    alg = ctmrg_algorithm(ctmrg_alg, projector_alg)
    env, = leading_boundary(CTMRGEnv(psi, Espace), psi, alg)
    tag = "$(model.name), $ctmrg_alg, $projector_alg, $(dim(Vspace)):$(dim(Espace))"

    _, gref = energy_and_gradient(psi, env, alg, gradient_alg(:FixedPointGradient, :GMRES), model.H)
    @testset "reference against finite differences" begin
        dE_fd = finite_difference_derivative(psi, env, dir, alg, model.H)
        dE = gradient_derivative(psi, env, dir, gref)
        @info "$tag: reference vs finite differences" abs(dE - dE_fd) / abs(dE_fd)
        @test dE ≈ dE_fd rtol = rtol
    end
    for (label, rrule_alg, galg) in cases
        @testset "$label" begin
            kwargs = isnothing(galg) ? (; miniter = naive_miniter) : (;)
            calg = ctmrg_algorithm(ctmrg_alg, projector_alg, rrule_alg; kwargs...)
            _, g = energy_and_gradient(psi, env, calg, galg, model.H)
            err = gradient_error(g, gref)
            @info "$tag: $label" err
            @test err < rtol
        end
    end
    return nothing
end

function gradients_asymmetric(AT)
    return @testset "CTMRG gradients ($AT)" verbose = true begin
        @testset "$(model.name), $calg, $palg" for model in models, (calg, palg) in forward_algs
            test_ctmrg_gradients(AT, model, calg, palg, gradient_cases(model, calg, palg))
            # the implicit gradient is rejected where it is not defined
            if !implicit_allowed(calg, palg)
                @test_throws ArgumentError PEPSOptimize(;
                    boundary_alg = (; alg = calg, projector_alg = palg),
                    gradient_alg = (; alg = :ImplicitGradient),
                )
            end
        end
        # larger state, where truncation effects are visible
        @testset "Heisenberg, SimultaneousCTMRG, HalfInfiniteProjector, D = 3, χ = 16" begin
            cases = [("implicit, GMRES", :FullPullback, gradient_alg(:ImplicitGradient, :GMRES))]
            test_ctmrg_gradients(AT, first(models), first(forward_algs)..., cases; D = 3, χ = 16)
        end
    end
end

function gradients_asymmetric_276(AT)
    ## Regression test for gradient accuracy (https://github.com/QuantumKitHub/PEPSKit.jl/pull/276)
    return @testset "AD CTMRG energy gradient accuracy regression test (#276) ($AT)" begin
        Random.seed!(1234)

        boundary_alg = PEPSKit.CTMRGAlgorithm(; tol = 1.0e-10)
        gradient_alg = PEPSKit.GradientAlgorithm(; tol = 5.0e-8)

        function fg((peps, env))
            E, g = Zygote.withgradient(peps) do ψ
                env2, = PEPSKit.hook_pullback(
                    leading_boundary,
                    env,
                    ψ,
                    boundary_alg;
                    alg_rrule = gradient_alg,
                )
                return cost_function(ψ, env2, H)
            end
            return E, only(g)
        end

        # initialize randomly
        H = adapt(AT, heisenberg_XYZ(InfiniteSquare(1, 1)))
        peps = PEPSKit.peps_normalize(adapt(AT, InfinitePEPS(randn, ComplexF64, physicalspace(H)[1], ComplexSpace(3))))
        env0 = CTMRGEnv(randn, ComplexF64, peps, ComplexSpace(20))

        # test gradient against finite-difference
        Δx = 1.0e-5
        _, _, dfs1, dfs2 = OptimKit.optimtest(
            fg, (peps, env0);
            alpha = LinRange(-Δx, Δx, 2),
            retract = PEPSKit.peps_retract,
            inner = PEPSKit.real_inner,
        )

        # verify high gradient accuracy for small finite-difference step size
        @test dfs1 ≈ dfs2 rtol = 1.0e-2 * Δx
    end
end

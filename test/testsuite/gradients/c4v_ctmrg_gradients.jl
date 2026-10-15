using Test
using Adapt
using Random
using LinearAlgebra
using PEPSKit
using TensorKit
using Zygote
using KrylovKit

include("gradient_utility.jl")

## Test C4v CTMRG gradients
# -------------------------------------------
# Every gradient algorithm is compared with a converged fixed-point reference gradient, which is
# checked against finite differences. Gradients are C4v-symmetrized before comparing.

sd = 42039482052
symmetry = RotateReflect()
H = heisenberg_XYZ(InfiniteSquare())
Pspace = ComplexSpace(2)
ctmrg_tol = 1.0e-12
ctmrg_maxiter = 300 # QR needs about 250 iterations at D = 3, χ = 16
solver_tol = 1.0e-10
ctmrg_verbosity = 1
rtol = 1.0e-7 # gradient errors and finite differences
naive_miniter = 30 # CTMRG iterations naive AD differentiates through
projector_algs = (:C4vEighProjector, :C4vQRProjector)

decomposition_alg(projector_alg, rrule_alg) = if projector_alg == :C4vEighProjector
    EighAdjoint(; rrule_alg = (; alg = rrule_alg))
elseif projector_alg == :C4vQRProjector
    QRAdjoint(; rrule_alg = (; alg = rrule_alg))
else
    error("unknown projector alg: $projector_alg")
end

c4v_ctmrg_alg(projector_alg, rrule_alg = :FullPullback; kwargs...) = PEPSKit.CTMRGAlgorithm(;
    alg = :C4vCTMRG, projector_alg, decomposition_alg = decomposition_alg(projector_alg, rrule_alg),
    tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity, kwargs...,
)
gradient_alg(alg, solver) = PEPSKit.GradientAlgorithm(; alg, solver_alg = (; alg = solver, tol = solver_tol))

# (label, decomposition rrule, gradient algorithm or nothing for naive AD)
function gradient_cases(projector_alg)
    cases = Any[
        ("naive AD", :FullPullback, nothing),
        ("implicit, GMRES", :FullPullback, gradient_alg(:ImplicitGradient, :GMRES)),
    ]
    projector_alg == :C4vEighProjector || return cases
    return append!(
        cases, Any[
            ("fixed point, TruncPullback", :TruncPullback, gradient_alg(:FixedPointGradient, :GMRES)),
            ("fixed point, GeomSum", :FullPullback, gradient_alg(:FixedPointGradient, :GeomSum)),
            ("fixed point, ManualIter", :FullPullback, gradient_alg(:FixedPointGradient, :ManualIter)),
            ("fixed point, BiCGStab", :FullPullback, gradient_alg(:FixedPointGradient, :BiCGStab)),
            ("fixed point, Arnoldi", :FullPullback, gradient_alg(:FixedPointGradient, :Arnoldi)),
            ("implicit, BiCGStab", :FullPullback, gradient_alg(:ImplicitGradient, :BiCGStab)),
        ]
    )
end

function test_c4v_gradients(AT, projector_alg, D, χ, cases; seed = sd)
    Random.seed!(seed)
    psi = symmetrize!(adapt(AT, InfinitePEPS(Pspace, ComplexSpace(D))), symmetry)
    dir = symmetrize!(adapt(AT, InfinitePEPS(Pspace, ComplexSpace(D))), symmetry)
    alg = c4v_ctmrg_alg(projector_alg)
    env, = leading_boundary(PEPSKit.initialize_random_c4v_env(psi, ComplexSpace(χ)), psi, alg)
    tag = "C4v $projector_alg D=$D χ=$χ"

    _, gref = energy_and_gradient(psi, env, alg, gradient_alg(:FixedPointGradient, :GMRES), H)
    symmetrize!(gref, symmetry)
    @testset "reference against finite differences" begin
        dE_fd = finite_difference_derivative(psi, env, dir, alg, H)
        dE = gradient_derivative(psi, env, dir, gref)
        @info "$tag: reference vs finite differences" abs(dE - dE_fd) / abs(dE_fd)
        @test dE ≈ dE_fd rtol = rtol
    end
    for (label, rrule_alg, galg) in cases
        @testset "$label" begin
            kwargs = isnothing(galg) ? (; miniter = naive_miniter) : (;)
            _, g = energy_and_gradient(psi, env, c4v_ctmrg_alg(projector_alg, rrule_alg; kwargs...), galg, H)
            err = gradient_error(symmetrize!(g, symmetry), gref)
            @info "$tag: $label" err
            @test err < rtol
        end
    end
    return nothing
end

function gradients_c4v(AT)
    return @testset "C4v CTMRG gradients ($AT)" verbose = true begin
        @testset "$palg, D = 2, χ = 6" for palg in projector_algs
            test_c4v_gradients(AT, palg, 2, 6, gradient_cases(palg))
        end
        # larger state, where truncation effects are visible
        @testset "$palg, D = 3, χ = 16" for palg in projector_algs
            cases = [("implicit, GMRES", :FullPullback, gradient_alg(:ImplicitGradient, :GMRES))]
            # sd lands on a sign-flipping eigh fixed point here, see #436
            test_c4v_gradients(AT, palg, 3, 16, cases; seed = 11)
        end
    end
end

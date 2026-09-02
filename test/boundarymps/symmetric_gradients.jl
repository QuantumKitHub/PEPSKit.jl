using Test
using Random
using LinearAlgebra
using PEPSKit
using MPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

sd = 42039482049

## Test symmetric boundary MPS gradients through implicit differentiation
# -----------------------------------------------------------------------
χbond = 2
χenv = 8
symmetry = RotateReflect()
Pspaces = [ComplexSpace(2), ComplexSpace(2)]
Vspaces = [ComplexSpace(χbond), ComplexSpace(χbond)]
Espaces = [ComplexSpace(χenv), ComplexSpace(χenv)]
models = [heisenberg_XYZ(InfiniteSquare()), transverse_field_ising(InfiniteSquare())]
names = ["Heisenberg", "Ising"]

boundary_tol = 1.0e-10
boundary_maxiter = 400
boundary_verbosity = 1
gradtol = 1.0e-8
comp_tol = 1.0e-5
steps = -0.01:0.005:0.01

mps_algs = [:VUMPS, :VOMPS]

## Tests
# ------
@testset "Symmetric boundary MPS energy gradients for $(names[i]) model" verbose = true for i in
    eachindex(
        models
    )
    H = models[i]
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]

    Random.seed!(sd)
    psi = symmetrize!(InfinitePEPS(Pspace, Vspace), symmetry)
    dir = symmetrize!(InfinitePEPS(Pspace, Vspace), symmetry)

    gradient_alg = ImplicitGradient(; solver_alg = (; alg = :GMRES, tol = gradtol))

    # reference gradient from a CTMRG contraction differentiated through the fixed point
    ctmrg_alg = SimultaneousCTMRG(;
        tol = boundary_tol, maxiter = boundary_maxiter, verbosity = boundary_verbosity
    )
    ctmrg_gradient_alg = FixedPointGradient(; solver_alg = (; alg = :GMRES, tol = gradtol))
    ctmrg_env₀ = CTMRGEnv(psi, Espace)
    ctmrg_env, = leading_boundary(ctmrg_env₀, psi, ctmrg_alg)
    N_ref = norm(psi, ctmrg_env)
    E_ref, g_ref = Zygote.withgradient(psi) do ψ
        env, = PEPSKit.hook_pullback(
            leading_boundary, ctmrg_env₀, ψ, ctmrg_alg; alg_rrule = ctmrg_gradient_alg
        )
        return cost_function(ψ, env, H)
    end
    g_ref = only(g_ref)
    symmetrize!(g_ref, symmetry)

    @testset "mps_alg=:$mps_alg" for mps_alg in mps_algs
        boundary_alg = SymmetricBoundaryMPS(;
            mps_alg, tol = boundary_tol,
            maxiter = boundary_maxiter, verbosity = boundary_verbosity,
        )
        env₀ = SymmetricBoundaryMPSEnv(psi, Espace)
        env, info = leading_boundary(env₀, psi, boundary_alg)

        # the forward contraction should agree with CTMRG
        @test info.converged
        @test network_value(psi, env) ≈ N_ref rtol = 1.0e-6

        @info "optimtest of mps_alg=:$mps_alg on $(names[i])"
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha = steps,
            retract = PEPSKit.peps_retract,
            inner = PEPSKit.real_inner,
        ) do (peps, e)
            E, g = Zygote.withgradient(peps) do ψ
                env´, = PEPSKit.hook_pullback(
                    leading_boundary, e, ψ, boundary_alg; alg_rrule = gradient_alg
                )
                return cost_function(ψ, env´, H)
            end
            g = only(g)
            symmetrize!(g, symmetry)
            return E, g
        end
        @test dfs1 ≈ dfs2 atol = 1.0e-2

        @info "direct gradient comparison of mps_alg=:$mps_alg on $(names[i])"
        E_trial, g_trial = Zygote.withgradient(psi) do ψ
            env´, = PEPSKit.hook_pullback(
                leading_boundary, env₀, ψ, boundary_alg; alg_rrule = gradient_alg
            )
            return cost_function(ψ, env´, H)
        end
        g_trial = only(g_trial)
        symmetrize!(g_trial, symmetry)

        @test E_trial ≈ E_ref rtol = comp_tol
        @test norm(g_ref - g_trial) / norm(g_ref) < comp_tol
    end
end

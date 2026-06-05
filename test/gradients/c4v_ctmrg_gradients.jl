using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

sd = 42039482052

## Test C4v CTMRG gradients
# -------------------------------------------
χbond = 2
χenv = 6
symmetry = RotateReflect()
Pspaces = [ComplexSpace(2)]
Vspaces = [ComplexSpace(χbond)]
Espaces = [ComplexSpace(χenv)]
models = [heisenberg_XYZ(InfiniteSquare())]
names = ["Heisenberg"]

gradtol = 1.0e-4
ctmrg_verbosity = 1
ctmrg_algs = [[:C4vCTMRG]]
projector_algs = [[:C4vEighProjector, :C4vQRProjector]]
decomposition_rrule_algs = [[:FullPullback, :TruncPullback]]
gradient_algs = [[nothing, :FixedPointGradient]]
gradient_solver_algs = [[:GeomSum, :ManualIter, :GMRES, :BiCGStab, :Arnoldi]]
steps = -0.01:0.005:0.01

# record which rrule alg is compatible with which projector alg
allowed_rrule_algs = Dict(
    :C4vEighProjector => keys(PEPSKit.EIGH_RRULE_SYMBOLS),
    :C4vQRProjector => keys(PEPSKit.QR_RRULE_SYMBOLS),
)

# be selective on which configurations to test the naive gradient for
naive_gradient_combinations = [(:C4vCTMRG, :C4vEighProjector, :FullPullback), (:C4vCTMRG, :C4vQRProjector, :FullPullback)]
naive_gradient_done = Set()

## Tests
# ------
@testset "AD C4v CTMRG energy gradients for $(names[i]) model" verbose = true for i in
    eachindex(
        models
    )
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]
    calgs = ctmrg_algs[i]
    palgs = projector_algs[i]
    dalgs = decomposition_rrule_algs[i]
    galgs = gradient_algs[i]
    gsalgs = gradient_solver_algs[i]
    @testset "ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, decomposition_rrule_alg=:$decomposition_rrule_alg and gradient_alg=(alg = :$gradient_alg, solver_alg = :$gradient_solver_alg)" for (
            ctmrg_alg, projector_alg, decomposition_rrule_alg, gradient_alg, gradient_solver_alg,
        ) in Iterators.product(
            calgs, palgs, dalgs, galgs, gsalgs
        )

        # check for allowed algorithm combinations when testing naive gradient
        if isnothing(gradient_alg)
            combo = (ctmrg_alg, projector_alg, decomposition_rrule_alg)
            combo in naive_gradient_combinations || continue
            combo in naive_gradient_done && continue
            push!(naive_gradient_done, combo)
            gradient_solver_alg = nothing # unused in naive gradient, so set to nothing to avoid confusion
        end

        # check for allowed combinations of projector alg and decomposition rrule alg
        decomposition_rrule_alg in allowed_rrule_algs[projector_alg] || continue

        # construct appropriate decomposition struct to pass custom rrule alg
        decomposition_alg = if projector_alg == :C4vEighProjector
            EighAdjoint(; rrule_alg = (; alg = decomposition_rrule_alg))
        elseif projector_alg == :C4vQRProjector
            QRAdjoint(; rrule_alg = (; alg = decomposition_rrule_alg))
        else
            error("unknown projector alg: $projector_alg")
        end

        @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, decomposition_rrule_alg=:$decomposition_rrule_alg and gradient_alg=(; alg = :$gradient_alg, solver_alg = (; alg = :$gradient_solver_alg)) on $(names[i])"
        Random.seed!(sd)
        dir = InfinitePEPS(Pspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace)
        symmetrize!(psi, symmetry)
        symmetrize!(dir, symmetry)
        # instantiate to avoid having to type this twice...
        contrete_ctmrg_alg = PEPSKit.CTMRGAlgorithm(;
            alg = ctmrg_alg,
            verbosity = ctmrg_verbosity,
            projector_alg = projector_alg,
            decomposition_alg,
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing # TODO: add this to the PEPSKit.GradientAlgorithm selector?
        else
            PEPSKit.GradientAlgorithm(;
                alg = gradient_alg, solver_alg = (; alg = gradient_solver_alg, tol = gradtol)
            )
        end
        env0 = PEPSKit.initialize_random_c4v_env(psi, Espace)
        env, = leading_boundary(env0, psi, contrete_ctmrg_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha = steps,
            retract = PEPSKit.peps_retract,
            inner = PEPSKit.real_inner,
        ) do (peps, env)
            E, g = Zygote.withgradient(peps) do psi
                env2, = PEPSKit.hook_pullback(
                    leading_boundary,
                    env,
                    psi,
                    contrete_ctmrg_alg;
                    alg_rrule = concrete_gradient_alg,
                )
                return cost_function(psi, env2, models[i])
            end
            g = only(g)
            symmetrize!(g, symmetry)
            return E, g
        end
        @test dfs1 ≈ dfs2 atol = 1.0e-2
    end
end

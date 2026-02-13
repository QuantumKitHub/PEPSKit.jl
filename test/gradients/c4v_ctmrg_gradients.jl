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
ctmrg_algs = [[:c4v]]
projector_algs = [[:c4v_eigh, :c4v_qr]]
decomposition_rrule_algs = [[:full, :trunc, :qr]]
gradient_algs = [[nothing, :geomsum, :manualiter, :linsolver, :eigsolver]]
gradient_iterschemes = [[:fixed, :diffgauge]]
steps = -0.01:0.005:0.01

# record which rrule alg is compatible with which projector alg
allowed_rrule_algs = Dict(
    :c4v_eigh => keys(PEPSKit.EIGH_RRULE_SYMBOLS),
    :c4v_qr => keys(PEPSKit.QR_RRULE_SYMBOLS),
)

# be selective on which configurations to test the naive gradient for
naive_gradient_combinations = [
    (:c4v, :c4v_eigh, :full, :fixed),
    (:c4v, :c4v_eigh, :full, :diffgauge),
    (:c4v, :c4v_qr, :qr, :fixed),
    (:c4v, :c4v_qr, :qr, :diffgauge),
]
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
    gischemes = gradient_iterschemes[i]
    @testset "ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, decomposition_rrule_alg=:$decomposition_rrule_alg, gradient_alg=:$gradient_alg, gradient_iterscheme=:$gradient_iterscheme" for (
            ctmrg_alg, projector_alg, decomposition_rrule_alg, gradient_alg, gradient_iterscheme,
        ) in Iterators.product(
            calgs, palgs, dalgs, galgs, gischemes
        )

        # check for allowed algorithm combinations when testing naive gradient
        if isnothing(gradient_alg)
            combo = (ctmrg_alg, projector_alg, decomposition_rrule_alg, gradient_iterscheme)
            combo in naive_gradient_combinations || continue
            combo in naive_gradient_done && continue
            push!(naive_gradient_done, combo)
        end

        # check for allowed combinations of projector alg and decomposition rrule alg
        decomposition_rrule_alg in allowed_rrule_algs[projector_alg] || continue

        @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, decomposition_rrule_alg=:$decomposition_rrule_alg, gradient_alg=:$gradient_alg and gradient_iterscheme=:$gradient_iterscheme on $(names[i])"
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
            decomposition_alg = (; rrule_alg = (; alg = decomposition_rrule_alg)),
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing # TODO: add this to the PEPSKit.GradMode selector?
        else
            PEPSKit.GradMode(; alg = gradient_alg, tol = gradtol, iterscheme = gradient_iterscheme)
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

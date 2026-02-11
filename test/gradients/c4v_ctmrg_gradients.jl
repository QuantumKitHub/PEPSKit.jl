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
projector_algs = [[:c4v_eigh]]
eigh_rrule_algs = [[:full, :trunc]] # TODO: handle projector-algorithm-dependence
gradient_algs = [[nothing, :geomsum, :manualiter, :linsolver, :eigsolver]]
gradient_iterschemes = [[:fixed, :diffgauge]]
steps = -0.01:0.005:0.01

# be selective on which configurations to test the naive gradient for
naive_gradient_combinations = [
    (:c4v, :c4v_eigh, :full, :fixed),
    (:c4v, :c4v_eigh, :trunc, :diffgauge),
]
naive_gradient_done = Set()

# mark the broken configurations as broken explicitly
broken_gradients = Dict(
    "eigh_rrule_alg" => Set([:full]),
    "gradient_iterscheme" => Set([:diffgauge]),
)
function _check_broken(
        ctmrg_alg, projector_alg, eigh_rrule_alg, gradient_alg, gradient_iterscheme
    )
    # naive gradients should always work
    isnothing(gradient_alg) && return false
    # evaluate brokenness
    eigh_rrule_alg in broken_gradients["eigh_rrule_alg"] && return true
    gradient_iterscheme in broken_gradients["gradient_iterscheme"] && return true
    return false
end


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
    ealgs = eigh_rrule_algs[i]
    galgs = gradient_algs[i]
    gischemes = gradient_iterschemes[i]
    @testset "ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, eigh_rrule_alg=:$eigh_rrule_alg, gradient_alg=:$gradient_alg, gradient_iterscheme=:$gradient_iterscheme" for (
            ctmrg_alg, projector_alg, eigh_rrule_alg, gradient_alg, gradient_iterscheme,
        ) in Iterators.product(
            calgs, palgs, ealgs, galgs, gischemes
        )

        # check for allowed algorithm combinations when testing naive gradient
        if isnothing(gradient_alg)
            combo = (ctmrg_alg, projector_alg, eigh_rrule_alg, gradient_iterscheme)
            combo in naive_gradient_combinations || continue
            combo in naive_gradient_done && continue
            push!(naive_gradient_done, combo)
        end

        @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, eigh_rrule_alg=:$eigh_rrule_alg, gradient_alg=:$gradient_alg and gradient_iterscheme=:$gradient_iterscheme on $(names[i])"
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
            decomposition_alg = (; rrule_alg = (; alg = eigh_rrule_alg)),
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
        if _check_broken(
                ctmrg_alg, projector_alg, eigh_rrule_alg, gradient_alg, gradient_iterscheme
            )
            @test_broken dfs1 ≈ dfs2 atol = 1.0e-2
        else
            @test dfs1 ≈ dfs2 atol = 1.0e-2
        end
    end
end

using Test
using Random
using PEPSKit
using TensorKit
using Zygote
using OptimKit
using KrylovKit

## Test models, gradmodes and CTMRG algorithm
# -------------------------------------------
χbond = 2
χenv = 6
Pspaces = [ComplexSpace(2), Vect[FermionParity](0 => 1, 1 => 1)]
Vspaces = [ComplexSpace(χbond), Vect[FermionParity](0 => χbond / 2, 1 => χbond / 2)]
Espaces = [ComplexSpace(χenv), Vect[FermionParity](0 => χenv / 2, 1 => χenv / 2)]
models = [heisenberg_XYZ(InfiniteSquare()), pwave_superconductor(InfiniteSquare())]
names = ["Heisenberg", "p-wave superconductor"]

gradtol = 1e-4
ctmrg_verbosity = 0
ctmrg_algs = [[:sequential, :simultaneous], [:sequential, :simultaneous]]
projector_algs = [[:halfinfinite, :fullinfinite], [:halfinfinite, :fullinfinite]]
svd_rrule_algs = [[:tsvd, :arnoldi], [:tsvd, :arnoldi]]
gradient_algs = [
    [nothing, :geomsum, :manualiter, :linsolver, :eigsolver],
    [:geomsum, :manualiter, :linsolver, :eigsolver],
]
gradient_iterschemes = [[:fixed, :diffgauge], [:fixed, :diffgauge]]
steps = -0.01:0.005:0.01

## Tests
# ------
@testset "AD CTMRG energy gradients for $(names[i]) model" verbose = true for i in
                                                                              eachindex(
    models
)
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]
    calgs = ctmrg_algs[i]
    palgs = projector_algs[i]
    salgs = svd_rrule_algs[i]
    galgs = gradient_algs[i]
    gischemes = gradient_iterschemes[i]
    @testset "ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, svd_rrule_alg=:$svd_rrule_alg, gradient_alg=:$gradient_alg, gradient_iterscheme=:$gradient_iterscheme" for (
        ctmrg_alg, projector_alg, svd_rrule_alg, gradient_alg, gradient_iterscheme
    ) in Iterators.product(
        calgs, palgs, salgs, galgs, gischemes
    )
        # filter all disallowed combinations
        (ctmrg_alg == :sequential && gradient_iterscheme == :fixed) && continue

        @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, svd_rrule_alg=:$svd_rrule_alg, gradient_alg=:$gradient_alg and gradient_iterscheme=:$gradient_iterscheme on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPS(Pspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace)
        # instantiate to avoid having to type this twice...
        contrete_ctmrg_alg = PEPSKit.CTMRGAlgorithm(;
            alg=ctmrg_alg,
            verbosity=ctmrg_verbosity,
            projector_alg=projector_alg,
            svd_alg=(; rrule_alg=(; alg=svd_rrule_alg)),
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing # TODO: add this to the PEPSKit.GradMode selector?
        else
            PEPSKit.GradMode(; alg=gradient_alg, tol=gradtol, iterscheme=gradient_iterscheme)
        end
        env, = leading_boundary(CTMRGEnv(psi, Espace), psi, contrete_ctmrg_alg)
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha=steps,
            retract=PEPSKit.peps_retract,
            inner=PEPSKit.real_inner,
        ) do (peps, env)
            E, g = Zygote.withgradient(peps) do psi
                env2, = PEPSKit.hook_pullback(
                    leading_boundary,
                    env,
                    psi,
                    contrete_ctmrg_alg;
                    alg_rrule=concrete_gradient_alg,
                )
                return cost_function(psi, env2, models[i])
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1e-2
    end
end

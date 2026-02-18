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

gradtol = 1.0e-4
ctmrg_verbosity = 0
ctmrg_algs = [[:sequential, :simultaneous], [:sequential, :simultaneous]]
projector_algs = [[:halfinfinite, :fullinfinite], [:halfinfinite, :fullinfinite]]
svd_rrule_algs = [[:full, :arnoldi], [:full, :arnoldi]]
gradient_algs = [
    [nothing, :geomsum, :manualiter, :linsolver, :eigsolver],
    [:geomsum, :manualiter, :linsolver, :eigsolver],
]
gradient_iterschemes = [[:fixed, :diffgauge], [:fixed, :diffgauge]]
steps = -0.01:0.005:0.01

naive_gradient_combinations = [
    (:simultaneous, :halfinfinite, :full),
    (:simultaneous, :fullinfinite, :full),
    (:sequential, :halfinfinite, :full),
]
naive_gradient_done = Set()

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
            ctmrg_alg, projector_alg, svd_rrule_alg, gradient_alg, gradient_iterscheme,
        ) in Iterators.product(
            calgs, palgs, salgs, galgs, gischemes
        )
        # filter all disallowed combinations
        (ctmrg_alg == :sequential && gradient_iterscheme == :fixed) && continue

        # check for allowed algorithm combinations when testing naive gradient
        if isnothing(gradient_alg)
            combo = (ctmrg_alg, projector_alg, svd_rrule_alg)
            combo in naive_gradient_combinations || continue
            combo in naive_gradient_done && continue
            push!(naive_gradient_done, combo)
        end

        @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, svd_rrule_alg=:$svd_rrule_alg, gradient_alg=:$gradient_alg and gradient_iterscheme=:$gradient_iterscheme on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPS(Pspace, Vspace)
        psi = InfinitePEPS(Pspace, Vspace)
        # instantiate to avoid having to type this twice...
        contrete_ctmrg_alg = PEPSKit.CTMRGAlgorithm(;
            alg = ctmrg_alg,
            verbosity = ctmrg_verbosity,
            projector_alg = projector_alg,
            decomposition_alg = (; rrule_alg = (; alg = svd_rrule_alg)),
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing # TODO: add this to the PEPSKit.GradMode selector?
        else
            PEPSKit.GradMode(; alg = gradient_alg, tol = gradtol, iterscheme = gradient_iterscheme)
        end
        env, = leading_boundary(CTMRGEnv(psi, Espace), psi, contrete_ctmrg_alg)
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

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1.0e-2
    end
end

## Regression test for gradient accuracy (https://github.com/QuantumKitHub/PEPSKit.jl/pull/276)
@testset "AD CTMRG energy gradient accuracy regression test (#276)" begin
    Random.seed!(1234)

    boundary_alg = PEPSKit.CTMRGAlgorithm(; tol = 1.0e-10)
    gradient_alg = PEPSKit.GradMode(; alg = :linsolver, tol = 5.0e-8, iterscheme = :fixed)

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
    H = heisenberg_XYZ(InfiniteSquare(1, 1))
    peps = PEPSKit.peps_normalize(InfinitePEPS(randn, ComplexF64, physicalspace(H)[1], ComplexSpace(3)))
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

using Test
using Random
using PEPSKit
using TensorKit
using Enzyme
using OptimKit
using KrylovKit
using Adapt
# Enzyme names basic blocks after Julia types (e.g. `zeroType.<T>`); the tape types
# here exceed LLVM's default 1024-char cap on non-global value names, which its
# textual IR parser rejects when Enzyme round-trips the module in `check_ir!`.
Enzyme.LLVM.clopts("--non-global-value-max-name-size=1048576")

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
ctmrg_algs = [[:SequentialCTMRG, :SimultaneousCTMRG], [:SequentialCTMRG, :SimultaneousCTMRG]]
projector_algs = [[:HalfInfiniteProjector, :FullInfiniteProjector], [:HalfInfiniteProjector, :FullInfiniteProjector]]
svd_rrule_algs = [[:FullPullback, :TruncPullback, :Arnoldi], [:FullPullback, :Arnoldi]]
gradient_algs = [[nothing, :FixedPointGradient], [:FixedPointGradient]]
gradient_solver_algs = [
    [:GeomSum, :ManualIter, :GMRES, :BiCGStab, :Arnoldi],
    [:GeomSum, :ManualIter, :GMRES, :BiCGStab, :Arnoldi],
]
steps = -0.01:0.005:0.01

# don't check naive AD gradients for all algorithm combinations, since it's slow
naive_gradient_combinations = [
    (:SimultaneousCTMRG, :HalfInfiniteProjector, :FullPullback),
    (:SimultaneousCTMRG, :FullInfiniteProjector, :FullPullback),
    #(:SequentialCTMRG, :HalfInfiniteProjector, :FullPullback),
]
naive_gradient_done = Set()

# fixed-point differentiation is incompatible with sequential CTMRG
function _check_disallowed_combination(
        ctmrg_alg, projector_alg, decomposition_rrule_alg, gradient_alg
    )
    ctmrg_alg == :SequentialCTMRG && !isnothing(gradient_alg) && return true
    return false
end


## Tests
# ------
function enzyme_gradients_asymmetric(AT)
    naive_gradient_done = Set()
    return @testset "Enzyme AD CTMRG energy gradients for $(names[i]) model ($AT)" verbose = true for i in
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
        gsalgs = gradient_solver_algs[i]
        @testset "ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, svd_rrule_alg=:$svd_rrule_alg, gradient_alg=(; alg = :$gradient_alg, solver_alg = (; alg = :$gradient_solver_alg))" for (
                ctmrg_alg, projector_alg, svd_rrule_alg, gradient_alg, gradient_solver_alg,
            ) in Iterators.product(
                calgs, palgs, salgs, galgs, gsalgs
            )

            # only run GMRES for the implicit gradient, and skip distinction between decomposition rrule algs
            if gradient_alg == :ImplicitGradient
                gradient_solver_alg == :GMRES || continue
                svd_rrule_alg == first(salgs) || continue
            end

            # check for allowed algorithm combinations when testing naive gradient
            if isnothing(gradient_alg)
                combo = (ctmrg_alg, projector_alg, svd_rrule_alg)
                combo in naive_gradient_combinations || continue
                combo in naive_gradient_done && continue
                push!(naive_gradient_done, combo)
                gradient_solver_alg = nothing # unused in naive gradient, so set to nothing to avoid confusion
            end


            # filter disallowed algorithm combinations
            if _check_disallowed_combination(
                    ctmrg_alg, projector_alg, svd_rrule_alg, gradient_alg
                )
                # but verify that its use would throw an error
                @test_throws ArgumentError PEPSOptimize(;
                    boundary_alg = (; alg = ctmrg_alg, projector_alg, decomposition_alg = (; rrule_alg = (; alg = svd_rrule_alg))),
                    gradient_alg = (; alg = gradient_alg, solver_alg = (; alg = gradient_solver_alg, tol = gradtol)),
                )
                continue
            end

            @info "optimtest of ctmrg_alg=:$ctmrg_alg, projector_alg=:$projector_alg, svd_rrule_alg=:$svd_rrule_alg and gradient_alg=(; alg = :$gradient_alg, solver_alg = (; alg = :$gradient_solver_alg)) on $(names[i])"
            Random.seed!(42039482030)
            dir = adapt(AT, InfinitePEPS(Pspace, Vspace))
            psi = adapt(AT, InfinitePEPS(Pspace, Vspace))
            # instantiate to avoid having to type this twice...
            contrete_ctmrg_alg = PEPSKit.CTMRGAlgorithm(;
                alg = ctmrg_alg,
                verbosity = ctmrg_verbosity,
                projector_alg = projector_alg,
                decomposition_alg = SVDAdjoint(; rrule_alg = (; alg = svd_rrule_alg)),
            )
            # instantiate because hook_pullback doesn't go through the keyword selector...
            concrete_gradient_alg = if isnothing(gradient_alg)
                nothing # TODO: add this to the PEPSKit.GradientAlgorithm selector?
            else
                PEPSKit.GradientAlgorithm(;
                    alg = gradient_alg, solver_alg = (; alg = gradient_solver_alg, tol = gradtol)
                )
            end
            env, = leading_boundary(CTMRGEnv(psi, Espace), psi, contrete_ctmrg_alg)
            model = adapt(AT, models[i])
            alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
                (psi, env),
                dir;
                alpha = steps,
                retract = PEPSKit.peps_retract,
                inner = PEPSKit.real_inner,
            ) do (peps, env)
                function energ(psi)
                    env2, info = PEPSKit.hook_pullback(
                        leading_boundary, env, psi, contrete_ctmrg_alg;
                        alg_rrule = concrete_gradient_alg,
                    )
                    return cost_function(psi, env2, model)
                end
                dpeps = zerovector(peps)
                _, E = Enzyme.autodiff(
                    set_runtime_activity(ReverseWithPrimal), Const(energ), Active,
                    Duplicated(peps, dpeps),
                )
                return E, dpeps
            end
            @test dfs1 ≈ dfs2 atol = 1.0e-2
        end
    end
end

function enzyme_gradients_asymmetric_276(AT)
    ## Regression test for gradient accuracy (https://github.com/QuantumKitHub/PEPSKit.jl/pull/276)
    return @testset "Enzyme AD CTMRG energy gradient accuracy regression test (#276) ($AT)" begin
        Random.seed!(1234)

        boundary_alg = PEPSKit.CTMRGAlgorithm(; tol = 1.0e-10)
        gradient_alg = PEPSKit.GradientAlgorithm(; tol = 5.0e-8)

        function fg((peps, env))
            function energ(ψ)
                env2, = leading_boundary(env, ψ, boundary_alg)
                return cost_function(ψ, env2, H)
            end
            E, gs = Enzyme.autodiff(ReverseWithPrimal, Const(energ), Active, Duplicated(peps, zerovector(peps)))
            return E, only(gs)
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

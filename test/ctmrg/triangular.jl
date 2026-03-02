using Test
using Random
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using Zygote
using LinearAlgebra
using KrylovKit

sd = 123456
Random.seed!(sd)

D = 2
χ = 7
T = ComplexF64

ctmrg_tol = 1.0e-8
ctmrg_maxiter = 300
ctmrg_verbosity = 2

const ising_βc_triangular = BigFloat(BigFloat(asinh(BigFloat(sqrt(BigFloat(1.0) / BigFloat(3.0))))) / BigFloat(2.0))
const f_onsager_triangular::BigFloat = -3.20253248660790791834355252025862951439

function classical_ising_triangular(β)
    t = Float64[exp(β) exp(-β); exp(-β) exp(β)]

    r = eigen(t)
    nt = r.vectors * sqrt(LinearAlgebra.Diagonal(r.values)) * r.vectors

    O = zeros(2, 2, 2, 2, 2, 2)
    O[1, 1, 1, 1, 1, 1] = 1
    O[2, 2, 2, 2, 2, 2] = 1

    H = [1 1; 1 -1] / sqrt(2)

    @tensor o[-1 -2 -3; -4 -5 -6] := O[1 2 3; 4 5 6] * nt[-1; 1] * nt[-2; 2] * nt[-3; 3] * nt[-4; 4] * nt[-5; 5] * nt[-6; 6]
    @tensor o2[-1 -2 -3; -4 -5 -6] := o[1 2 3; 4 5 6] * H[-1; 1] * H[-2; 2] * H[-3; 3] * H[-4; 4] * H[-5; 5] * H[-6; 6]
    return TensorMap(o2, ℂ^2 * ℂ^2 * ℂ^2, ℂ^2 * ℂ^2 * ℂ^2)
end

@testset "CTM_triangular - Random tensor" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(79413165445)
            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            pspace = ComplexSpace(2)
            vspace = ComplexSpace(D)
            envspace = ComplexSpace(χ)

            ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)
            pf = randn(T, vspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ vspace)
            sandwiches = [pf, (ket, bra)]
            # sandwiches = [(ket, bra), pf]
            unitcell = (2, 2)

            for (sandwich, V) in zip(sandwiches, [vspace, vspace ⊗ vspace'])
                # for (sandwich, vspace) in zip(sandwiches, [vspace ⊗ vspace', vspace])
                network = InfiniteTriangularNetwork(fill(sandwich, unitcell))
                env₀ = CTMRGEnvTriangular(randn, T, V, envspace; unitcell)
                env, info = leading_boundary(env₀, network, alg)
                @test info.convergence_metric < 1.0 # this is not much of a test
            end
        end
    end
end

@testset "CTM_triangular - Classical Ising" begin
    χ_local = 20
    T_local = Float64

    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(156484561351)

            alg = SimultaneousCTMRGTriangular(;
                tol = ctmrg_tol, maxiter = ctmrg_maxiter, verbosity = ctmrg_verbosity,
                conditioning, projector_alg
            )
            sz = (1, 1)
            T = classical_ising_triangular(ising_βc_triangular)
            pf = InfiniteTriangularNetwork(fill(T, sz))

            vspace = codomain(T)[1]
            envspace = ComplexSpace(χ_local)
            env₀ = CTMRGEnvTriangular(randn, T_local, vspace, envspace; unitcell = sz)
            env, info = leading_boundary(env₀, pf, alg)

            nw_value = network_value(pf, env)
            lz = real(log(nw_value))
            fs = lz * -1 / ising_βc_triangular
            @test fs ≈ f_onsager_triangular rtol = 1.0e-4
        end
    end
end


## Gradient tests for triangular CTMRG
# ------------------------------------

Pspaces = [ComplexSpace(2)]
Vspaces = [ComplexSpace(χbond)]
Espaces = [ComplexSpace(χenv)]
# TODO: actually add triangular lattice models...
models = [transverse_field_ising(InfiniteSquare(); J = 1.0, g = 1.0)]
names = ["Heisenberg"]

gradtol = 1.0e-4
ctmrg_verbosity = 0
ctmrg_algs = [[:triangular]]
projector_algs = [[:twothirds, :full]]
conditionings = [[true, false]]
# svd_rrule_algs = nothing # TODO: allow this
gradient_algs = [[nothing]]
# gradient_iterschemes = [[:fixed, :diffgauge]] # TODO: support this
steps = -0.01:0.005:0.01

# have to use a custom retraction, since the state type of hardcoded...
function peps_triangular_retract(x, η, α)
    peps = x[1]
    env = deepcopy(x[2])

    retractions = norm_preserving_retract.(unitcell(peps), unitcell(η), α)
    peps´ = typeof(peps)(map(first, retractions))
    ξ = typeof(peps)(map(last, retractions))

    return (peps´, env), ξ
end


@testset "Triangular CTMRG energy gradient test for $(names[i]) model" verbose = true for i in
    eachindex(
        models
    )
    Pspace = Pspaces[i]
    Vspace = Vspaces[i]
    Espace = Espaces[i]
    # calgs = ctmrg_algs[i]
    palgs = projector_algs[i]
    conds = conditionings[i]
    # salgs = svd_rrule_algs[i]
    galgs = gradient_algs[i]
    # gischemes = gradient_iterschemes[i]

    # TODO: allow passing actual LocalOperator
    H = models[i]
    O1 = last(first(filter(t -> length(first(t)) == 1, H.terms)))
    O2 = last(first(filter(t -> length(first(t)) == 2, H.terms)))

    @testset "ctmrg_alg=:simultaneous_triangular, projector_alg=:$projector_alg, conditioning=$conditioning, gradient_alg=:$gradient_alg" for (
            projector_alg, conditioning, gradient_alg,
        ) in Iterators.product(palgs, conds, galgs)

        @info "optimtest of ctmrg_alg=:simultaneous_triangular, projector_alg=:$projector_alg, conditioning=$conditioning, gradient_alg=:$gradient_alg on $(names[i])"
        Random.seed!(42039482030)
        dir = InfinitePEPSTriangular(Pspace, Vspace)
        psi = InfinitePEPSTriangular(Pspace, Vspace)

        # TODO: actually go through CTMRGAlgorithm selection
        contrete_ctmrg_alg = SimultaneousCTMRGTriangular(;
            projector_alg, conditioning, verbosity = ctmrg_verbosity,
            trunctype = :FixedSpaceTruncation
        )
        # instantiate because hook_pullback doesn't go through the keyword selector...
        concrete_gradient_alg = if isnothing(gradient_alg)
            nothing # TODO: add this to the PEPSKit.GradMode selector?
        else
            PEPSKit.GradMode(; alg = gradient_alg, tol = gradtol, iterscheme = gradient_iterscheme)
        end
        env, = leading_boundary(
            CTMRGEnvTriangular(randn, ComplexF64, Vspace ⊗ Vspace', Espace),
            psi, contrete_ctmrg_alg
        )
        alphas, fs, dfs1, dfs2 = OptimKit.optimtest(
            (psi, env),
            dir;
            alpha = steps,
            retract = peps_triangular_retract,
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
                return PEPSKit.energy(psi, env2, O1, O2)
            end

            return E, only(g)
        end
        @test dfs1 ≈ dfs2 atol = 1.0e-2
    end
end

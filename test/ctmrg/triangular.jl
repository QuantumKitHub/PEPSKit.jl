using Test
using Random
using MatrixAlgebraKit
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using Zygote

@testset "CTM_triangular - Random tensor" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            alg = SimultaneousCTMRGTria(; conditioning, projector_alg)
            T = Float64
            χ = 7
            pspace = ℂ^2
            vspace = ℂ^3
            envspace = ℂ^χ

            ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)
            pf = randn(T, vspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ vspace)
            sandwiches = [pf, (ket, bra)]
            unitcell = (1, 1)

            for (sandwich, vspace) in zip(sandwiches, [vspace, vspace ⊗ vspace'])
                network = InfiniteTriangularNetwork(fill(sandwich, unitcell))
                env₀ = CTMRGTriaEnv(randn, T, vspace, envspace; unitcell)
                env, info = leading_boundary(env₀, network, alg)
                @test info.convergence_metric < 1.0
            end
        end
    end
end

@testset "CTM_triangular - Differentiability" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            alg = SimultaneousCTMRGTria(; conditioning, projector_alg)
            T = Float64
            χ = 7
            pspace = ℂ^2
            vspace = ℂ^3
            envspace = ℂ^χ
            ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)

            unitcell = (1, 1)
            network = InfiniteTriangularNetwork(fill((ket, bra), unitcell))
            env₀ = CTMRGTriaEnv(randn, T, vspace ⊗ vspace', envspace; unitcell)

            optimizer_alg = LBFGS(4; maxiter = 10)
            real_inner(_, η₁, η₂) = real(dot(η₁, η₂))
            reuse_env = true
            peps₀ = copy(ket)

            function peps_retract(x, η, α)
                peps = x[1]
                env = deepcopy(x[2])

                retractions = norm_preserving_retract.(unitcell(peps), unitcell(η), α)
                peps´ = InfinitePEPS(map(first, retractions))
                ξ = InfinitePEPS(map(last, retractions))

                return (peps´, env), ξ
            end
            retract = peps_retract


            # optimize operator cost function
            (peps_final, env_final), cost_final, ∂cost, numfg, convergence_history = optimize(
                (peps₀, env₀), optimizer_alg;
                retract, inner = real_inner,
            ) do (peps, env)
                start_time = time_ns()
                E, gs = withgradient(peps) do ψ
                    env′, info = hook_pullback(
                        leading_boundary, env, ψ, alg.boundary_alg;
                        alg_rrule = alg.gradient_alg,
                    )
                    # ignore_derivatives() do
                    #     reuse_env && update!(env, env′)
                    #     push!(truncation_errors, info.truncation_error)
                    #     push!(condition_numbers, info.condition_number)
                    # end
                    return energy(ψ, env′, operator)
                end
                g = only(gs)  # `withgradient` returns tuple of gradients `gs`
                push!(gradnorms_unitcell, norm.(g.A))
                push!(times, (time_ns() - start_time) * 1.0e-9)
                return E, g
            end
        end
    end
end

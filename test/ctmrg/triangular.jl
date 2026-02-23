using Test
using Random
using MatrixAlgebraKit
using TensorKit
using MPSKit
using PEPSKit
using OptimKit
using Zygote
using LinearAlgebra
using KrylovKit

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
            alg = SimultaneousCTMRGTriangular(; maxiter = 300, conditioning, projector_alg)
            eltype = ComplexF64
            χ = 7
            pspace = ℂ^2
            vspace = ℂ^3
            envspace = ℂ^χ

            ket = randn(eltype, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
            bra = copy(ket)
            pf = randn(eltype, vspace ⊗ vspace ⊗ vspace, vspace ⊗ vspace ⊗ vspace)
            sandwiches = [pf, (ket, bra)]
            # sandwiches = [(ket, bra), pf]
            unitcell = (2, 2)

            for (sandwich, vspace) in zip(sandwiches, [vspace, vspace ⊗ vspace'])
            # for (sandwich, vspace) in zip(sandwiches, [vspace ⊗ vspace', vspace])
                network = InfiniteTriangularNetwork(fill(sandwich, unitcell))
                env₀ = CTMRGEnvTriangular(randn, eltype, vspace, envspace; unitcell)
                env, info = leading_boundary(env₀, network, alg)
                @test info.convergence_metric < 1.0
            end
        end
    end
end

@testset "CTM_triangular - Classical Ising" begin
    for conditioning in [true false]
        for projector_alg in [:twothirds :full]
            Random.seed!(156484561351)

            alg = SimultaneousCTMRGTriangular(; maxiter = 300, conditioning, projector_alg)
            unitcell = (1, 1)
            T = classical_ising_triangular(ising_βc_triangular)
            pf = InfiniteTriangularNetwork(fill(T, unitcell))

            χ = 20
            vspace = codomain(T)[1]
            envspace = ℂ^χ
            eltype = Float64
            env₀ = CTMRGEnvTriangular(randn, eltype, vspace, envspace; unitcell)
            env, info = leading_boundary(env₀, pf, alg)

            nw_value = network_value(pf, env)
            lz = real(log(nw_value))
            fs = lz * -1 / ising_βc_triangular
            @test fs ≈ f_onsager_triangular rtol = 1.0e-4
        end
    end
end

# @testset "CTM_triangular - Differentiability" begin
#     for conditioning in [true false]
#         for projector_alg in [:twothirds :full]
#             ctm_alg = SimultaneousCTMRGTriangular(; conditioning, projector_alg)
#             T = ComplexF64
#             S = Trivial
#             χ = 7
#             pspace = ℂ^2

#             J = 1.0
#             g = 2.5

#             ZZ = rmul!(PEPSKit.σᶻᶻ(T, S), -J)
#             X = rmul!(PEPSKit.σˣ(T, S), g * -J)
        
#             vspace = ℂ^3
#             envspace = ℂ^χ
#             ket = randn(T, pspace, vspace ⊗ vspace ⊗ vspace ⊗ vspace' ⊗ vspace' ⊗ vspace')
#             bra = copy(ket)

#             unitcell = (1, 1)
#             network = InfiniteTriangularNetwork(fill((ket, bra), unitcell))
#             env₀ = CTMRGEnvTriangular(randn, T, vspace ⊗ vspace', envspace; unitcell)

#             env, info = leading_boundary(env₀, network, ctm_alg)
#             E₀ = PEPSKit.energy(network, env, X, ZZ)
#             optimizer_alg = LBFGS(4; maxiter = 10)
#             real_inner(_, η₁, η₂) = real(dot(η₁, η₂))
#             reuse_env = true
#             peps₀ = InfinitePEPSTriangular(copy(ket))

#             function peps_retract(x, η, α)
#                 peps = x[1]
#                 env = deepcopy(x[2])

#                 retractions = norm_preserving_retract.(unitcell(peps), unitcell(η), α)
#                 peps´ = InfinitePEPS(map(first, retractions))
#                 ξ = InfinitePEPS(map(last, retractions))

#                 return (peps´, env), ξ
#             end
#             retract = peps_retract

#             # alg = PEPSOptimize()
#             gradtol = 1e-3
#             gradient_alg = EigSolver(; solver_alg=Arnoldi(; tol=gradtol, eager=true), iterscheme=:fixed)
#             # optimize operator cost function
#             (peps_final, env_final), cost_final, ∂cost, numfg, convergence_history = optimize(
#                 (peps₀, env₀), optimizer_alg;
#                 retract, inner = real_inner,
#             ) do (peps, env)
#                 start_time = time_ns()
#                 E, gs = withgradient(peps) do ψ
#                     env′, info = PEPSKit.hook_pullback(
#                         leading_boundary, env, ψ, ctm_alg;
#                         alg_rrule = gradient_alg,
#                     )
#                     # ignore_derivatives() do
#                     #     reuse_env && update!(env, env′)
#                     #     push!(truncation_errors, info.truncation_error)
#                     #     push!(condition_numbers, info.condition_number)
#                     # end
#                     return energy(ψ, env′, operator_onesite, operator_twosite)
#                 end
#                 g = only(gs)  # `withgradient` returns tuple of gradients `gs`
#                 push!(gradnorms_unitcell, norm.(g.A))
#                 push!(times, (time_ns() - start_time) * 1.0e-9)
#                 return E, g
#             end
#         end
#     end
# end

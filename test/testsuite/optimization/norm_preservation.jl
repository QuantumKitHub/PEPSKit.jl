using Test
using Random
using LinearAlgebra
using TensorKit
using PEPSKit
using Adapt

"""Check local tangent projection and norm-preserving retraction of PEPS tensors."""
function optimization_tangent_projection(AT)
    return @testset "PEPS tangent projection ($T, $AT)" for T in (Float64, ComplexF64)
        Random.seed!(1234)
        peps = adapt(AT, InfinitePEPS(randn, T, ℂ^2, ℂ^2; unitcell = (1, 2)))
        rmul!(peps[1, 2], 3)
        direction = adapt(AT, InfinitePEPS(randn, T, ℂ^2, ℂ^2; unitcell = (1, 2)))
        original_peps = deepcopy(peps)
        original_direction = deepcopy(direction)

        tangent = PEPSKit.peps_project_tangent(direction, peps)
        (retracted, _), _ = PEPSKit.peps_retract((peps, nothing), tangent, 0.3)
        for i in eachindex(peps)
            @test real(dot(peps[i], tangent[i])) ≈ 0 atol = 1.0e-12
            @test norm(retracted[i]) ≈ norm(peps[i]) rtol = 1.0e-12
            @test peps[i] == original_peps[i]
            @test direction[i] == original_direction[i]
        end
        if T <: Complex
            phase_direction = im * peps
            @test norm(PEPSKit.peps_project_tangent(phase_direction, peps) - phase_direction) < 1.0e-12
        end
    end
end

"""Check local tensor norms during CTMRG optimization with and without preconditioning."""
function optimization_norm_preservation(AT)
    return @testset "Traditional CTMRG optimization preserves local tensor norms ($AT)" begin
        Random.seed!(123)
        H = adapt(AT, heisenberg_XYZ(InfiniteSquare(); Jx = -1, Jy = 1, Jz = -1))
        peps₀ = adapt(AT, InfinitePEPS(randn, ComplexF64, ℂ^2, ℂ^2))
        env₀ = adapt(AT, CTMRGEnv(peps₀, ℂ^4))

        @testset "preconditioner = $precondition_alg" for precondition_alg in (nothing, (;))
            norms = Float64[]
            radial_gradients = Float64[]
            """Record local tensor norms and radial gradient components after each accepted step."""
            function record_norms((peps, env), E, g, iter)
                append!(norms, norm.(PEPSKit.unitcell(peps)))
                append!(radial_gradients, abs.(real.(dot.(PEPSKit.unitcell(peps), PEPSKit.unitcell(g)))))
                return (peps, env), E, g
            end
            fixedpoint(
                H, peps₀, deepcopy(env₀);
                boundary_alg = (; alg = :SequentialCTMRG, maxiter = 100),
                gradient_alg = (; solver_alg = (; alg = :GMRES)),
                optimizer_alg = (; maxiter = 4, tol = 1.0e-8),
                precondition_alg, verbosity = 0, (finalize!) = record_norms,
            )
            @test !isempty(norms)
            @test all(n -> isapprox(n, 1; atol = 1.0e-12), norms)
            @test all(r -> r < 1.0e-12, radial_gradients)
        end
    end
end

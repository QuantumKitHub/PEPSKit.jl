using Test
using Random
using LinearAlgebra
using TensorKit
using KrylovKit
using VectorInterface: zerovector
using PEPSKit
using Adapt

sd = 20261120
Venv = ComplexSpace(8)

function ctmrg_c4v_krylov_restart(AT)
    return @testset "C4v environment with real corners survives a Krylov restart ($AT)" begin
        Random.seed!(sd)
        peps = adapt(AT, InfinitePEPS(randn, ComplexF64, ComplexSpace(2), ComplexSpace(2)))
        env = initialize_random_c4v_env(peps, Venv)
        @test scalartype(env.corners[1]) <: Real
        @test scalartype(env.edges[1]) <: Complex

        @test typeof(zerovector(env)) == typeof(env)

        # a real-linear map with many distinct real eigenvalues, so that Arnoldi restarts
        M = randn(Float64, Venv ← Venv)
        M = adapt(AT, (M + M') / 2)
        f(e) = C4vCTMRGEnv(map(c -> 0.5 * c, e.corners), map(t -> t * M, e.edges))
        vals, vecs, info = realeigsolve(f, env, 1, :LM, Arnoldi(; krylovdim = 4, maxiter = 50, tol = 1.0e-12))
        @test info.numiter > 1
        @test typeof(first(vecs)) == typeof(env)
    end
end

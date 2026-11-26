using Test, Random, LinearAlgebra, TensorKit, PEPSKit, Adapt

function ctmrg_subspace_iteration(AT)
    return @testset "SI with existing projectors ($AT)" begin
        @testset "$projector, $flavor, fermionic=$fermionic, $unitcell" for projector in (:HalfInfiniteProjector, :FullInfiniteProjector), flavor in (SimultaneousCTMRG, SequentialCTMRG), fermionic in (false, true), unitcell in ((1, 1), (2, 1))
            Random.seed!(265)
            V = fermionic ? Vect[FermionParity](0 => 1, 1 => 1) : ℂ^2
            χ = fermionic ? Vect[FermionParity](0 => 2, 1 => 1) : ℂ^3
            ψ = InfinitePEPS(V, V; unitcell)
            env = adapt(AT, CTMRGEnv(ψ, χ))
            ψ = adapt(AT, ψ)
            K = fuse(χ ⊗ V ⊗ V')
            full = flavor(; projector_alg = projector, miniter = 2, maxiter = 2, tol = 0.0, verbosity = 0)
            si = flavor(;
                projector_alg = projector, miniter = 2, maxiter = 2, tol = 0.0, verbosity = 0,
                decomposition_alg = (; alg = :SubspaceIteration, subspace = K)
            )
            a, = leading_boundary(env, ψ, full)
            b, info = leading_boundary(env, ψ, si)
            H = adapt(AT, fermionic ? pwave_superconductor(InfiniteSquare(unitcell...)) : heisenberg_XYZ(InfiniteSquare(unitcell...)))
            @test cost_function(ψ, b, H) ≈ cost_function(ψ, a, H) rtol = 1.0e-8 atol = 1.0e-10
            @test abs(norm(ψ, b)) ≈ abs(norm(ψ, a)) rtol = 1.0e-8
            @test haskey(info.contraction_metrics, :subspace_error)
            @test si.projector_alg.decomposition_alg.fwd_alg.cache[] === nothing
        end
        @testset "Per-cut warm starts, $flavor" for flavor in (SimultaneousCTMRG, SequentialCTMRG)
            ψ = InfinitePEPS(ℂ^2, ℂ^2; unitcell = (2, 3))
            env = adapt(AT, CTMRGEnv(ψ, ℂ^3))
            ψ = adapt(AT, ψ)
            alg = flavor(; projector_alg = :FullInfiniteProjector, decomposition_alg = (; alg = :SubspaceIteration, subspace = ℂ^5), verbosity = 0)
            active = PEPSKit._prepare_ctmrg(alg, env)
            grid = active.projector_alg.decomposition_alg.fwd_alg.cache
            @test length(unique(objectid(x.cache) for x in grid)) == length(grid)
            for _ in 1:2
                env, = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(ψ), env, active)
            end
            @test all(x -> x.cache[].iteration == 2, grid)
            @test all(x -> dim(domain(x.cache[].X)) == 5, grid)
        end
    end
end

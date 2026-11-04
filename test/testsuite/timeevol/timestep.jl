using Test
using Random
using TensorKit
using PEPSKit
using Adapt

function timeevol_timestep(AT)
    return @testset "SimpleUpdate timestep ($AT)" begin
        Nr, Nc = 2, 2
        H = adapt(AT, real(heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(Nr, Nc); Jx = 1, Jy = 1, Jz = 1)))
        Pspace, Vspace = ℂ^2, ℂ^4
        ψ0 = adapt(AT, InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc)))
        env0 = SUWeight(ψ0)
        alg = SimpleUpdate(; trunc = truncerror(; atol = 1.0e-10) & truncrank(4))
        dt, nstep = 1.0e-2, 50
        # manual timestep
        evolver = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
        allocator = evolver.allocator
        ψ1, env1, info1 = deepcopy(ψ0), deepcopy(env0), nothing
        for iter in 0:(nstep - 1)
            ψ1, env1, info1 = timestep(evolver, ψ1, env1)
        end
        # time_evolve
        ψ2, env2, info2 = time_evolve(ψ0, H, dt, nstep, alg, env0)
        # for-loop syntax
        # Restart the exhausted evolver using the public timestep keywords.
        ψ3, env3, info3 = timestep(evolver, ψ0, env0; iter = 0, t = 0.0)
        for state in evolver
            ψ3, env3, info3 = state
        end
        # results should be *exactly* the same
        @test ψ1 == ψ2 == ψ3
        @test env1 == env2 == env3
        @test info1 == info2 == info3
        @test evolver.allocator === allocator
        timeevol_su_allocator(AT)
    end
end

"Check fresh/reused scratch equivalence, growth, output ownership, and restarts."
function timeevol_su_allocator(AT)
    H = adapt(AT, real(heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(2, 2))))
    return @testset "Persistent scratch ($State, MPO=$force_mpo, $AT)" for
        State in (InfinitePEPS, InfinitePEPO), force_mpo in (false, true)
        ψ0 = adapt(AT, State(randn, Float64, fill(ℂ^2, 2, 2), fill(ℂ^2, 2, 2)))
        env0 = SUWeight(ψ0)
        alg = SimpleUpdate(; trunc = truncrank(4), force_mpo, purified = (State === InfinitePEPS))
        it = TimeEvolver(ψ0, H, 0.01, 4, alg, env0)
        allocator = it.allocator
        ψ, env, info = saved = timestep(it, ψ0, env0)
        @test (ψ, env, info.ϵ) == PEPSKit.su_iter(ψ0, it.circuit, it.alg, env0)
        snapshot = deepcopy(saved)
        for _ in 2:4
            expected = PEPSKit.su_iter(ψ, it.circuit, it.alg, env)
            ψ, env, info = timestep(it, ψ, env)
            @test (ψ, env, info.ϵ) == expected
            @test saved == snapshot # later scratch reuse must not overwrite earlier outputs
        end
        ψ, env, info = timestep(it, ψ0, env0; iter = 0, t = 0.0)
        @test (ψ, env, info.ϵ) == PEPSKit.su_iter(ψ0, it.circuit, it.alg, env0)
        @test it.allocator === allocator
    end
end

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
        ψ1, env1, info1 = deepcopy(ψ0), deepcopy(env0), nothing
        for iter in 0:(nstep - 1)
            ψ1, env1, info1 = timestep(evolver, ψ1, env1)
        end
        # time_evolve
        ψ2, env2, info2 = time_evolve(ψ0, H, dt, nstep, alg, env0)
        # for-loop syntax
        ## manually reset internal state of evolver
        evolver.state = PEPSKit.SUState(0, 0.0, ψ0, env0)
        ψ3, env3, info3 = nothing, nothing, nothing
        for state in evolver
            ψ3, env3, info3 = state
        end
        # results should be *exactly* the same
        @test ψ1 == ψ2 == ψ3
        @test env1 == env2 == env3
        @test info1 == info2 == info3

        # Restart an exhausted evolver using the public timestep keywords.
        ψreset, envreset, inforeset = timestep(evolver, ψ0, env0; iter = 0, t = 0.0)
        fresh = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
        ψfresh, envfresh, infofresh = timestep(fresh, ψ0, env0)
        @test ψreset == ψfresh
        @test envreset == envfresh
        @test inforeset == infofresh
        @test evolver.state.iter == 1
        @test evolver.state.t == dt

        # Iteration and time offsets affect bookkeeping, not the update.
        ψoffset, envoffset, infooffset = timestep(evolver, ψ0, env0; iter = 7, t = 1.25)
        @test ψoffset == ψfresh
        @test envoffset == envfresh
        @test evolver.state.iter == 8
        @test evolver.state.t == infooffset.t == 1.25 + dt
    end
end

using Test
using Random
using TensorKit
using PEPSKit

Nr, Nc = 2, 2
H = real(heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(Nr, Nc); Jx = 1, Jy = 1, Jz = 1))
Pspace, Vspace = ℂ^2, ℂ^4
ψ0 = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc))
dt, nstep = 0.1, 20
trunc = truncerror(; atol = 1.0e-10) & truncrank(4)

@testset "SimpleUpdate timestep" begin
    alg = SimpleUpdate(; trunc)
    env0 = SUWeight(ψ0)
    # manual timestep
    evolver = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
    ψ1, env1, info1 = deepcopy(ψ0), deepcopy(env0), nothing
    for iter in 0:(nstep - 1)
        ψ1, env1, info1 = timestep(evolver, ψ1, env1)
    end
    # time_evolve
    evolver = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
    ψ2, env2, info2 = time_evolve(evolver)
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
end

using Test
using Random
using TensorKit
using PEPSKit

@testset "SimpleUpdate timestep" begin
    Nr, Nc = 2, 2
    H = real(heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(Nr, Nc); Jx = 1, Jy = 1, Jz = 1))
    Pspace, Vspace = ℂ^2, ℂ^4
    ψ0 = InfinitePEPS(rand, Float64, Pspace, Vspace; unitcell = (Nr, Nc))
    env0 = SUWeight(ψ0)
    alg = SimpleUpdate(; trunc = truncerror(; atol = 1.0e-10) & truncrank(4))
    dt, nstep = 1.0e-2, 50
    # manual timestep
    evolver = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
    ψ1, env1, info1 = deepcopy(ψ0), deepcopy(env0), nothing
    for iter in 0:(nstep - 1)
        ψ1, env1, info1 = timestep(evolver, ψ1, env1; iter)
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
end

@testset "FullUpdate timestep" begin
    Nr, Nc = 2, 2
    H = real(heisenberg_XYZ(ComplexF64, Trivial, InfiniteSquare(Nr, Nc); Jx = 1, Jy = 1, Jz = 1))
    ψ0 = PEPSKit.infinite_temperature_density_matrix(H)
    env0 = CTMRGEnv(ones, Float64, InfinitePEPS(ψ0), ℂ^1)
    opt_alg = FullEnvTruncation(; trunc = truncerror(; atol = 1.0e-10) & truncrank(4))
    ctm_alg = SequentialCTMRG(;
        tol = 1.0e-9, maxiter = 20, verbosity = 2,
        trunc = truncerror(; atol = 1.0e-10) & truncrank(8),
        projector_alg = :fullinfinite,
    )
    alg = FullUpdate(; opt_alg, ctm_alg)
    dt, nstep = 1.0e-2, 20
    # manual timestep
    evolver = TimeEvolver(ψ0, H, dt, nstep, alg, env0)
    ψ1, env1, info1 = deepcopy(ψ0), deepcopy(env0), nothing
    for iter in 0:(nstep - 1)
        ψ1, env1, info1 = timestep(evolver, ψ1, env1; iter)
    end
    # time_evolve
    ψ2, env2, info2 = time_evolve(ψ0, H, dt, nstep, alg, env0)
    # for-loop syntax
    ## manually reset internal state of evolver
    evolver.state = PEPSKit.FUState(0, 0.0, ψ0, env0, true)
    ψ3, env3, info3 = nothing, nothing, nothing
    for state in evolver
        ψ3, env3, info3 = state
    end
    # results should be *exactly* the same
    @test ψ1 == ψ2 == ψ3
    @test env1 == env2 == env3
    @test info1.wts == info2.wts == info3.wts
end

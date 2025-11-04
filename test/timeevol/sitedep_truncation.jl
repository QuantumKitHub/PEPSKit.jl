using Test
using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using PEPSKit: NORTH, EAST

function get_bonddims(peps::InfinitePEPS)
    xdims = collect(dim(domain(t, EAST)) for t in peps.A)
    ydims = collect(dim(domain(t, NORTH)) for t in peps.A)
    return stack([xdims, ydims]; dims = 1)
end

function get_bonddims(wts::SUWeight)
    xdims = collect(dim(space(wt, 1)) for wt in wts[1, :, :])
    ydims = collect(dim(space(wt, 1)) for wt in wts[2, :, :])
    return stack([xdims, ydims]; dims = 1)
end

@testset "Simple update: bipartite 2-site" begin
    Nr, Nc = 2, 2
    H = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx = 1.0, Jy = 1.0, Jz = 1.0))
    Random.seed!(100)
    peps0 = InfinitePEPS(rand, Float64, ℂ^2, ℂ^10; unitcell = (Nr, Nc))
    env0 = SUWeight(peps0)
    normalize!.(peps0.A, Inf)
    # set trunc to be compatible with bipartite structure
    bonddims = stack([[6 4; 4 6], [5 7; 7 5]]; dims = 1)
    trunc = SiteDependentTruncation(collect(truncrank(d) for d in bonddims))
    alg = SimpleUpdate(; ψ0 = peps0, env0, H, dt = 1.0e-2, nstep = 4, trunc, bipartite = true)
    peps, env, = time_evolve(alg)
    @test get_bonddims(peps) == bonddims
    @test get_bonddims(env) == bonddims
    # check bipartite structure is preserved
    for col in 1:2
        cp1 = PEPSKit._next(col, 2)
        @test (
            peps.A[1, col] == peps.A[2, cp1] &&
                env[1, 1, col] == env[1, 2, cp1] &&
                env[2, 1, col] == env[2, 2, cp1]
        )
    end
end

@testset "Simple update: generic 2-site and 3-site" begin
    Nr, Nc = 3, 4
    Random.seed!(100)
    peps0 = InfinitePEPS(rand, Float64, ℂ^2, ℂ^10; unitcell = (Nr, Nc))
    normalize!.(peps0.A, Inf)
    env0 = SUWeight(peps0)
    # Site dependent truncation
    bonddims = rand(2:8, 2, Nr, Nc)
    @show bonddims
    trunc = SiteDependentTruncation(collect(truncrank(d) for d in bonddims))
    # 2-site SU
    H = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx = 1.0, Jy = 1.0, Jz = 1.0))
    alg = SimpleUpdate(; ψ0 = peps0, env0, H, dt = 1.0e-2, nstep = 4, trunc)
    peps, env, = time_evolve(alg)
    @test get_bonddims(peps) == bonddims
    @test get_bonddims(env) == bonddims
    # 3-site SU
    H = real(j1_j2_model(InfiniteSquare(Nr, Nc); J1 = 1.0, J2 = 0.5, sublattice = false))
    alg = SimpleUpdate(; ψ0 = peps0, env0, H, dt = 1.0e-2, nstep = 4, trunc)
    peps, env, = time_evolve(alg)
    @test get_bonddims(peps) == bonddims
    @test get_bonddims(env) == bonddims
end

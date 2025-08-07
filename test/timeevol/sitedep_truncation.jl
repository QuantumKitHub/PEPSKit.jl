using Test
using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using PEPSKit: NORTH, EAST

function get_bonddims(peps::InfinitePEPS)
    xdims = collect(dim(domain(t, EAST)) for t in peps.A)
    ydims = collect(dim(domain(t, NORTH)) for t in peps.A)
    return stack([xdims, ydims]; dims=1)
end

function get_bonddims(wts::SUWeight)
    xdims = collect(dim(space(wt, 1)) for wt in wts[1, :, :])
    ydims = collect(dim(space(wt, 1)) for wt in wts[2, :, :])
    return stack([xdims, ydims]; dims=1)
end

@testset "Simple update: bipartite 2-site" begin
    Nr, Nc = 2, 2
    ham = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx=1.0, Jy=1.0, Jz=1.0))
    Random.seed!(100)
    peps0 = InfinitePEPS(rand, Float64, ℂ^2, ℂ^10; unitcell=(Nr, Nc))
    env0 = SUWeight(peps0)
    normalize!.(peps0.A, Inf)
    # set trscheme to be compatible with bipartite structure
    bonddims = stack([[6 4; 4 6], [5 7; 7 5]]; dims=1)
    trscheme = SiteDependentTruncation(collect(truncdim(d) for d in bonddims))
    alg = SimpleUpdate(1e-2, 1e-14, 4, trscheme)
    peps, env, = simpleupdate(peps0, ham, alg, env0; bipartite=true)
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
    ham = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx=1.0, Jy=1.0, Jz=1.0))
    Random.seed!(100)
    peps0 = InfinitePEPS(rand, Float64, ℂ^2, ℂ^10; unitcell=(Nr, Nc))
    normalize!.(peps0.A, Inf)
    env0 = SUWeight(peps0)
    # Site dependent truncation
    bonddims = rand(2:8, 2, Nr, Nc)
    @show bonddims
    trscheme = SiteDependentTruncation(collect(truncdim(d) for d in bonddims))
    alg = SimpleUpdate(1e-2, 1e-14, 2, trscheme)
    # 2-site SU
    peps, env, = simpleupdate(peps0, ham, alg, env0; bipartite=false)
    @test get_bonddims(peps) == bonddims
    @test get_bonddims(env) == bonddims
    # 3-site SU
    peps, env, = simpleupdate(peps0, ham, alg, env0; bipartite=false, force_3site=true)
    @test get_bonddims(peps) == bonddims
    @test get_bonddims(env) == bonddims
end

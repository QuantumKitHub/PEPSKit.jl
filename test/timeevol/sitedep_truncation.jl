using Test
using LinearAlgebra
using Random
using TensorKit
using PEPSKit
using PEPSKit: NORTH, EAST

function get_bonddims(wpeps::InfiniteWeightPEPS)
    xdims = collect(dim(domain(t, EAST)) for t in wpeps.vertices)
    ydims = collect(dim(domain(t, NORTH)) for t in wpeps.vertices)
    return stack([xdims, ydims]; dims=1)
end

@testset "Simple update: bipartite 2-site" begin
    Nr, Nc = 2, 2
    ham = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx=1.0, Jy=1.0, Jz=1.0))
    Random.seed!(100)
    wpeps0 = InfiniteWeightPEPS(rand, Float64, ℂ^2, ℂ^10; unitcell=(Nr, Nc))
    normalize!.(wpeps0.vertices, Inf)
    # set trscheme to be compatible with bipartite structure
    bonddims = stack([[6 4; 4 6], [5 7; 7 5]]; dims=1)
    trscheme = SiteDependentTruncation(collect(truncdim(d) for d in bonddims))
    alg = SimpleUpdate(1e-2, 1e-14, 4, trscheme)
    wpeps, = simpleupdate(wpeps0, ham, alg; bipartite=true)
    @test get_bonddims(wpeps) == bonddims
    # check bipartite structure is preserved
    for col in 1:2
        cp1 = PEPSKit._next(col, 2)
        @test (
            wpeps.vertices[1, col] == wpeps.vertices[2, cp1] &&
            wpeps.weights[1, 1, col] == wpeps.weights[1, 2, cp1] &&
            wpeps.weights[2, 1, col] == wpeps.weights[2, 2, cp1]
        )
    end
end

@testset "Simple update: generic 2-site and 3-site" begin
    Nr, Nc = 3, 4
    ham = real(heisenberg_XYZ(InfiniteSquare(Nr, Nc); Jx=1.0, Jy=1.0, Jz=1.0))
    Random.seed!(100)
    wpeps0 = InfiniteWeightPEPS(rand, Float64, ℂ^2, ℂ^10; unitcell=(Nr, Nc))
    normalize!.(wpeps0.vertices, Inf)
    # Site dependent truncation
    bonddims = rand(2:8, 2, Nr, Nc)
    @show bonddims
    trscheme = SiteDependentTruncation(collect(truncdim(d) for d in bonddims))
    alg = SimpleUpdate(1e-2, 1e-14, 2, trscheme)
    # 2-site SU
    wpeps, = simpleupdate(wpeps0, ham, alg; bipartite=false)
    @test get_bonddims(wpeps) == bonddims
    # 3-site SU
    wpeps, = simpleupdate(wpeps0, ham, alg; bipartite=false, force_3site=true)
    @test get_bonddims(wpeps) == bonddims
end

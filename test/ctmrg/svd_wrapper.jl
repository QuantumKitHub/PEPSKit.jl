using Test
using Random
using LinearAlgebra
using TensorKit
using KrylovKit
using ChainRulesCore, Zygote
using PEPSKit

# Gauge-invariant loss function
function lossfun(A, alg, R=TensorMap(randn, space(A)); trunc=notrunc())
    U, _, V, = tsvd(A; trunc, alg)
    return real(dot(R, U * V))  # Overlap with random tensor R is gauge-invariant and differentiable, also for m≠n
end

m, n = 20, 30
dtype = ComplexF64
χ = 12
trunc = truncdim(χ)
# lorentz_broad = 1e-12
adjoint_tol = 1e-16
rtol = 1e-9
r = TensorMap(randn, dtype, ℂ^m ← ℂ^n)
R = TensorMap(randn, space(r))

@testset "Non-truncacted SVD" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, FullSVD(), R), r)
    l_oldsvd, g_oldsvd = withgradient(A -> lossfun(A, OldSVD(), R), r)
    l_itersvd, g_itersvd = withgradient(
        A -> lossfun(A, IterSVD(; howmany=min(m, n), adjoint_tol), R), r
    )

    @test l_oldsvd ≈ l_itersvd ≈ l_fullsvd 
    @test norm(g_fullsvd[1] - g_oldsvd[1]) / norm(g_fullsvd[1]) < rtol
    @test norm(g_fullsvd[1] - g_itersvd[1]) / norm(g_fullsvd[1]) < rtol
end

@testset "Truncated SVD with χ=$χ" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, FullSVD(), R; trunc), r)
    l_oldsvd, g_oldsvd = withgradient(A -> lossfun(A, OldSVD(), R; trunc), r)
    l_itersvd, g_itersvd = withgradient(
        A -> lossfun(A, IterSVD(; howmany=χ, adjoint_tol), R; trunc), r
    )

    @test l_oldsvd ≈ l_itersvd ≈ l_fullsvd 
    @test norm(g_fullsvd[1] - g_oldsvd[1]) / norm(g_fullsvd[1]) > rtol
    @test norm(g_fullsvd[1] - g_itersvd[1]) / norm(g_fullsvd[1]) < rtol
end

# @testset "Truncated SVD with χ=$χ and ε=$lorentz_broad broadening" begin
#     l_fullsvd, g_fullsvd = withgradient(
#         A -> lossfun(A, FullSVD(; lorentz_broad, R; trunc), r
#     )
#     l_oldsvd, g_oldsvd = withgradient(A -> lossfun(A, OldSVD(; lorentz_broad), R; trunc), r)
#     l_itersvd, g_itersvd = withgradient(
#         A -> lossfun(A, IterSVD(; howmany=χ, lorentz_broad), R; trunc), r
#     )

#     @test l_oldsvd ≈ l_itersvd ≈ l_fullsvd 
#     @test norm(g_fullsvd[1] - g_oldsvd[1]) / norm(g_fullsvd[1]) > rtol
#     @test norm(g_fullsvd[1] - g_itersvd[1]) / norm(g_fullsvd[1]) < rtol
# end

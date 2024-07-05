using Test
using Random
using LinearAlgebra
using TensorKit
using KrylovKit
using ChainRulesCore, Zygote
using PEPSKit

# Gauge-invariant loss function
function lossfun(A, alg, R=TensorMap(randn, space(A)), trunc=notrunc())
    U, _, V, = PEPSKit.tsvd(A, alg; trunc)
    return real(dot(R, U * V))  # Overlap with random tensor R is gauge-invariant and differentiable, also for m≠n
end

m, n = 20, 30
dtype = ComplexF64
χ = 12
trunc = truncspace(ℂ^χ)
# lorentz_broad = 1e-12
adjoint_tol = 1e-14  # Don't make this too small, g_itersvd will be weird
rtol = 1e-9
r = TensorMap(randn, dtype, ℂ^m ← ℂ^n)
R = TensorMap(randn, space(r))

@testset "Non-truncacted SVD" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, TensorKit.SVD(), R), r)
    l_oldsvd, g_oldsvd = withgradient(A -> lossfun(A, OldSVD(), R), r)
    l_itersvd, g_itersvd = withgradient(
        A -> lossfun(A, IterSVD(; alg_rrule=GMRES(; tol=adjoint_tol)), R, truncspace(ℂ^min(m, n))), r
    )

    @test l_oldsvd ≈ l_itersvd ≈ l_fullsvd
    @test norm(g_fullsvd[1] - g_oldsvd[1]) / norm(g_fullsvd[1]) < rtol
    @test norm(g_fullsvd[1] - g_itersvd[1]) / norm(g_fullsvd[1]) < rtol
end

@testset "Truncated SVD with χ=$χ" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, TensorKit.SVD(), R, trunc), r)
    l_oldsvd, g_oldsvd = withgradient(A -> lossfun(A, OldSVD(), R, trunc), r)
    l_itersvd, g_itersvd = withgradient(
        A -> lossfun(A, IterSVD(; alg_rrule=GMRES(; tol=adjoint_tol)), R, trunc), r
    )

    @test l_oldsvd ≈ l_itersvd ≈ l_fullsvd
    @test norm(g_fullsvd[1] - g_oldsvd[1]) / norm(g_fullsvd[1]) > rtol
    @test norm(g_fullsvd[1] - g_itersvd[1]) / norm(g_fullsvd[1]) < rtol
end

# TODO: Add when Lorentzian broadening is implemented
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

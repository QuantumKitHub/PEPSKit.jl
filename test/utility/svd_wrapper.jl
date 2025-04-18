using Test
using Random
using LinearAlgebra
using TensorKit
using ChainRulesCore, Zygote
using Accessors
using PEPSKit
# using PEPSKit: HalfInfiniteEnv

# Gauge-invariant loss function
function lossfun(A, alg, R=randn(space(A)), trunc=notrunc())
    U, S, V, = PEPSKit.tsvd(A, alg; trunc)
    return real(dot(R, U * V)) + dot(S, S)  # Overlap with random tensor R is gauge-invariant and differentiable, also for m≠n
end

m, n = 20, 30
dtype = ComplexF64
χ = 12
trunc = truncspace(ℂ^χ)
rtol = 1e-9
Random.seed!(123456789)
r = randn(dtype, ℂ^m, ℂ^n)
R = randn(space(r))
broadenings = [10.0^k for k in -16:-4]

full_alg = SVDAdjoint(; rrule_alg=(; alg=:tsvd), broadening=0)
iter_alg = SVDAdjoint(; fwd_alg=(; alg=:iterative))

@testset "Non-truncacted SVD" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, full_alg, R), r)
    l_itersvd, g_itersvd = withgradient(A -> lossfun(A, iter_alg, R), r)

    @test l_itersvd ≈ l_fullsvd
    @test g_fullsvd[1] ≈ g_itersvd[1] rtol = rtol
end

@testset "Truncated SVD with χ=$χ" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, full_alg, R, trunc), r)
    l_itersvd, g_itersvd = withgradient(A -> lossfun(A, iter_alg, R, trunc), r)

    @test l_itersvd ≈ l_fullsvd
    @test g_fullsvd[1] ≈ g_itersvd[1] rtol = rtol
end

@testset "Truncated SVD with χ=$χ and ε=$ε broadening" for ε in broadenings
    broadened_alg = @set full_alg.broadening = ε
    l_unbroadened, g_unbroadened = withgradient(A -> lossfun(A, full_alg, R, trunc), r)
    l_broadened, g_broadened = withgradient(A -> lossfun(A, broadened_alg, R, trunc), r)

    @test l_unbroadened ≈ l_broadened
    @test 1e1 * norm(g_broadened[1]) * ε > norm(g_unbroadened[1] - g_broadened[1]) > ε
end

symm_m, symm_n = 18, 24
symm_space = Z2Space(0 => symm_m, 1 => symm_n)
symm_trspace = truncspace(Z2Space(0 => symm_m ÷ 2, 1 => symm_n ÷ 3))
symm_r = randn(dtype, symm_space, symm_space)
symm_R = randn(dtype, space(symm_r))

@testset "IterSVD of symmetric tensors" begin
    l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, full_alg, symm_R), symm_r)
    l_itersvd, g_itersvd = withgradient(A -> lossfun(A, iter_alg, symm_R), symm_r)
    @test l_itersvd ≈ l_fullsvd
    @test g_fullsvd[1] ≈ g_itersvd[1] rtol = rtol

    l_fullsvd_tr, g_fullsvd_tr = withgradient(
        A -> lossfun(A, full_alg, symm_R, symm_trspace), symm_r
    )
    l_itersvd_tr, g_itersvd_tr = withgradient(
        A -> lossfun(A, iter_alg, symm_R, symm_trspace), symm_r
    )
    @test l_itersvd_tr ≈ l_fullsvd_tr
    @test g_fullsvd_tr[1] ≈ g_itersvd_tr[1] rtol = rtol

    iter_alg_fallback = @set iter_alg.fwd_alg.fallback_threshold = 0.4  # Do dense SVD in one block, sparse SVD in the other
    l_itersvd_fb, g_itersvd_fb = withgradient(
        A -> lossfun(A, iter_alg_fallback, symm_R, symm_trspace), symm_r
    )
    @test l_itersvd_fb ≈ l_fullsvd_tr
    @test g_fullsvd_tr[1] ≈ g_itersvd_fb[1] rtol = rtol
end

@testset "Truncated symmetric SVD with χ=$χ and ε=$ε broadening" for ε in broadenings
    broadened_alg = @set full_alg.broadening = ε
    l_unbroadened, g_unbroadened = withgradient(
        A -> lossfun(A, full_alg, symm_R, symm_trspace), symm_r
    )
    l_broadened, g_broadened = withgradient(
        A -> lossfun(A, broadened_alg, symm_R, symm_trspace), symm_r
    )

    @test l_unbroadened ≈ l_broadened
    @test 1e1 * norm(g_broadened[1]) * ε > norm(g_unbroadened[1] - g_broadened[1]) > ε
end

# TODO: Add when IterSVD is implemented for HalfInfiniteEnv
# χbond = 2
# χenv = 6
# ctm_alg = CTMRG(; tol=1e-10, verbosity=2, svd_alg=SVDAdjoint())
# Random.seed!(91283219347)
# H = heisenberg_XYZ(InfiniteSquare())
# psi = InfinitePEPS(2, χbond)
# env = leading_boundary(CTMRGEnv(psi, ComplexSpace(χenv)), psi, ctm_alg);
# hienv = HalfInfiniteEnv(
#     env.corners[1],
#     env.corners[2],
#     env.edges[4],
#     env.edges[1],
#     env.edges[1],
#     env.edges[2],
#     psi[1],
#     psi[1],
#     psi[1],
#     psi[1],
# )
# hienv_dense = hienv()
# env_R = randn(space(hienv))

# PEPSKit.tsvd!(hienv, iter_alg)

# @testset "IterSVD with HalfInfiniteEnv function handle" begin
#     # Equivalence of dense and sparse contractions
#     x₀ = PEPSKit.random_start_vector(hienv)
#     x′ = hienv(x₀, Val(false))
#     x″ = hienv(x′, Val(true))
#     x‴ = hienv(x″, Val(false))

#     a = hienv_dense * x₀
#     b = hienv_dense' * a
#     c = hienv_dense * b
#     @test a ≈ x′
#     @test b ≈ x″
#     @test c ≈ x‴

#     # l_fullsvd, g_fullsvd = withgradient(A -> lossfun(A, full_alg, env_R), hienv_dense)
#     # l_itersvd, g_itersvd = withgradient(A -> lossfun(A, iter_alg, env_R), hienv)
#     # @test l_itersvd ≈ l_fullsvd
#     # @test g_fullsvd[1] ≈ g_itersvd[1] rtol = rtol
# end

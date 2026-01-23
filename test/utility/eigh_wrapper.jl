using Test
using Random
using LinearAlgebra
using TensorKit
using ChainRulesCore, Zygote
using Accessors
using FixedPointAD

# Gauge-invariant loss function
function lossfun(A, alg, R = randn(space(A)), trunc = notrunc())
    D, V, = eigh_trunc(A, alg; trunc)
    return real(dot(R, V * V')) + dot(D, D)  # Overlap with random tensor R is gauge-invariant and differentiable
end

dtype = ComplexF64
n = 20
χ = 10
trunc = truncspace(ℂ^χ)
rtol = 1.0e-9
Random.seed!(123456789)
r = randn(dtype, ℂ^n, ℂ^n)
r = 0.5 * (r + r') # make r Hermitian
R = randn(space(r))
R = 0.5 * (R + R')

full_alg = EighAdjoint(; fwd_alg = (; alg = :qriteration), rrule_alg = (; alg = :trunc))
iter_alg = EighAdjoint(; fwd_alg = (; alg = :lanczos), rrule_alg = (; alg = :trunc))

@testset "Non-truncacted eigh" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, R), r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, R), r)

    @test l_iter ≈ l_full
    @test g_full[1] ≈ g_iter[1] rtol = rtol
end

@testset "Truncated eigh with χ=$χ" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, R, trunc), r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, R, trunc), r)

    @test l_iter ≈ l_full
    @test g_full[1] ≈ g_iter[1] rtol = rtol
end

symm_m, symm_n = 18, 24
symm_space = Z2Space(0 => symm_m, 1 => symm_n)
symm_trspace = truncspace(Z2Space(0 => symm_m ÷ 2, 1 => symm_n ÷ 3))
symm_r = randn(dtype, symm_space, symm_space)
symm_r = 0.5 * (symm_r + symm_r')
symm_R = randn(dtype, space(symm_r))
symm_R = 0.5 * (symm_R + symm_R')

@testset "IterEig of symmetric tensors" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, symm_R), symm_r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, symm_R), symm_r)
    @test l_iter ≈ l_full
    @test g_full[1] ≈ g_iter[1] rtol = rtol

    l_full_tr, g_full_tr = withgradient(
        A -> lossfun(A, full_alg, symm_R, symm_trspace), symm_r
    )
    l_iter_tr, g_iter_tr = withgradient(
        A -> lossfun(A, iter_alg, symm_R, symm_trspace), symm_r
    )
    @test l_iter_tr ≈ l_full_tr
    @test g_full_tr[1] ≈ g_iter_tr[1] rtol = rtol

    iter_alg_fallback = @set iter_alg.fwd_alg.fallback_threshold = 0.4  # Do dense decomposition in one block, sparse one in the other
    l_iter_fb, g_iter_fb = withgradient(
        A -> lossfun(A, iter_alg_fallback, symm_R, symm_trspace), symm_r
    )
    @test l_iter_fb ≈ l_full_tr
    @test g_full_tr[1] ≈ g_iter_fb[1] rtol = rtol
end

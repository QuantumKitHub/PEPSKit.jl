
using Test
using Random
using LinearAlgebra
using TensorKit
using Mooncake 
using Accessors
using PEPSKit

using MatrixAlgebraKit: TruncatedAlgorithm, diagview

# Gauge-invariant loss function
function lossfun(A, alg, R = randn(space(A)), trunc = notrunc())
    alg = @set alg.fwd_alg = TruncatedAlgorithm(alg.fwd_alg, trunc)
    D, V, = eigh_trunc(project_hermitian(A), alg)
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

full_alg = EighAdjoint(; fwd_alg = (; alg = :QRIteration), rrule_alg = (; alg = :FullPullback))
trunc_alg = EighAdjoint(; fwd_alg = (; alg = :QRIteration), rrule_alg = (; alg = :TruncPullback))
iter_alg = EighAdjoint(; fwd_alg = (; alg = :Lanczos), rrule_alg = (; alg = :TruncPullback))

@testset "Non-truncated eigh" begin
    full_lossfun = A -> lossfun(A, full_alg, R)
    trunc_lossfun = A -> lossfun(A, trunc_alg, R)
    iter_lossfun = A -> lossfun(A, iter_alg, R)
    
    full_rrule = Mooncake.build_rrule(full_lossfun, r)
    trunc_rrule = Mooncake.build_rrule(trunc_lossfun, r)
    iter_rrule = Mooncake.build_rrule(iter_lossfun, r)

    l_full, g_full = Mooncake.value_and_gradient!!(full_rrule, full_lossfun, r)
    l_trunc, g_trunc = Mooncake.value_and_gradient!!(trunc_rrule, trunc_lossfun, r)
    l_iter, g_iter = Mooncake.value_and_gradient!!(iter_rrule, iter_lossfun, r)

    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[2] ≈ g_trunc[2] rtol = rtol
    @test g_full[2] ≈ g_iter[2] rtol = rtol
    @test g_trunc[2] ≈ g_iter[2] rtol = rtol
end

@testset "Truncated eigh with χ=$χ" begin
    full_lossfun = A -> lossfun(A, full_alg, R, trunc)
    trunc_lossfun = A -> lossfun(A, trunc_alg, R, trunc)
    iter_lossfun = A -> lossfun(A, iter_alg, R, trunc)

    full_rrule = Mooncake.build_rrule(full_lossfun, r)
    trunc_rrule = Mooncake.build_rrule(trunc_lossfun, r)
    iter_rrule = Mooncake.build_rrule(iter_lossfun, r)

    l_full, g_full = Mooncake.value_and_gradient!!(full_rrule, full_lossfun, r)
    l_trunc, g_trunc = Mooncake.value_and_gradient!!(trunc_rrule, trunc_lossfun, r)
    l_iter, g_iter = Mooncake.value_and_gradient!!(iter_rrule, iter_lossfun, r)

    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[2] ≈ g_trunc[2] rtol = rtol
    @test g_full[2] ≈ g_iter[2] rtol = rtol
    @test g_trunc[2] ≈ g_iter[2] rtol = rtol
end

@testset "Truncated eigh broadening for $(alg.rrule_alg)" for alg in [full_alg, trunc_alg]
    d, v = eigh_full(r)
    d.data[1:2:n] .= d.data[2:2:n] # make every eigenvalue two-fold degenerate
    r_degen = v * d * v'

    no_broadening_no_cutoff_alg = @set alg.rrule_alg.degeneracy_atol = 1.0e-30
    small_broadening_alg = @set alg.rrule_alg.degeneracy_atol = 1.0e-13

    only_lossfun = A -> lossfun(A, alg, R, trunc)
    no_broadening_lossfun = A -> lossfun(A, no_broadening_no_cutoff_alg, R, trunc)
    small_broadening_lossfun = A -> lossfun(A, small_broadening_alg, R, trunc)

    only_rrule = Mooncake.build_rrule(only_lossfun, r_degen)
    no_broadening_rrule = Mooncake.build_rrule(no_broadening_lossfun, r_degen)
    small_broadening_rrule = Mooncake.build_rrule(small_broadening_lossfun, r_degen)

    l_only_cutoff, g_only_cutoff = Mooncake.value_and_gradient!!(only_rrule, only_lossfun, r_degen) # cutoff sets degenerate difference to zero
    l_no_broadening_no_cutoff, g_no_broadening_no_cutoff = Mooncake.value_and_gradient!!( # degenerate singular value differences lead to divergent contributions
        no_broadening_rrule, no_broadening_lossfun, r_degen,
    )
    l_small_broadening, g_small_broadening = Mooncake.value_and_gradient!!( # broadening smoothens divergent contributions
        small_broadening_rrule, small_broadening_lossfun, r_degen,
    )

    @test l_only_cutoff ≈ l_no_broadening_no_cutoff ≈ l_small_broadening
    @test norm(g_no_broadening_no_cutoff[2] - g_small_broadening[2]) > 1.0e-2 # divergences mess up the gradient
    @test g_only_cutoff[2] ≈ g_small_broadening[2] rtol = rtol # cutoff and broadening have similar effect
end

symm_m, symm_n = 18, 24
symm_space = Z2Space(0 => symm_m, 1 => symm_n)
symm_trspace = truncspace(Z2Space(0 => symm_m ÷ 2, 1 => symm_n ÷ 3))
symm_r = randn(dtype, symm_space, symm_space)
symm_r = 0.5 * (symm_r + symm_r')
symm_R = randn(dtype, space(symm_r))
symm_R = 0.5 * (symm_R + symm_R')

@testset "IterEig of symmetric tensors" begin
    full_lossfun = A -> lossfun(A, full_alg, symm_R)
    trunc_lossfun = A -> lossfun(A, trunc_alg, symm_R)
    iter_lossfun = A -> lossfun(A, iter_alg, symm_R)

    full_rrule = Mooncake.build_rrule(full_lossfun, symm_r)
    trunc_rrule = Mooncake.build_rrule(trunc_lossfun, symm_r)
    iter_rrule = Mooncake.build_rrule(iter_lossfun, symm_r)

    l_full, g_full = Mooncake.value_and_gradient!!(full_rrule, full_lossfun, symm_r)
    l_trunc, g_trunc = Mooncake.value_and_gradient!!(trunc_rrule, trunc_lossfun, symm_r)
    l_iter, g_iter = Mooncake.value_and_gradient!!(iter_rrule, iter_lossfun, symm_r)

    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[2] ≈ g_trunc[2] rtol = rtol
    @test g_full[2] ≈ g_iter[2] rtol = rtol
    @test g_trunc[2] ≈ g_iter[2] rtol = rtol

    full_lossfun = A -> lossfun(A, full_alg, symm_R, symm_trspace)
    trunc_lossfun = A -> lossfun(A, trunc_alg, symm_R, symm_trspace)
    iter_lossfun = A -> lossfun(A, iter_alg, symm_R, symm_trspace)

    full_rrule = Mooncake.build_rrule(full_lossfun, symm_r)
    trunc_rrule = Mooncake.build_rrule(trunc_lossfun, symm_r)
    iter_rrule = Mooncake.build_rrule(iter_lossfun, symm_r)

    l_full_tr, g_full_tr = Mooncake.value_and_gradient!!(full_rrule, full_lossfun, symm_r)
    l_trunc_tr, g_trunc_tr = Mooncake.value_and_gradient!!(trunc_rrule, trunc_lossfun, symm_r)
    l_iter_tr, g_iter_tr = Mooncake.value_and_gradient!!(iter_rrule, iter_lossfun, symm_r)
    @test l_full_tr ≈ l_trunc_tr ≈ l_iter_tr
    @test g_full_tr[2] ≈ g_trunc_tr[2] rtol = rtol
    @test g_full_tr[2] ≈ g_iter_tr[2] rtol = rtol
    @test g_trunc_tr[2] ≈ g_iter_tr[2] rtol = rtol

    iter_alg_fallback = @set iter_alg.fwd_alg.fallback_threshold = 0.4  # Do dense decomposition in one block, sparse one in the other
    fb_lossfun = A -> lossfun(A, iter_alg_fallback, symm_R, symm_trspace)
    fb_rrule = Mooncake.build_rrule(fb_lossfun, symm_r)
    l_iter_fb, g_iter_fb = Mooncake.value_and_gradient!!(fb_rrule, fb_lossfun, symm_r)
    @test l_iter_fb ≈ l_trunc_tr ≈ l_full_tr
    @test g_full_tr[2] ≈ g_iter_fb[2] rtol = rtol
    @test g_trunc_tr[2] ≈ g_iter_fb[2] rtol = rtol
end
#=
@testset "Truncated symmetric eigh broadening for $(alg.rrule_alg)" for alg in [full_alg, trunc_alg]
    d, v = eigh_full(symm_r)
    # make every singular value in the 0-sector three-fold degenerate
    b0 = diagview(block(d, Z2Irrep(0)))
    b0[1:3:symm_m] .= b0[3:3:symm_m]
    b0[2:3:symm_m] .= b0[3:3:symm_m]
    # make every singular value in the 1-sector two-fold degenerate
    b1 = diagview(block(d, Z2Irrep(1)))
    b1[1:2:symm_n] .= b1[2:2:symm_n]
    symm_r_degen = v * d * v'

    no_broadening_no_cutoff_alg = @set alg.rrule_alg.degeneracy_atol = 1.0e-30
    small_broadening_alg = @set alg.rrule_alg.degeneracy_atol = 1.0e-13

    l_only_cutoff, g_only_cutoff = withgradient(
        A -> lossfun(A, alg, symm_R, symm_trspace), symm_r_degen
    ) # cutoff sets degenerate difference to zero
    l_no_broadening_no_cutoff, g_no_broadening_no_cutoff = withgradient( # degenerate singular value differences lead to divergent contributions
        A -> lossfun(A, no_broadening_no_cutoff_alg, symm_R, symm_trspace),
        symm_r_degen,
    )
    l_small_broadening, g_small_broadening = withgradient( # broadening smoothens divergent contributions
        A -> lossfun(A, small_broadening_alg, symm_R, symm_trspace),
        symm_r_degen,
    )

    @test l_only_cutoff ≈ l_no_broadening_no_cutoff ≈ l_small_broadening
    @test norm(g_no_broadening_no_cutoff[1] - g_small_broadening[1]) > 1.0e-2 # divergences mess up the gradient
    @test g_only_cutoff[1] ≈ g_small_broadening[1] rtol = rtol # cutoff and broadening have similar effect
end=#

using Test
using Random
using LinearAlgebra
using TensorKit
using ChainRulesCore, Zygote
using Accessors
using PEPSKit
# using PEPSKit: HalfInfiniteEnv

# Gauge-invariant loss function
function lossfun(A, alg, R = randn(space(A)), trunc = notrunc())
    U, S, V, = svd_trunc(A, alg; trunc)
    return real(dot(R, U * V)) + dot(S, S)  # Overlap with random tensor R is gauge-invariant and differentiable, also for m≠n
end

dtype = ComplexF64
m, n = 20, 30
χ = 12
trunc = truncspace(ℂ^χ)
rtol = 1.0e-9
Random.seed!(123456789)
r = randn(dtype, ℂ^m, ℂ^n)
R = randn(space(r))

full_alg = SVDAdjoint(; rrule_alg = (; alg = :full, degeneracy_atol = 1.0e-13))
trunc_alg = SVDAdjoint(; rrule_alg = (; alg = :trunc, degeneracy_atol = 1.0e-13))
iter_alg = SVDAdjoint(; fwd_alg = (; alg = :iterative))

@testset "Non-truncated SVD" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, R), r)
    l_trunc, g_trunc = withgradient(A -> lossfun(A, trunc_alg, R), r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, R), r)

    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[1] ≈ g_trunc[1] rtol = rtol
    @test g_full[1] ≈ g_iter[1] rtol = rtol
    @test g_trunc[1] ≈ g_iter[1] rtol = rtol
end

@testset "Truncated SVD with χ=$χ" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, R, trunc), r)
    l_trunc, g_trunc = withgradient(A -> lossfun(A, trunc_alg, R, trunc), r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, R, trunc), r)

    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[1] ≈ g_trunc[1] rtol = rtol
    @test g_full[1] ≈ g_iter[1] rtol = rtol
    @test g_trunc[1] ≈ g_iter[1] rtol = rtol
end

@testset "Truncated SVD broadening for $(alg.rrule_alg)" for alg in [full_alg, trunc_alg]
    u, s, v, = svd_compact(r)
    s.data[1:2:m] .= s.data[2:2:m] # make every singular value two-fold degenerate
    r_degen = u * s * v

    no_broadening_no_cutoff_alg = @set full_alg.rrule_alg.degeneracy_atol = 1.0e-30
    small_broadening_alg = @set full_alg.rrule_alg.degeneracy_atol = 1.0e-13

    l_only_cutoff, g_only_cutoff = withgradient(
        A -> lossfun(A, full_alg, R, trunc), r_degen
    ) # cutoff sets degenerate difference to zero
    l_no_broadening_no_cutoff, g_no_broadening_no_cutoff = withgradient( # degenerate singular value differences lead to divergent contributions
        A -> lossfun(A, no_broadening_no_cutoff_alg, R, trunc), r_degen,
    )
    l_small_broadening, g_small_broadening = withgradient( # broadening smoothens divergent contributions
        A -> lossfun(A, small_broadening_alg, R, trunc), r_degen,
    )

    @test l_only_cutoff ≈ l_no_broadening_no_cutoff ≈ l_small_broadening
    @test norm(g_no_broadening_no_cutoff[1] - g_small_broadening[1]) > 1.0e-2 # divergences mess up the gradient
    @test g_only_cutoff[1] ≈ g_small_broadening[1] rtol = rtol # cutoff and broadening have similar effect
end

symm_m, symm_n = 18, 24
symm_space = Z2Space(0 => symm_m, 1 => symm_n)
symm_trspace = truncspace(Z2Space(0 => symm_m ÷ 2, 1 => symm_n ÷ 3))
symm_r = randn(dtype, symm_space, symm_space)
symm_R = randn(dtype, space(symm_r))

@testset "IterSVD of symmetric tensors" begin
    l_full, g_full = withgradient(A -> lossfun(A, full_alg, symm_R), symm_r)
    l_trunc, g_trunc = withgradient(A -> lossfun(A, trunc_alg, symm_R), symm_r)
    l_iter, g_iter = withgradient(A -> lossfun(A, iter_alg, symm_R), symm_r)
    @test l_full ≈ l_trunc ≈ l_iter
    @test g_full[1] ≈ g_trunc[1] rtol = rtol
    @test g_full[1] ≈ g_iter[1] rtol = rtol
    @test g_trunc[1] ≈ g_iter[1] rtol = rtol

    l_full_tr, g_full_tr = withgradient(
        A -> lossfun(A, full_alg, symm_R, symm_trspace), symm_r
    )
    l_trunc_tr, g_trunc_tr = withgradient(
        A -> lossfun(A, trunc_alg, symm_R, symm_trspace), symm_r
    )
    l_iter_tr, g_iter_tr = withgradient(
        A -> lossfun(A, iter_alg, symm_R, symm_trspace), symm_r
    )
    @test l_full_tr ≈ l_trunc_tr ≈ l_iter_tr
    @test g_full_tr[1] ≈ g_trunc_tr[1] rtol = rtol
    @test g_full_tr[1] ≈ g_iter_tr[1] rtol = rtol
    @test g_trunc_tr[1] ≈ g_iter_tr[1] rtol = rtol

    iter_alg_fallback = @set iter_alg.fwd_alg.fallback_threshold = 0.4  # Do dense decomposition in one block, sparse one in the other
    l_iter_fb, g_iter_fb = withgradient(
        A -> lossfun(A, iter_alg_fallback, symm_R, symm_trspace), symm_r
    )
    @test l_iter_fb ≈ l_trunc_tr ≈ l_full_tr
    @test g_full_tr[1] ≈ g_iter_fb[1] rtol = rtol
    @test g_trunc_tr[1] ≈ g_iter_fb[1] rtol = rtol
end

@testset "Truncated symmetric SVD broadening for $(alg.rrule_alg)" for alg in [full_alg, trunc_alg]
    u, s, v, = svd_compact(symm_r)
    s.data[1:2:m] .= s.data[2:2:m] # make every singular value two-fold degenerate
    symm_r_degen = u * s * v

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
end

# TODO: Add when IterSVD is implemented for HalfInfiniteEnv
# χbond = 2
# χenv = 6
# ctm_alg = CTMRG(; tol=1e-10, verbosity=2, svd_alg=SVDAdjoint())
# Random.seed!(91283219347)
# H = heisenberg_XYZ(InfiniteSquare())
# psi = InfinitePEPS(ComplexSpace(2), ComplexSpace(χbond))
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

# svd_trunc!(hienv, iter_alg)

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

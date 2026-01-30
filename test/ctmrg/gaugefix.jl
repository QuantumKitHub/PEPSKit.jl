using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iteration, calc_elementwise_convergence
using PEPSKit: ScramblingEnvGauge, ScramblingEnvGaugeC4v
using PEPSKit: peps_normalize

spacetypes = [ComplexSpace, Z2Space]
scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
ctmrg_algs_asymm = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs_asymm = [:halfinfinite, :fullinfinite]
projector_algs_c4v = [:c4v_eigh] #, :c4v_qr]
gauge_algs_asymm = [ScramblingEnvGauge()]
gauge_algs_c4v = [ScramblingEnvGaugeC4v()]
tol = 1.0e-6  # large tol due to χ=6
χ = 6
atol = 1.0e-4

function _pre_converge_env(
        ::Type{T}, alg, physical_space, peps_space, env_space, unitcell;
        seed = 985293852935829
    ) where {T}
    Random.seed!(seed)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(rand, T, physical_space, peps_space; unitcell)
    alg == :c4v && (psi = peps_normalize(symmetrize!(psi, RotateReflect())))
    env₀ = if alg == :c4v
        initialize_singlet_c4v_env(T, psi, env_space)
    else
        CTMRGEnv(psi, env_space)
    end
    env_conv, = leading_boundary(env₀, psi; alg, tol)
    return env_conv, psi
end

# pre-converge CTMRG environments with given spacetype, scalartype and unit cell
preconv = Dict()
for (S, T, unitcell) in Iterators.product(spacetypes, scalartypes, unitcells)
    if S == ComplexSpace
        result = _pre_converge_env(T, :sequential, S(2), S(2), S(χ), unitcell)
    elseif S == Z2Space
        result = _pre_converge_env(
            T, :sequential, S(0 => 1, 1 => 1), S(0 => 1, 1 => 1),
            S(0 => χ ÷ 2, 1 => χ ÷ 2), unitcell
        )
    end
    push!(preconv, (S, T, unitcell) => result)
end
preconv_c4v = Dict()
for (S, T) in Iterators.product(spacetypes, scalartypes)
    if S == ComplexSpace
        result = _pre_converge_env(T, :c4v, S(2), S(2), S(χ), (1, 1))
    elseif S == Z2Space
        result = _pre_converge_env(
            T, :c4v, S(0 => 1, 1 => 1), S(0 => 1, 1 => 1), S(0 => χ ÷ 2, 1 => χ ÷ 2), (1, 1)
        )
    end
    push!(preconv_c4v, (S, T) => result)
end

# asymmetric CTMRG
@testset "($S) - ($T) - ($unitcell) - ($ctmrg_alg) - ($projector_alg) - ($gauge_alg)" for (
        S, T, unitcell, ctmrg_alg, projector_alg, gauge_alg,
    ) in Iterators.product(
        spacetypes, scalartypes, unitcells, ctmrg_algs_asymm, projector_algs_asymm, gauge_algs_asymm
    )
    alg = ctmrg_alg(; tol, projector_alg)
    env_pre, psi = preconv[(S, T, unitcell)]
    n = InfiniteSquareNetwork(psi)
    env, = leading_boundary(env_pre, psi, alg)
    env′, = ctmrg_iteration(n, env, alg)
    env_fixed, = gauge_fix(env′, gauge_alg, env)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end

# C4v CTMRG
@testset "($S) - ($T) - ($projector_alg) - ($gauge_alg)" for (
        S, T, projector_alg, gauge_alg,
    ) in Iterators.product(
        spacetypes, scalartypes, projector_algs_c4v, gauge_algs_c4v
    )
    alg = C4vCTMRG(; tol, projector_alg)
    env_pre, psi = preconv_c4v[(S, T)]
    n = InfiniteSquareNetwork(psi)
    env, = leading_boundary(env_pre, psi, alg)
    env′, = ctmrg_iteration(n, env, alg)
    env_fixed, = gauge_fix(env′, gauge_alg, env)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end

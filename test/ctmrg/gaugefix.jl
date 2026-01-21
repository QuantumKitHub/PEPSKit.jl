using Test
using Random
using PEPSKit
using TensorKit

using PEPSKit: ctmrg_iteration, calc_elementwise_convergence
using PEPSKit: ScramblingEnvGauge, ScramblingEnvGaugeC4v

spacetypes = [ComplexSpace, Z2Space]
scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
ctmrg_algs = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs = [:halfinfinite, :fullinfinite]
gauge_algs = [ScramblingEnvGauge()]
tol = 1.0e-6  # large tol due to χ=6
χ = 6
atol = 1.0e-4

function _pre_converge_env(
        ::Type{T}, physical_space, peps_space, ctm_space, unitcell; seed = 12345
    ) where {T}
    Random.seed!(seed)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(rand, T, physical_space, peps_space; unitcell)
    env₀ = CTMRGEnv(psi, ctm_space)
    env_conv, = leading_boundary(env₀, psi; alg = :sequential, tol)
    return env_conv, psi
end

# pre-converge CTMRG environments with given spacetype, scalartype and unit cell
preconv = Dict()
for (S, T, unitcell) in Iterators.product(spacetypes, scalartypes, unitcells)
    if S == ComplexSpace
        result = _pre_converge_env(T, S(2), S(2), S(χ), unitcell)
    elseif S == Z2Space
        result = _pre_converge_env(
            T, S(0 => 1, 1 => 1), S(0 => 1, 1 => 1), S(0 => χ ÷ 2, 1 => χ ÷ 2), unitcell
        )
    end
    push!(preconv, (S, T, unitcell) => result)
end

# asymmetric CTMRG
@testset "($S) - ($T) - ($unitcell) - ($ctmrg_alg) - ($projector_alg) - ($gauge_alg)" for (
        S, T, unitcell, ctmrg_alg, projector_alg, gauge_alg,
    ) in Iterators.product(
        spacetypes, scalartypes, unitcells, ctmrg_algs, projector_algs, gauge_algs
    )
    alg = ctmrg_alg(; tol, projector_alg)
    env_pre, psi = preconv[(S, T, unitcell)]
    n = InfiniteSquareNetwork(psi)
    env, = leading_boundary(env_pre, psi, alg)
    env′, = ctmrg_iteration(n, env, alg)
    env_fixed, = gauge_fix(env′, gauge_alg, env)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end

projector_algs_c4v = [:c4v_eigh, :c4v_qr]
gauge_algs_c4v = [ScramblingEnvGaugeC4v()]

# C4v CTMRG
@testset "($S) - ($T) - ($projector_alg) - ($gauge_alg)" for (
        S, T, unitcell, ctmrg_alg, projector_alg, gauge_alg,
    ) in Iterators.product(
        spacetypes, scalartypes, projector_algs_c4v, gauge_algs_c4v
    )
    alg = C4vCTMRG(; tol, projector_alg)
    env_pre, psi = preconv[(S, T, unitcell)] # TODO
    n = InfiniteSquareNetwork(psi)
    env, = leading_boundary(env_pre, psi, alg)
    env′, = ctmrg_iteration(n, env, alg)
    env_fixed, = gauge_fix(env′, gauge_alg, env)
    @test calc_elementwise_convergence(env, env_fixed) ≈ 0 atol = atol
end

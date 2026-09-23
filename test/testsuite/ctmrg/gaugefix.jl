using Test
using Random
using PEPSKit
using TensorKit
using Adapt
using PEPSKit: ctmrg_iteration, calc_elementwise_convergence
using PEPSKit: ScramblingEnvGauge, ScramblingEnvGaugeC4v
using PEPSKit: peps_normalize

spacetypes = [ComplexSpace, Z2Space]
scalartypes = [Float64, ComplexF64]
unitcells = [(1, 1), (2, 2), (3, 2)]
ctmrg_algs_asymm = [SequentialCTMRG, SimultaneousCTMRG]
projector_algs_asymm = [:HalfInfiniteProjector, :FullInfiniteProjector]
projector_algs_c4v = [:C4vEighProjector, :C4vQRProjector]
gauge_algs_asymm = [ScramblingEnvGauge()]
gauge_algs_c4v = [ScramblingEnvGaugeC4v()]

# minimal subsets of the combinations above which still cover every option value at least once
minimal_combinations_asymm = [
    (ComplexSpace, Float64, (1, 1), SequentialCTMRG, :HalfInfiniteProjector, ScramblingEnvGauge()),
    (Z2Space, ComplexF64, (2, 2), SimultaneousCTMRG, :FullInfiniteProjector, ScramblingEnvGauge()),
    (ComplexSpace, ComplexF64, (3, 2), SimultaneousCTMRG, :HalfInfiniteProjector, ScramblingEnvGauge()),
]
minimal_combinations_c4v = [
    (ComplexSpace, Float64, :C4vEighProjector, ScramblingEnvGaugeC4v()),
    (Z2Space, ComplexF64, :C4vQRProjector, ScramblingEnvGaugeC4v()),
]
tol = 1.0e-6  # large tol due to χ=6
χ = 6
atol = 1.0e-4

function _pre_converge_env(
        ::Type{T}, alg, physical_space, peps_space, env_space, unitcell;
        seed = 985293852935829
    ) where {T}
    Random.seed!(seed)  # Seed RNG to make random environment consistent
    psi = InfinitePEPS(rand, T, physical_space, peps_space; unitcell)
    alg == :C4vCTMRG && (psi = peps_normalize(symmetrize!(psi, RotateReflect())))
    env₀ = if alg == :C4vCTMRG
        initialize_singlet_c4v_env(T, psi, env_space)
    else
        CTMRGEnv(psi, env_space)
    end
    env_conv, = leading_boundary(env₀, psi; alg, tol)
    return env_conv, psi
end

# Pre-converged CTMRG environments, keyed by spacetype, scalartype and unit cell.
# These are computed on first use rather than when this file is included: including
# the test suite is what every test worker does, while only the gauge fixing tests
# below need these environments. `_pre_converge_env` reseeds the RNG itself, so a
# lazily computed environment is identical to an eagerly computed one.
const preconv = Dict()
const preconv_c4v = Dict()

function _preconverged_env(S, ::Type{T}, unitcell) where {T}
    return get!(preconv, (S, T, unitcell)) do
        if S == ComplexSpace
            _pre_converge_env(T, :SequentialCTMRG, S(2), S(2), S(χ), unitcell)
        elseif S == Z2Space
            _pre_converge_env(
                T, :SequentialCTMRG, S(0 => 1, 1 => 1), S(0 => 1, 1 => 1),
                S(0 => χ ÷ 2, 1 => χ ÷ 2), unitcell
            )
        else
            error("unsupported space type $S")
        end
    end
end

function _preconverged_env_c4v(S, ::Type{T}) where {T}
    return get!(preconv_c4v, (S, T)) do
        if S == ComplexSpace
            _pre_converge_env(T, :C4vCTMRG, S(2), S(2), S(χ), (1, 1))
        elseif S == Z2Space
            _pre_converge_env(
                T, :C4vCTMRG, S(0 => 1, 1 => 1), S(0 => 1, 1 => 1), S(0 => χ ÷ 2, 1 => χ ÷ 2), (1, 1)
            )
        else
            error("unsupported space type $S")
        end
    end
end

function ctmrg_gaugefix_asymmetric(AT; minimal::Bool = false)
    return @testset "($S) - ($T) - ($unitcell) - ($ctmrg_alg) - ($projector_alg) - ($gauge_alg) - ($AT)" for (
            S, T, unitcell, ctmrg_alg, projector_alg, gauge_alg,
        ) in (
            minimal ? minimal_combinations_asymm :
                Iterators.product(
                spacetypes, scalartypes, unitcells, ctmrg_algs_asymm, projector_algs_asymm, gauge_algs_asymm
            )
        )
        alg = ctmrg_alg(; tol, projector_alg)
        env_pre, psi = _preconverged_env(S, T, unitcell)
        psi = adapt(AT, psi)
        env_pre = adapt(AT, env_pre)
        n = InfiniteSquareNetwork(psi)
        env, = leading_boundary(env_pre, psi, alg)
        env′, = ctmrg_iteration(n, env, alg)
        env_fixed = gauge_fix(env′, env, gauge_alg)
        env_diff = calc_elementwise_convergence(env, env_fixed)
        @info "Diff between iters = $(env_diff)"
        @test env_diff ≈ 0 atol = atol
    end
end

function ctmrg_gaugefix_c4v(AT; minimal::Bool = false)
    return @testset "($S) - ($T) - ($projector_alg) - ($gauge_alg) - ($AT)" for (
            S, T, projector_alg, gauge_alg,
        ) in (
            minimal ? minimal_combinations_c4v :
                Iterators.product(spacetypes, scalartypes, projector_algs_c4v, gauge_algs_c4v)
        )
        alg = C4vCTMRG(; tol, projector_alg)
        env_pre, psi = _preconverged_env_c4v(S, T)
        psi = adapt(AT, psi)
        env_pre = adapt(AT, env_pre)
        n = InfiniteSquareNetwork(psi)
        env, = leading_boundary(env_pre, psi, alg)
        env′, = ctmrg_iteration(n, env, alg)
        env_fixed = gauge_fix(env′, env, gauge_alg)
        env_diff = calc_elementwise_convergence(env, env_fixed)
        @info "Diff between iters = $(env_diff)"
        @test env_diff ≈ 0 atol = atol
    end
end

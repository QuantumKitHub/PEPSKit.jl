"""
$(TYPEDEF)

Simultaneous CTMRG for triangular PEPS and classical networks, with direct Ta/Tb
updates of all six edge directions. See [`leading_boundary`](@ref) for convergence and truncation diagnostics.

## Constructors

    SimultaneousCTMRGTriangular(; kwargs...)

* `tol::Real=$(Defaults.ctmrg_tol)` : Convergence tolerance for corner and edge spectra.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)` : Maximum number of CTMRG iterations.
* `miniter::Int=$(Defaults.ctmrg_miniter)` : Minimum number of CTMRG iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)` : Output verbosity, as in [`leading_boundary`](@ref).
* `decomposition_alg::Union{SVDAdjoint,NamedTuple}=(;)` : SVD algorithm and reverse rule,
  as in square CTMRG. A `NamedTuple` specifies the forward algorithm; use [`SVDAdjoint`](@ref)
  to configure its pullback. Full corner and edge conditioning require a dense SVD.
* `projector_alg::Symbol=:full` : Corner-projector construction, either `:twothirds` or `:full`.
* `conditioning::Bool=true` : Square-root SVD conditioning of the Ta/Tb edge split.
  This does not change the corner projectors; `:full` always splits its opposite
  corner factor with a square-root SVD, while `:twothirds` uses its factors directly.
* `trunctype::Symbol=:truncrank` : Keep the initial environment dimension.
  Use `:FixedSpaceTruncation` to preserve each environment bond space, including its sectors.
"""
struct SimultaneousCTMRGTriangular{S <: SVDAdjoint} <: CTMRGAlgorithmTriangular
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    conditioning::Bool
    projector_alg::Symbol
    trunctype::Symbol
    decomposition_alg::S
end
function SimultaneousCTMRGTriangular(;
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        miniter = Defaults.ctmrg_miniter,
        verbosity = Defaults.ctmrg_verbosity,
        conditioning = true,
        projector_alg = :full,
        trunctype = :truncrank,
        decomposition_alg = (;)
    )
    if decomposition_alg isa NamedTuple
        decomposition_alg = SVDAdjoint(; fwd_alg = decomposition_alg)
    end
    return SimultaneousCTMRGTriangular(tol, maxiter, miniter, verbosity, conditioning, projector_alg, trunctype, decomposition_alg)
end

# Based on
# https://arxiv.org/pdf/2510.04907

function ctmrg_iteration(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, alg::SimultaneousCTMRGTriangular)
    trunc_type = getfield(@__MODULE__, alg.trunctype)
    if trunc_type == FixedSpaceTruncation
        trunc = FixedSpaceTruncation()
    else
        trunc = trunc_type(dim(domain(env.C[1, 1, 1])[1]))
    end
    Pas, Pbs, S, corner_truncation_error = calculate_projectors(network, env, trunc, alg.projector_alg, alg.decomposition_alg)

    env = renormalize_corners!(network, env, Pas, Pbs)
    env = normalize_corners(env)

    mats, env = semi_renormalize(network, env, Pas, Pbs, trunc, alg.decomposition_alg)
    env, edge_truncation_error = renormalize_edges(network, env, mats, trunc, alg.decomposition_alg; alg.conditioning)
    truncation_error = max(corner_truncation_error, edge_truncation_error)
    info = (;
        contraction_metrics = (; truncation_error, corner_truncation_error, edge_truncation_error), S,
    )
    return env, info
end

function calculate_projectors(network, env, trunc, projector_alg, decomposition_alg = SVDAdjoint())
    if projector_alg == :full
        return calculate_full_projectors(network, env, trunc, decomposition_alg)
    elseif projector_alg == :twothirds
        return calculate_twothirds_projectors(network, env, trunc, decomposition_alg)
    else
        @error "projector_alg = $projector_alg not defined"
    end
end

# TODO: remove once this once everything is plumbed through a proper projector algorithm
_truncation_strategy(trunc::TruncationStrategy, _) = trunc
function _truncation_strategy(::FixedSpaceTruncation, edge)
    tspace = space(edge, 1)
    return isdual(tspace) ? truncspace(flip(tspace)) : truncspace(tspace)
end

# Include retained values suppressed by the inverse square root, without subtracting
# nearly equal norms. Normalize by norm(S), as in the square CTMRG projectors.
function _relative_truncation_error(S, isqrtS, truncation_error)
    return ignore_derivatives() do
        discarded_S = DiagonalTensorMap(S.data .* iszero.(isqrtS.data), space(S, 1))
        discarded_norm = hypot(truncation_error, norm(discarded_S))
        retained_norm = norm(S)
        iszero(retained_norm) && return iszero(discarded_norm) ? zero(discarded_norm) : oftype(discarded_norm, Inf)
        return discarded_norm / retained_norm
    end
end

_normalize_corner_environment(ρ) = ρ / norm(ρ)

function calculate_twothirds_projectors(network::InfiniteTriangularNetwork{P}, env, trunc, decomposition_alg = SVDAdjoint()) where {P}
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, _prev(dir, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, _next(dir, 6), r, c)
        ρL = _normalize_corner_environment(ρL)
        ρR = _normalize_corner_environment(ρR)
        ρρ = _normalize_corner_environment(ρL * ρR)

        trunc′ = _truncation_strategy(trunc, env.Ea[dir, r, c])
        svd_alg = _set_decomposition_truncation(decomposition_alg, trunc′)
        U, S, V, truncation_error = svd_trunc(ρρ, svd_alg)
        iqsrtS = sdiag_pow(S, -0.5)
        truncation_error = _relative_truncation_error(S, iqsrtS, truncation_error)

        Pb = ρR * V' * iqsrtS
        Pa = iqsrtS * U' * ρL
        return Pa, Pb, S, truncation_error
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3), maximum(x -> x[4], projectors)
end

function calculate_full_projectors(network::InfiniteTriangularNetwork{P}, env, trunc, decomposition_alg = SVDAdjoint()) where {P}
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρ̄ = build_double_corner_matrix_triangular(network, env, mod1(dir + 3, 6), r, c)
        ρ̄ = _normalize_corner_environment(ρ̄)
        Ū, S̄, V̄ᴴ = svd_full(ρ̄, decomposition_alg)
        sqrtS̄ = sdiag_pow(S̄, 0.5)
        ρ̄ᴿ = Ū * sqrtS̄
        ρ̄ᴸ = sqrtS̄ * V̄ᴴ
        ρL = _normalize_corner_environment(ρ̄ᴸ * ρL)
        ρR = _normalize_corner_environment(ρR * ρ̄ᴿ)
        ρρ = _normalize_corner_environment(ρL * ρR)

        trunc′ = _truncation_strategy(trunc, env.Ea[dir, r, c])
        svd_alg = _set_decomposition_truncation(decomposition_alg, trunc′)
        U, S, Vᴴ, truncation_error = svd_trunc(ρρ, svd_alg)
        isqrtS = sdiag_pow(S, -0.5)
        truncation_error = _relative_truncation_error(S, isqrtS, truncation_error)
        Pb = ρR * Vᴴ' * isqrtS
        Pa = isqrtS * U' * ρL

        return Pa, Pb, S, truncation_error
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3), maximum(x -> x[4], projectors)
end

function renormalize_corners!(network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PEPSSandwichTriangular}
    coordinates = eachcoordinate(network, 1:6)
    new_corners′ = similar(env.C)
    new_corners = dtmap!!(new_corners′, coordinates) do (dir, r, c)
        return renormalize_corner_triangular((dir, r, c), network, env, Pas, Pbs)
    end
    return CTMRGEnvTriangular(new_corners, env.Ea, env.Eb)
end

function renormalize_corners!(network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PFTensorTriangular}
    coordinates = eachcoordinate(network, 1:6)
    new_corners′ = similar(env.C)
    new_corners = dtmap!!(new_corners′, coordinates) do (dir, r, c)
        return renormalize_corner_triangular((dir, r, c), network, env, Pas, Pbs)
    end
    return CTMRGEnvTriangular(new_corners, env.Ea, env.Eb)
end

function _permute_edge(t::Union{T1, T2}) where {E, S, T1 <: AbstractTensorMap{E, S, 2, 1}, T2 <: AbstractTensorMap{E, S, 1, 2}}
    return permute(t, ((1, 3), (2,)))
end

function _permute_edge(t::Union{T1, T2}) where {E, S, T1 <: AbstractTensorMap{E, S, 3, 1}, T2 <: AbstractTensorMap{E, S, 1, 3}}
    return permute(t, ((1, 3, 4), (2,)))
end

function semi_renormalize(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, Pas, Pbs, trunc, decomposition_alg = SVDAdjoint())
    edges = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        mat = semi_renormalize_edge(network, env, Pas, Pbs, dir, r, c)
        trunc′ = _truncation_strategy(trunc, env.Ea[dir, r, c])
        svd_alg = _set_decomposition_truncation(decomposition_alg, trunc′)
        U, S, V = svd_trunc(mat, svd_alg)
        sqrtS = sdiag_pow(S, 0.5)
        Ẽb = U * sqrtS
        Ẽa = _permute_edge(sqrtS * V)
        return mat, Ẽa, Ẽb
    end
    return getindex.(edges, 1), CTMRGEnvTriangular(env.C, getindex.(edges, 2), getindex.(edges, 3))
end

function _truncate_edge(mat, ρL, ρR, trunc, decomposition_alg = SVDAdjoint(); conditioning = true)
    ρL /= norm(ρL)
    ρR /= norm(ρR)
    if conditioning
        _, SL, VLᴴ = svd_full(ρL, decomposition_alg)
        UR, SR, _ = svd_full(ρR, decomposition_alg)
        sqrtSL = sdiag_pow(SL, 0.5)
        sqrtSR = sdiag_pow(SR, 0.5)
        ρL = sqrtSL * VLᴴ
        ρR = UR * sqrtSR
    end

    ρ = ρL * mat * ρR

    ρ /= norm(ρ)
    svd_alg = _set_decomposition_truncation(decomposition_alg, trunc)
    U, S, Vᴴ, truncation_error = svd_trunc(ρ, svd_alg)
    isqrtS = sdiag_pow(S, -0.5)
    truncation_error = _relative_truncation_error(S, isqrtS, truncation_error)
    Ea = _permute_edge(isqrtS * U' * ρL * mat)
    Eb = mat * ρR * Vᴴ' * isqrtS
    return Ea, Eb, truncation_error
end

function renormalize_edges(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, mats, trunc, decomposition_alg = SVDAdjoint(); conditioning = true)
    edges = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        ρL, ρR = build_halfinfinite_environments(network, env, dir, r, c)
        trunc′ = _truncation_strategy(trunc, env.Ea[dir, r, c])
        return _truncate_edge(mats[dir, r, c], ρL, ρR, trunc′, decomposition_alg; conditioning)
    end
    env′ = normalize_edges(CTMRGEnvTriangular(env.C, getindex.(edges, 1), getindex.(edges, 2)))
    return env′, maximum(x -> x[3], edges)
end

function normalize_corners(env::CTMRGEnvTriangular)
    C_normalized = map(env.C) do C
        return C / norm(C)
    end
    return CTMRGEnvTriangular(C_normalized, env.Ea, env.Eb)
end

function normalize_edges(env::CTMRGEnvTriangular)
    Ea_normalized = map(env.Ea) do e
        return e / norm(e)
    end
    Eb_normalized = map(env.Eb) do e
        return e / norm(e)
    end
    return CTMRGEnvTriangular(env.C, Ea_normalized, Eb_normalized)
end

function calculate_error(Ss, Ss_prev)
    ε = Inf
    for (S, S_prev) in zip(Ss, Ss_prev)
        if space(S) == space(S_prev)
            ε = norm(S^4 - S_prev^4)
        else
            return Inf
        end
    end
    return ε
end

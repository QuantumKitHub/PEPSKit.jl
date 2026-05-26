struct SimultaneousCTMRGTriangular <: CTMRGAlgorithmTriangular
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    conditioning::Bool
    projector_alg::Symbol
    trunctype::Symbol
end
function SimultaneousCTMRGTriangular(;
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        miniter = Defaults.ctmrg_miniter,
        verbosity = Defaults.ctmrg_verbosity,
        conditioning = true,
        projector_alg = :twothirds,
        trunctype = :truncrank
    )
    return SimultaneousCTMRGTriangular(tol, maxiter, miniter, verbosity, conditioning, projector_alg, trunctype)
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
    Pas, Pbs, S = calculate_projectors(network, env, trunc, alg.projector_alg)

    env = renormalize_corners!(network, env, Pas, Pbs)
    env = normalize_corners(env)

    Ẽas, Ẽbs, Ẽastr, Ẽbstr = semi_renormalize(network, env, Pas, Pbs, trunc)
    Qas, Qbs = build_matrix_second_projectors(network, env, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; alg.conditioning)

    env = renormalize_edges(env, Ẽas, Ẽbs, Qas, Qbs)
    env = normalize_edges(env)
    return env, S
end

function calculate_projectors(network, env, trunc, projector_alg)
    if projector_alg == :full
        return calculate_full_projectors(network, env, trunc)
    elseif projector_alg == :twothirds
        return calculate_twothirds_projectors(network, env, trunc)
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

function calculate_twothirds_projectors(network::InfiniteTriangularNetwork{P}, env, trunc) where {P}
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, _prev(dir, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, _next(dir, 6), r, c)
        ρρ = ρL * ρR
        ρρ /= norm(ρρ)

        trunc = _truncation_strategy(trunc, env.Ea[dir, r, c])
        U, S, V = svd_trunc(ρρ; trunc)
        iqsrtS = sdiag_pow(S, -0.5)

        Pb = ρR * V' * iqsrtS
        Pa = iqsrtS * U' * ρL
        return Pa, Pb, S
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3)
end

function calculate_full_projectors(network::InfiniteTriangularNetwork{P}, env, trunc) where {P}
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρ̄ = build_double_corner_matrix_triangular(network, env, mod1(dir + 3, 6), r, c)
        ρ̄ /= norm(ρ̄)
        Ū, S̄, V̄ᴴ = svd_full(ρ̄)
        sqrtS̄ = sdiag_pow(S̄, 0.5)
        ρ̄ᴿ = Ū * sqrtS̄
        ρ̄ᴸ = sqrtS̄ * V̄ᴴ
        ρρ = ρ̄ᴸ * ρL * ρR * ρ̄ᴿ
        ρρ /= norm(ρρ)

        trunc = _truncation_strategy(trunc, env.Ea[dir, r, c])
        U, S, Vᴴ = svd_trunc(ρρ; trunc)
        isqrtS = sdiag_pow(S, -0.5)
        Pb = ρR * ρ̄ᴿ * Vᴴ' * isqrtS
        Pa = isqrtS * U' * ρ̄ᴸ * ρL

        return Pa, Pb, S
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3)
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

function semi_renormalize(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, Pas, Pbs, trunc)
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        mat = semi_renormalize_edge(network, env, Pas, Pbs, dir, r, c)

        U, S, V = svd_full(mat)

        sqrtS = sdiag_pow(S, 0.5)

        Ẽb = U * sqrtS
        Ẽa = _permute_edge(sqrtS * V)

        trunc = _truncation_strategy(trunc, env.Ea[dir, r, c])
        Utr, Str, Vtr = svd_trunc(mat; trunc)
        sqrt_Str = sdiag_pow(Str, 0.5)
        Ẽbtr = Utr * sqrt_Str
        Ẽatr = _permute_edge(sqrt_Str * Vtr)

        return Ẽa, Ẽb, Ẽatr, Ẽbtr
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3), getindex.(projectors, 4)
end

function build_matrix_second_projectors(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; conditioning = true)
    projectors = dtmap(eachcoordinate(network, 1:6)) do (dir, r, c)
        trunc = _truncation_strategy(trunc, env.Ea[dir, r, c])
        σL, σR = build_halfinfinite_projectors(network, env, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c)
        if conditioning
            σL /= norm(σL)
            σR /= norm(σR)
            _, SL, VLᴴ = svd_full(σL)
            UR, SR, _ = svd_full(σR)

            sqrtSL = sdiag_pow(SL, 0.5)
            sqrtSR = sdiag_pow(SR, 0.5)

            FLU = sqrtSL * VLᴴ
            FRU = UR * sqrtSR

            mat = FLU * FRU
            mat /= norm(mat)
            WU, SU, QUᴴ = svd_trunc(mat; trunc)
            isqrtSU = sdiag_pow(SU, -0.5)

            Qa = isqrtSU * WU' * FLU
            Qb = FRU * QUᴴ' * isqrtSU
        else
            mat = σL * σR
            mat /= norm(mat)
            U, S, V = svd_trunc(mat; trunc)
            isqrtS = sdiag_pow(S, -0.5)
            Qa = isqrtS * U' * σL
            Qb = σR * V' * isqrtS
        end
        return Qa, Qb
    end
    PL = map(x -> x[1], projectors)
    PR = map(x -> x[2], projectors)
    return PL, PR
end

function renormalize_edges(env::CTMRGEnvTriangular, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {E, S, T1 <: AbstractTensorMap{E, S, 3, 1}, T2 <: AbstractTensorMap{E, S, 1, 1}}
    new_edges = dtmap(eachcoordinate(env, 1:6)) do (dir, r, c)
        Eb_new = Ẽbs[dir, r, c] * Qbs[dir, r, c]
        Ea_new = permute(Qas[dir, r, c] * permute(Ẽas[dir, r, c], ((1,), (2, 3, 4))), ((1, 2, 3), (4,)))
        return Ea_new, Eb_new
    end
    Ea = map(x -> x[1], new_edges)
    Eb = map(x -> x[2], new_edges)
    return CTMRGEnvTriangular(env.C, Ea, Eb)
end

function renormalize_edges(env::CTMRGEnvTriangular, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {T1 <: CTMRGEdgeTensor, T2 <: CTMRGCornerTensor}
    new_edges = dtmap(eachcoordinate(env, 1:6)) do (dir, r, c)
        Eb_new = Ẽbs[dir, r, c] * Qbs[dir, r, c]
        Ea_new = permute(Qas[dir, r, c] * permute(Ẽas[dir, r, c], ((1,), (2, 3))), ((1, 2), (3,)))
        return Ea_new, Eb_new
    end
    return CTMRGEnvTriangular(env.C, getindex.(new_edges, 1), getindex.(new_edges, 2))
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

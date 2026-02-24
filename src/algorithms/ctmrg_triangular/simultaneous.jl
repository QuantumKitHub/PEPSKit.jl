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
    env = normalize_corners!(env)

    Ẽas, Ẽbs, Ẽastr, Ẽbstr = semi_renormalize(network, env, Pas, Pbs, trunc)
    Qas, Qbs = build_matrix_second_projectors(network, env, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; alg.conditioning)

    env = renormalize_edges(env, Ẽas, Ẽbs, Qas, Qbs)
    normalize_edges!(env)
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

function calculate_twothirds_projectors(network::InfiniteTriangularNetwork{P}, env, trunc) where {P}
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = Tuple{AbstractTensorMap{E, S, 1, 2 + (network[1, 1] isa Tuple)}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, DiagonalTensorMap{real(E), S}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρρ = ρL * ρR
        ρρ /= norm(ρρ)

        U, S, V = svd_trunc(ρρ; trunc)

        Pb = ρR * V' * sdiag_pow(S, -1 / 2)
        Pa = sdiag_pow(S, -1 / 2) * U' * ρL
        return Pa, Pb, S
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3)
end

function calculate_full_projectors(network::InfiniteTriangularNetwork{P}, env, trunc) where {P}
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = Tuple{AbstractTensorMap{E, S, 1, 2 + (network[1, 1] isa Tuple)}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, DiagonalTensorMap{real(E), S}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρ̄ = build_double_corner_matrix_triangular(network, env, mod1(dir + 3, 6), r, c)
        ρ̄ /= norm(ρ̄)
        Ū, S̄, V̄ᴴ = svd_full(ρ̄)
        ρ̄ᴿ = Ū * sqrt(S̄)
        ρ̄ᴸ = sqrt(S̄) * V̄ᴴ
        ρρ = ρ̄ᴸ * ρL * ρR * ρ̄ᴿ
        ρρ /= norm(ρρ)

        U, S, Vᴴ = svd_trunc(ρρ; trunc)

        Pb = ρR * ρ̄ᴿ * Vᴴ' * sdiag_pow(S, -1 / 2)
        Pa = sdiag_pow(S, -1 / 2) * U' * ρ̄ᴸ * ρL

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
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = Tuple{AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        mat = semi_renormalize_edge(network, env, Pas, Pbs, dir, r, c)

        U, S, V = svd_full(mat)

        Ẽb = U * sqrt(S)
        Ẽa = _permute_edge(sqrt(S) * V)

        Utr, Str, Vtr = svd_trunc(mat; trunc)
        Ẽbtr = Utr * sqrt(Str)
        Ẽatr = _permute_edge(sqrt(Str) * Vtr)

        return Ẽa, Ẽb, Ẽatr, Ẽbtr
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3), getindex.(projectors, 4)
end

function build_matrix_second_projectors(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; conditioning = true)
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = Tuple{AbstractTensorMap{E, S, 1, 1}, AbstractTensorMap{E, S, 1, 1}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        σL, σR = build_halfinfinite_projectors(network, env, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c)
        if conditioning
            σL /= norm(σL)
            σR /= norm(σR)
            UL, SL, VLᴴ = svd_full(σL)
            UR, SR, VRᴴ = svd_full(σR)

            FLU = sqrt(SL) * VLᴴ
            FRU = UR * sqrt(SR)

            mat = FLU * FRU
            mat /= norm(mat)
            WU, SU, QUᴴ = svd_trunc(mat; trunc)

            Qa = sdiag_pow(SU, -1 / 2) * WU' * FLU
            Qb = FRU * QUᴴ' * sdiag_pow(SU, -1 / 2)
        else
            mat = σL * σR
            mat /= norm(mat)
            U, S, V = svd_trunc(mat; trunc)
            Qa = sdiag_pow(S, -1 / 2) * U' * σL
            Qb = σR * V' * sdiag_pow(S, -1 / 2)
        end
        return Qa, Qb
    end
    return getindex.(projectors, 1), getindex.(projectors, 2)
end

function renormalize_edges(env::CTMRGEnvTriangular, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {E, S, T1 <: AbstractTensorMap{E, S, 3, 1}, T2 <: AbstractTensorMap{E, S, 1, 1}}
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    T_proj = Tuple{AbstractTensorMap{E, S, 3, 1}, AbstractTensorMap{E, S, 3, 1}}
    new_edges′ = similar(env.Ea, T_proj)
    new_edges = dtmap!!(new_edges′, coordinates) do (dir, r, c)
        Eb_new = Ẽbs[dir, r, c] * Qbs[dir, r, c]
        Ea_new = permute(Qas[dir, r, c] * permute(Ẽas[dir, r, c], ((1,), (2, 3, 4))), ((1, 2, 3), (4,)))
        return Ea_new, Eb_new
    end
    return CTMRGEnvTriangular(env.C, getindex.(new_edges, 1), getindex.(new_edges, 2))
end

function renormalize_edges(env::CTMRGEnvTriangular, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {E, S, T1 <: AbstractTensorMap{E, S, 2, 1}, T2 <: AbstractTensorMap{E, S, 1, 1}}
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    T_proj = Tuple{AbstractTensorMap{E, S, 2, 1}, AbstractTensorMap{E, S, 2, 1}}
    new_edges′ = similar(env.Ea, T_proj)
    new_edges = dtmap!!(new_edges′, coordinates) do (dir, r, c)
        Eb_new = Ẽbs[dir, r, c] * Qbs[dir, r, c]
        Ea_new = permute(Qas[dir, r, c] * permute(Ẽas[dir, r, c], ((1,), (2, 3))), ((1, 2), (3,)))
        return Ea_new, Eb_new
    end
    return CTMRGEnvTriangular(env.C, getindex.(new_edges, 1), getindex.(new_edges, 2))
end

function normalize_corners!(env)
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    new_corners′ = similar(env.C, typeof(env.C[1, 1, 1]))
    new_corners = dtmap!!(new_corners′, coordinates) do (dir, r, c)
        env_C_new = env.C[dir, r, c] / norm(env.C[dir, r, c])
        return env_C_new
    end
    return CTMRGEnvTriangular(new_corners, env.Ea, env.Eb)
end

function normalize_edges!(env)
    (r, c) = (1, 1)
    for dir in 1:6
        env.Ea[dir, r, c] /= norm(env.Ea[dir, r, c])
        env.Eb[dir, r, c] /= norm(env.Eb[dir, r, c])
    end
    return env
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

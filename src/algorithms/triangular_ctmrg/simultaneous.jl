struct SimultaneousCTMRGTria <: CTMRGTriaAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    conditioning::Bool
    projector_alg::Symbol
    trunctype::Symbol
end
function SimultaneousCTMRGTria(;
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        miniter = Defaults.ctmrg_miniter,
        verbosity = Defaults.ctmrg_verbosity,
        conditioning = true,
        projector_alg = :twothirds,
        trunctype = :truncrank
    )
    return SimultaneousCTMRGTria(tol, maxiter, miniter, verbosity, conditioning, projector_alg, trunctype)
end

# Based on
# https://arxiv.org/pdf/2510.04907

function ctmrg_iteration(network::InfiniteTriangularNetwork, env::CTMRGTriaEnv, alg::SimultaneousCTMRGTria)
    trunc_type = getfield(@__MODULE__, alg.trunctype)
    trunc = trunc_type(dim(domain(env.C[1, 1, 1])[1]))
    Pas, Pbs, S = calculate_projectors(network, env, trunc, alg.projector_alg)

    renormalize_corners!(network, env, Pas, Pbs)
    normalize_corners!(env)

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
    T_proj = Tuple{AbstractTensorMap{E, S, 1, 2 + (network[1, 1] isa Tuple)}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 1, 1}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρρ = ρL * ρR
        ρρ /= norm(ρρ)

        U, S, V = svd_trunc(ρρ; trunc = trunc & trunctol(; atol = 1.0e-20))

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
    T_proj = Tuple{AbstractTensorMap{E, S, 1, 2 + (network[1, 1] isa Tuple)}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 1, 1}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        ρL = build_double_corner_matrix_triangular(network, env, mod1(dir - 1, 6), r, c)
        ρR = build_double_corner_matrix_triangular(network, env, mod1(dir + 1, 6), r, c)
        ρ̄ = build_double_corner_matrix_triangular(network, env, mod1(dir + 3, 6), r, c)
        ρ̄ /= norm(ρ̄)
        Ū, S̄, V̄ᴴ = svd_trunc(ρ̄; trunc = trunctol(; atol = 1.0e-20))
        ρ̄ᴿ = Ū * sqrt(S̄)
        ρ̄ᴸ = sqrt(S̄) * V̄ᴴ
        ρρ = ρ̄ᴸ * ρL * ρR * ρ̄ᴿ
        ρρ /= norm(ρρ)

        U, S, Vᴴ = svd_trunc(ρρ; trunc = trunctol(; atol = 1.0e-20))
        U, S, Vᴴ = svd_trunc(ρρ; trunc)
        U, S, Vᴴ = svd_trunc(ρρ; trunc = trunc & trunctol(; atol = 1.0e-20))

        Pb = ρR * ρ̄ᴿ * Vᴴ' * sdiag_pow(S, -1 / 2)
        Pa = sdiag_pow(S, -1 / 2) * U' * ρ̄ᴸ * ρL

        return Pa, Pb, S
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3)
end

function renormalize_corners!(network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PEPSTriaSandwich}
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}
    new_corners′ = similar(coordinates, T_proj)
    new_corners = dtmap!!(new_corners′, coordinates) do (dir, r, c)
        @tensor opt = true env_C_new[-1 -2 -3; -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 3 10; 6] * env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][4 2 11; 1] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][6 7 12; 8] *
            rotl60(ket(network[r, c]), mod(dir - 1, 6))[15; 3 7 9 -2 5 2] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[15; 10 12 14 -3 13 11]) *
            Pas[mod1(dir - 1, 6), r, c][-1; 4 5 13] * Pbs[dir, r, c][8 9 14; -4]
        return env_C_new
    end
    return new_corners
end

function renormalize_corners!(network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PFTriaTensor}
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = AbstractTensorMap{E, S, 2, 1}
    new_corners′ = similar(coordinates, T_proj)
    new_corners = dtmap!!(new_corners′, coordinates) do (dir, r, c)
        @tensor opt = true env_C_new[-1 -2; -3] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 3; 6] * env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][4 2; 1] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][6 7; 8] *
            rotl60(network[r, c], mod(dir - 1, 6))[2 5 -2; 3 7 9] * Pas[mod1(dir - 1, 6), r, c][-1; 4 5] * Pbs[dir, r, c][8 9; -3]
        return env_C_new
    end
    return new_corners
end

function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, dir::Int, r::Int, c::Int) where {P <: PFTriaTensor}
    @tensor opt = true mat[-1 -2; -3 -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][6 5; 1] * env.C[_coordinates(dir + 1, 0, r, c, size(network))...][1 3; 2] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][-1 7; 6] * env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][2 4; -3] * rotl60(network[r, c], mod(dir - 1, 6))[7 -2 -4; 5 3 4]
    return mat
end

function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, dir::Int, r::Int, c::Int) where {P <: PEPSTriaSandwich}
    @tensor opt = true mat[-1 -2 -3; -4 -5 -6] := env.C[_coordinates(dir, 0, r, c, size(network))...][6 5 8; 1] * env.C[_coordinates(dir + 1, 0, r, c, size(network))...][1 3 9; 2] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][-1 7 10; 6] * env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][2 4 11; -4] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[12; 5 3 4 -5 -2 7] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[12; 8 9 11 -6 -3 10])
    return mat
end

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, Pas, Pbs, dir, r, c) where {P <: PEPSTriaSandwich}
    @tensor opt = true mat[-1 -2 -3; -4 -5 -6] := Pas[dir, r, c][-1; 1 2 8] * Pbs[dir, r, c][6 7 9; -4] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][1 3 10; 4] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][4 5 11; 6] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[12; 3 5 7 -5 -2 2] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[12; 10 11 9 -6 -3 8])
    return mat / norm(mat)
end

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, Pas, Pbs, dir, r, c) where {P <: PFTriaTensor}
    @tensor opt = true mat[-1 -2; -3 -4] := Pas[dir, r, c][-1; 1 2] * Pbs[dir, r, c][6 7; -3] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][1 3; 4] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][4 5; 6] * rotl60(network[r, c], mod(dir - 1, 6))[2 -2 -4; 3 5 7]
    return mat / norm(mat)
end

function _permute_edge(t::Union{T1, T2}) where {E, S, T1 <: AbstractTensorMap{E, S, 2, 1}, T2 <: AbstractTensorMap{E, S, 1, 2}}
    return permute(t, ((1, 3), (2,)))
end

function _permute_edge(t::Union{T1, T2}) where {E, S, T1 <: AbstractTensorMap{E, S, 3, 1}, T2 <: AbstractTensorMap{E, S, 1, 3}}
    return permute(t, ((1, 3, 4), (2,)))
end

function semi_renormalize(network::InfiniteTriangularNetwork, env::CTMRGTriaEnv, Pas, Pbs, trunc)
    coordinates = eachcoordinate(network, 1:6)
    E = scalartype(env.C[1, 1, 1])
    S = spacetype(env.C[1, 1, 1])
    T_proj = Tuple{AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}, AbstractTensorMap{E, S, 2 + (network[1, 1] isa Tuple), 1}}
    projectors′ = similar(coordinates, T_proj)
    projectors = dtmap!!(projectors′, coordinates) do (dir, r, c)
        mat = semi_renormalize_edge(network, env, Pas, Pbs, dir, r, c)

        U, S, V = svd_trunc(mat; trunc = trunctol(; atol = 1.0e-20))

        Ẽb = U * sqrt(S)
        Ẽa = _permute_edge(sqrt(S) * V)

        Utr, Str, Vtr = svd_trunc(mat; trunc = trunc & trunctol(; atol = 1.0e-20))
        Ẽbtr = Utr * sqrt(Str)
        Ẽatr = _permute_edge(sqrt(Str) * Vtr)

        return Ẽa, Ẽb, Ẽatr, Ẽbtr
    end
    return getindex.(projectors, 1), getindex.(projectors, 2), getindex.(projectors, 3), getindex.(projectors, 4)
end

function build_halfinfinite_projectors(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c) where {P <: PEPSTriaSandwich}
    @tensor opt = true σL[-1 -2 -3; -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 2 10; 8] * env.C[_coordinates(dir - 1, 0, r, c, size(network))...][4 3 11; 1] * env.C[_coordinates(dir - 2, 0, r, c, size(network))...][6 5 12; 4] *
        Ẽbs[_coordinates(dir, 1, r, c, size(network))...][8 9 13; -4] * Ẽastr[_coordinates(dir - 3, 0, r, c, size(network))...][-1 7 14; 6] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[15 2 9 -2 7 5 3] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[15; 10 13 -3 14 12 11])
    @tensor opt = true σR[-1; -2 -3 -4] := env.C[_coordinates(dir + 1, 0, r, c, size(network))...][8 2 10; 1] * env.C[_coordinates(dir + 2, 0, r, c, size(network))...][1 3 11; 4] * env.C[_coordinates(dir + 3, 0, r, c, size(network))...][4 5 12; 6] *
        Ẽbstr[_coordinates(dir + 3, -1, r, c, size(network))...][6 7 13; -2] * Ẽas[_coordinates(dir, 0, r, c, size(network))...][-1 9 14; 8] *
        rotl60(ket(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...]), mod(dir - 1, 6))[15; 9 2 3 5 7 -3] * conj(rotl60(bra(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...]), mod(dir - 1, 6))[15; 14 10 11 12 13 -4])
    return σL, σR
end

function build_halfinfinite_projectors(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c) where {P <: PFTriaTensor}
    @tensor opt = true σL[-1 -2; -3] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 2; 8] * env.C[_coordinates(dir - 1, 0, r, c, size(network))...][4 3; 1] * env.C[_coordinates(dir - 2, 0, r, c, size(network))...][6 5; 4] *
        Ẽbs[_coordinates(dir, 1, r, c, size(network))...][8 9; -3] * Ẽastr[_coordinates(dir - 3, 0, r, c, size(network))...][-1 7; 6] *
        rotl60(network[r, c], mod(dir - 1, 6))[3 5 7; 2 9 -2]
    @tensor opt = true σR[-1; -2 -3] := env.C[_coordinates(dir + 1, 0, r, c, size(network))...][8 2; 1] * env.C[_coordinates(dir + 2, 0, r, c, size(network))...][1 3; 4] * env.C[_coordinates(dir + 3, 0, r, c, size(network))...][4 5; 6] *
        Ẽbstr[_coordinates(dir + 3, -1, r, c, size(network))...][6 7; -2] * Ẽas[_coordinates(dir, 0, r, c, size(network))...][-1 9; 8] *
        rotl60(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...], mod(dir - 1, 6))[-3 7 5; 9 2 3]
    return σL, σR
end

function build_matrix_second_projectors(network::InfiniteTriangularNetwork, env::CTMRGTriaEnv, Ẽas, Ẽbs, Ẽastr, Ẽbstr, trunc; conditioning = true)
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
            UL, SL, VLᴴ = svd_trunc(σL; trunc = trunctol(; atol = 1.0e-20))
            UR, SR, VRᴴ = svd_trunc(σR; trunc = trunctol(; atol = 1.0e-20))

            FLU = sqrt(SL) * VLᴴ
            FRU = UR * sqrt(SR)

            mat = FLU * FRU
            mat /= norm(mat)
            WU, SU, QUᴴ = svd_trunc(mat; trunc = trunc & trunctol(; atol = 1.0e-20))

            Qa = sdiag_pow(SU, -1 / 2) * WU' * FLU
            Qb = FRU * QUᴴ' * sdiag_pow(SU, -1 / 2)
        else
            mat = σL * σR
            mat /= norm(mat)
            U, S, V = svd_trunc(mat; trunc = trunc & trunctol(; atol = 1.0e-20))
            Qa = sdiag_pow(S, -1 / 2) * U' * σL
            Qb = σR * V' * sdiag_pow(S, -1 / 2)
        end
        return Qa, Qb
    end
    return getindex.(projectors, 1), getindex.(projectors, 2)
end

function renormalize_edges(env::CTMRGTriaEnv, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {E, S, T1 <: AbstractTensorMap{E, S, 3, 1}, T2 <: AbstractTensorMap{E, S, 1, 1}}
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    T_proj = Tuple{AbstractTensorMap{E, S, 3, 1}, AbstractTensorMap{E, S, 3, 1}}
    new_edges′ = similar(env.Ea, T_proj)
    new_edges = dtmap!!(new_edges′, coordinates) do (dir, r, c)
        @tensor Eb_new[-1 -2 -3; -4] := Ẽbs[dir, r, c][-1 -2 -3; 1] * Qbs[dir, r, c][1; -4]
        @tensor Ea_new[-1 -2 -3; -4] := Qas[dir, r, c][-1; 1] * Ẽas[dir, r, c][1 -2 -3; -4]
        return Ea_new, Eb_new
    end
    return CTMRGTriaEnv(env.C, getindex.(new_edges, 1), getindex.(new_edges, 2))
end

function renormalize_edges(env::CTMRGTriaEnv, Ẽas::Array{T1, 3}, Ẽbs::Array{T1, 3}, Qas::Array{T2, 3}, Qbs::Array{T2, 3}) where {E, S, T1 <: AbstractTensorMap{E, S, 2, 1}, T2 <: AbstractTensorMap{E, S, 1, 1}}
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    T_proj = Tuple{AbstractTensorMap{E, S, 2, 1}, AbstractTensorMap{E, S, 2, 1}}
    new_edges′ = similar(env.Ea, T_proj)
    new_edges = dtmap!!(new_edges′, coordinates) do (dir, r, c)
        @tensor Eb_new[-1 -2; -3] := Ẽbs[dir, r, c][-1 -2; 1] * Qbs[dir, r, c][1; -3]
        @tensor Ea_new[-1 -2; -3] := Qas[dir, r, c][-1; 1] * Ẽas[dir, r, c][1 -2; -3]
        return Ea_new, Eb_new
    end
    return CTMRGTriaEnv(env.C, getindex.(new_edges, 1), getindex.(new_edges, 2))
end

function normalize_corners!(env)
    coordinates = collect(Iterators.product(axes(env.Ea)...))
    new_env′ = similar(env.C, typeof(env.C[1, 1, 1]))
    new_env = dtmap!!(new_env′, coordinates) do (dir, r, c)
        env_C_new = env.C[dir, r, c] / norm(env.C[dir, r, c])
        return env_C_new
    end
    return new_env
end

function normalize_edges!(env)
    for dir in 1:6
        env.Ea[dir] /= norm(env.Ea[dir])
        env.Eb[dir] /= norm(env.Eb[dir])
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

# Expectation_values

function network_value(network::InfiniteTriangularNetwork, env::CTMRGTriaEnv)
    return zero(scalartype(network))
end

function energy(network::InfiniteTriangularNetwork{P}, env::CTMRGTriaEnv, H) where {P <: PEPSTriaSandwich}
    (r, c) = (1, 1)
    numerator = @tensor opt = true ket(network[r, c])[dLt; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(network[r, c])[dRt; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(network[r, c])[dLb DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(network[r, c])[dRb DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1][χNW DLt120 DLb120; χNa] * env.C[2][χNb DRt60 DRb60; χNE] * env.C[3][χNE DRt0 DRb0; χSE] *
        env.C[4][χSE DRt300 DRb300; χSa] * env.C[5][χSb DLt240 DLb240; χSW] * env.C[6][χSW DLt180 DLb180; χNW] *
        env.Eb[1][χNa DLt60 DLb60; χNC] * env.Ea[1][χNC DRt120 DRb120; χNb] *
        env.Eb[4][χSa DRt240 DRb240; χSC] * env.Ea[4][χSC DLt300 DLb300; χSb] *
        H[dLb dRb; dLt dRt]

    denumerator = @tensor opt = true ket(network[r, c])[dL; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(network[r, c])[dR; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(network[r, c])[dL; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(network[r, c])[dR; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1][χNW DLt120 DLb120; χNa] * env.C[2][χNb DRt60 DRb60; χNE] * env.C[3][χNE DRt0 DRb0; χSE] *
        env.C[4][χSE DRt300 DRb300; χSa] * env.C[5][χSb DLt240 DLb240; χSW] * env.C[6][χSW DLt180 DLb180; χNW] *
        env.Eb[1][χNa DLt60 DLb60; χNC] * env.Ea[1][χNC DRt120 DRb120; χNb] *
        env.Eb[4][χSa DRt240 DRb240; χSC] * env.Ea[4][χSC DLt300 DLb300; χSb]
    return numerator / denumerator
end

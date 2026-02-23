function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir::Int, r::Int, c::Int) where {P <: PFTensorTriangular}
    @tensor opt = true mat[-1 -2; -3 -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][6 5; 1] * env.C[_coordinates(dir + 1, 0, r, c, size(network))...][1 3; 2] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][-1 7; 6] * env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][2 4; -3] * rotl60(network[r, c], mod(dir - 1, 6))[7 -2 -4; 5 3 4]
    return mat
end

function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir::Int, r::Int, c::Int) where {P <: PEPSSandwichTriangular}
    @tensor opt = true mat[-1 -2 -3; -4 -5 -6] := env.C[_coordinates(dir, 0, r, c, size(network))...][6 5 8; 1] * env.C[_coordinates(dir + 1, 0, r, c, size(network))...][1 3 9; 2] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][-1 7 10; 6] * env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][2 4 11; -4] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[12; 5 3 4 -5 -2 7] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[12; 8 9 11 -6 -3 10])
    return mat
end

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Pas, Pbs, dir, r, c) where {P <: PEPSSandwichTriangular}
    @tensor opt = true mat[-1 -2 -3; -4 -5 -6] := Pas[dir, r, c][-1; 1 2 8] * Pbs[dir, r, c][6 7 9; -4] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][1 3 10; 4] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][4 5 11; 6] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[12; 3 5 7 -5 -2 2] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[12; 10 11 9 -6 -3 8])
    return mat / norm(mat)
end

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Pas, Pbs, dir, r, c) where {P <: PFTensorTriangular}
    @tensor opt = true mat[-1 -2; -3 -4] := Pas[dir, r, c][-1; 1 2] * Pbs[dir, r, c][6 7; -3] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][1 3; 4] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][4 5; 6] * rotl60(network[r, c], mod(dir - 1, 6))[2 -2 -4; 3 5 7]
    return mat / norm(mat)
end

function build_halfinfinite_projectors(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c) where {P <: PEPSSandwichTriangular}
    @tensor opt = true σL[-1 -2 -3; -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 2 10; 8] * env.C[_coordinates(dir - 1, 0, r, c, size(network))...][4 3 11; 1] * env.C[_coordinates(dir - 2, 0, r, c, size(network))...][6 5 12; 4] *
        Ẽbs[_coordinates(dir, 1, r, c, size(network))...][8 9 13; -4] * Ẽastr[_coordinates(dir - 3, 0, r, c, size(network))...][-1 7 14; 6] *
        rotl60(ket(network[r, c]), mod(dir - 1, 6))[15 2 9 -2 7 5 3] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[15; 10 13 -3 14 12 11])
    @tensor opt = true σR[-1; -2 -3 -4] := env.C[_coordinates(dir + 1, 0, r, c, size(network))...][8 2 10; 1] * env.C[_coordinates(dir + 2, 0, r, c, size(network))...][1 3 11; 4] * env.C[_coordinates(dir + 3, 0, r, c, size(network))...][4 5 12; 6] *
        Ẽbstr[_coordinates(dir + 3, -1, r, c, size(network))...][6 7 13; -2] * Ẽas[_coordinates(dir, 0, r, c, size(network))...][-1 9 14; 8] *
        rotl60(ket(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...]), mod(dir - 1, 6))[15; 9 2 3 5 7 -3] * conj(rotl60(bra(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...]), mod(dir - 1, 6))[15; 14 10 11 12 13 -4])
    return σL, σR
end

function build_halfinfinite_projectors(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Ẽas, Ẽbs, Ẽastr, Ẽbstr, dir, r, c) where {P <: PFTensorTriangular}
    @tensor opt = true σL[-1 -2; -3] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 2; 8] * env.C[_coordinates(dir - 1, 0, r, c, size(network))...][4 3; 1] * env.C[_coordinates(dir - 2, 0, r, c, size(network))...][6 5; 4] *
        Ẽbs[_coordinates(dir, 1, r, c, size(network))...][8 9; -3] * Ẽastr[_coordinates(dir - 3, 0, r, c, size(network))...][-1 7; 6] *
        rotl60(network[r, c], mod(dir - 1, 6))[3 5 7; 2 9 -2]
    @tensor opt = true σR[-1; -2 -3] := env.C[_coordinates(dir + 1, 0, r, c, size(network))...][8 2; 1] * env.C[_coordinates(dir + 2, 0, r, c, size(network))...][1 3; 4] * env.C[_coordinates(dir + 3, 0, r, c, size(network))...][4 5; 6] *
        Ẽbstr[_coordinates(dir + 3, -1, r, c, size(network))...][6 7; -2] * Ẽas[_coordinates(dir, 0, r, c, size(network))...][-1 9; 8] *
        rotl60(network[_coordinates(dir + 2, 0, r, c, size(network))[2:3]...], mod(dir - 1, 6))[-3 7 5; 9 2 3]
    return σL, σR
end

function renormalize_corner_triangular((dir, r, c), network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PFTensorTriangular}
    @tensor opt = true env_C_new[-1 -2; -3] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 3; 6] * env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][4 2; 1] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][6 7; 8] *
    rotl60(network[r, c], mod(dir - 1, 6))[2 5 -2; 3 7 9] * Pas[mod1(dir - 1, 6), r, c][-1; 4 5] * Pbs[dir, r, c][8 9; -3]
    return env_C_new
end

function renormalize_corner_triangular((dir, r, c), network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PEPSSandwichTriangular}
    @tensor opt = true env_C_new[-1 -2 -3; -4] := env.C[_coordinates(dir, 0, r, c, size(network))...][1 3 10; 6] * env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][4 2 11; 1] * env.Eb[_coordinates(dir, 1, r, c, size(network))...][6 7 12; 8] *
    rotl60(ket(network[r, c]), mod(dir - 1, 6))[15; 3 7 9 -2 5 2] * conj(rotl60(bra(network[r, c]), mod(dir - 1, 6))[15; 10 12 14 -3 13 11]) *
    Pas[mod1(dir - 1, 6), r, c][-1; 4 5 13] * Pbs[dir, r, c][8 9 14; -4]
    return env_C_new
end
# Directions refer to the local frame after rotl60; t/b denote ket/bra layers.

function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir::Int, r::Int, c::Int) where {P <: PFTensorTriangular}
    @tensor opt = true mat[χW D240; χE D300] :=
        env.C[_coordinates(dir, 0, r, c, size(network))...][χNW D120; χN] *
        env.C[_coordinates(dir + 1, 0, r, c, size(network))...][χN D60; χNE] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][χW D180; χNW] *
        env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][χNE D0; χE] *
        rotl60(network[r, c], mod(dir - 1, 6))[D180 D240 D300; D120 D60 D0]
    return mat
end

function build_double_corner_matrix_triangular(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir::Int, r::Int, c::Int) where {P <: PEPSSandwichTriangular}
    O_r_c = network[r, c]
    @tensor opt = true mat[χW Dt240 Db240; χE Dt300 Db300] :=
        env.C[_coordinates(dir, 0, r, c, size(network))...][χNW Dt120 Db120; χN] *
        env.C[_coordinates(dir + 1, 0, r, c, size(network))...][χN Dt60 Db60; χNE] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][χW Dt180 Db180; χNW] *
        env.Eb[_coordinates(dir + 1, 0, r, c, size(network))...][χNE Dt0 Db0; χE] *
        rotl60(ket(O_r_c), mod(dir - 1, 6))[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] *
        conj(rotl60(bra(O_r_c), mod(dir - 1, 6))[d; Db120 Db60 Db0 Db300 Db240 Db180])
    return mat
end

# Directions refer to the local frame after rotl60; t/b denote ket/bra layers.

function renormalize_corner_triangular((dir, r, c), network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PFTensorTriangular}
    @tensor opt = true env_C_new[χSW D300; χSE] :=
        env.C[_coordinates(dir, 0, r, c, size(network))...][χNW D120; χN] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][χW D180; χNW] *
        env.Eb[_coordinates(dir, 1, r, c, size(network))...][χN D60; χNE] *
        rotl60(network[r, c], mod(dir - 1, 6))[D180 D240 D300; D120 D60 D0] *
        Pas[_coordinates(dir - 1, 5, r, c, size(network))...][χSW; χW D240] * Pbs[dir, r, c][χNE D0; χSE]
    return env_C_new
end

function renormalize_corner_triangular((dir, r, c), network::InfiniteTriangularNetwork{P}, env, Pas, Pbs) where {P <: PEPSSandwichTriangular}
    O_r_c = network[r, c]
    @tensor opt = true env_C_new[χSW Dt300 Db300; χSE] :=
        env.C[_coordinates(dir, 0, r, c, size(network))...][χNW Dt120 Db120; χN] *
        env.Ea[_coordinates(dir - 1, 0, r, c, size(network))...][χW Dt180 Db180; χNW] *
        env.Eb[_coordinates(dir, 1, r, c, size(network))...][χN Dt60 Db60; χNE] *
        rotl60(ket(O_r_c), mod(dir - 1, 6))[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] *
        conj(rotl60(bra(O_r_c), mod(dir - 1, 6))[d; Db120 Db60 Db0 Db300 Db240 Db180]) *
        Pas[_coordinates(dir - 1, 5, r, c, size(network))...][χSW; χW Dt240 Db240] * Pbs[dir, r, c][χNE Dt0 Db0; χSE]
    return env_C_new
end

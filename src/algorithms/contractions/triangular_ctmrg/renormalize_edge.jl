# Directions refer to the local frame after rotl60; t/b denote ket/bra layers.

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Pas, Pbs, dir, r, c) where {P <: PEPSSandwichTriangular}
    O_r_c = network[r, c]
    @tensor opt = true mat[χW Dt240 Db240; χE Dt300 Db300] :=
        Pas[dir, r, c][χW; χNW Dt180 Db180] * Pbs[dir, r, c][χNE Dt0 Db0; χE] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][χNW Dt120 Db120; χN] *
        env.Eb[_coordinates(dir, 1, r, c, size(network))...][χN Dt60 Db60; χNE] *
        rotl60(ket(O_r_c), mod(dir - 1, 6))[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] *
        conj(rotl60(bra(O_r_c), mod(dir - 1, 6))[d; Db120 Db60 Db0 Db300 Db240 Db180])
    return mat / norm(mat)
end

function semi_renormalize_edge(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, Pas, Pbs, dir, r, c) where {P <: PFTensorTriangular}
    @tensor opt = true mat[χW D240; χE D300] :=
        Pas[dir, r, c][χW; χNW D180] *
        Pbs[dir, r, c][χNE D0; χE] *
        env.Ea[_coordinates(dir, 0, r, c, size(network))...][χNW D120; χN] *
        env.Eb[_coordinates(dir, 1, r, c, size(network))...][χN D60; χNE] *
        rotl60(network[r, c], mod(dir - 1, 6))[D180 D240 D300; D120 D60 D0]
    return mat / norm(mat)
end

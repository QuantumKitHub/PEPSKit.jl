# Directions refer to the local frame after rotl60; t/b denote ket/bra layers.

function build_halfinfinite_environments(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir, r, c) where {P <: PEPSSandwichTriangular}
    rL, cL = _coordinates(dir + 4, 0, r, c, _cell(network))[2:3]
    rR, cR = _coordinates(dir + 3, 0, r, c, _cell(network))[2:3]
    O_L = network[_cellsite(network, rL, cL)...]
    O_R = network[_cellsite(network, rR, cR)...]
    @tensor opt = true ρL[χSE Dt0 Db0; χN Dt60 Db60] :=
        env.C[_coordinates(dir, 0, rL, cL, _cell(network))...][χNW Dt120 Db120; χN] *
        env.C[_coordinates(dir - 1, 0, rL, cL, _cell(network))...][χSW Dt180 Db180; χNW] *
        env.C[_coordinates(dir - 2, 0, rL, cL, _cell(network))...][χS Dt240 Db240; χSW] *
        env.Ea[_coordinates(dir + 3, 0, rL, cL, _cell(network))...][χSE Dt300 Db300; χS] *
        rotl60(ket(O_L), mod(dir - 1, 6))[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] *
        conj(rotl60(bra(O_L), mod(dir - 1, 6))[d; Db120 Db60 Db0 Db300 Db240 Db180])
    @tensor opt = true ρR[χN Dt120 Db120; χSW Dt180 Db180] :=
        env.C[_coordinates(dir + 1, 0, rR, cR, _cell(network))...][χN Dt60 Db60; χNE] *
        env.C[_coordinates(dir + 2, 0, rR, cR, _cell(network))...][χNE Dt0 Db0; χSE] *
        env.C[_coordinates(dir + 3, 0, rR, cR, _cell(network))...][χSE Dt300 Db300; χS] *
        env.Eb[_coordinates(dir + 3, 1, rR, cR, _cell(network))...][χS Dt240 Db240; χSW] *
        rotl60(ket(O_R), mod(dir - 1, 6))[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] *
        conj(rotl60(bra(O_R), mod(dir - 1, 6))[d; Db120 Db60 Db0 Db300 Db240 Db180])
    return ρL, ρR
end

function build_halfinfinite_environments(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, dir, r, c) where {P <: PFTensorTriangular}
    rL, cL = _coordinates(dir + 4, 0, r, c, _cell(network))[2:3]
    rR, cR = _coordinates(dir + 3, 0, r, c, _cell(network))[2:3]
    @tensor opt = true ρL[χSE D0; χN D60] :=
        env.C[_coordinates(dir, 0, rL, cL, _cell(network))...][χNW D120; χN] *
        env.C[_coordinates(dir - 1, 0, rL, cL, _cell(network))...][χSW D180; χNW] *
        env.C[_coordinates(dir - 2, 0, rL, cL, _cell(network))...][χS D240; χSW] *
        env.Ea[_coordinates(dir + 3, 0, rL, cL, _cell(network))...][χSE D300; χS] *
        rotl60(network[_cellsite(network, rL, cL)...], mod(dir - 1, 6))[D180 D240 D300; D120 D60 D0]
    @tensor opt = true ρR[χN D120; χSW D180] :=
        env.C[_coordinates(dir + 1, 0, rR, cR, _cell(network))...][χN D60; χNE] *
        env.C[_coordinates(dir + 2, 0, rR, cR, _cell(network))...][χNE D0; χSE] *
        env.C[_coordinates(dir + 3, 0, rR, cR, _cell(network))...][χSE D300; χS] *
        env.Eb[_coordinates(dir + 3, 1, rR, cR, _cell(network))...][χS D240; χSW] *
        rotl60(network[_cellsite(network, rR, cR)...], mod(dir - 1, 6))[D180 D240 D300; D120 D60 D0]
    return ρL, ρR
end

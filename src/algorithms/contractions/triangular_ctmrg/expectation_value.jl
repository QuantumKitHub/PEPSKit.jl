function _contract_corners((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 1, 1}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    return @tensor opt = true env.C[1, _prev(r, end), c][χNW Dt120 Db120; χN] * env.C[2, _prev(r, end), _next(c, end)][χN Dt60 Db60; χNE] * env.C[3, r, _next(c, end)][χNE Dt0 Db0; χSE] *
        env.C[4, _next(r, end), c][χSE Dt300 Db300; χS] * env.C[5, _next(r, end), _prev(c, end)][χS Dt240 Db240; χSW] * env.C[6, r, _prev(c, end)][χSW Dt180 Db180; χNW] *
        ket(O_r_c)[dt; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] * conj(bra(O_r_c)[db; Db120 Db60 Db0 Db300 Db240 Db180]) * op[db; dt]
end

function _contract_edges_0((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 2, 2}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    O_r_cn = network[r, _next(c, end)]
    return @tensor opt = true ket(O_r_c)[dtL; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(O_r_cn)[dtR; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(O_r_c)[dbL; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(O_r_cn)[dbR; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1, _prev(r, end), c][χNW DLt120 DLb120; χaN] * env.C[2, _prev(r, end), _next(c + 1, end)][χbN DRt60 DRb60; χNE] * env.C[3, r, _next(c + 1, end)][χNE DRt0 DRb0; χSE] *
        env.C[4, _next(r, end), _next(c, end)][χSE DRt300 DRb300; χaS] * env.C[5, _next(r, end), _prev(c, end)][χbS DLt240 DLb240; χSW] * env.C[6, r, _prev(c, end)][χSW DLt180 DLb180; χNW] *
        env.Eb[1, _prev(r, end), _next(c, end)][χaN DLt60 DLb60; χCN] * env.Ea[1, _prev(r, end), _next(c, end)][χCN DRt120 DRb120; χbN] *
        env.Eb[4, _next(r, end), c][χaS DRt240 DRb240; χCS] * env.Ea[4, _next(r, end), c][χCS DLt300 DLb300; χbS] *
        op[dbL dbR; dtL dtR]
end

function _contract_edges_60((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 2, 2}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    O_rn_cp = network[_next(r, end), _prev(c, end)]
    return @tensor opt = true ket(O_r_c)[dtTR; DTRt120 DTRt60 DTRt0 DTRt300 DBLt60 DTRt180] * ket(O_rn_cp)[dtBL; DBLt120 DBLt60 DBLt0 DBLt300 DBLt240 DBLt180] *
        conj(bra(O_r_c)[dbTR; DTRb120 DTRb60 DTRb0 DTRb300 DBLb60 DTRb180]) * conj(bra(O_rn_cp)[dbBL; DBLb120 DBLb60 DBLb0 DBLb300 DBLb240 DBLb180]) *
        env.C[1, _prev(r, end), c][χbNW DTRt120 DTRb120; χN] * env.C[2, _prev(r, end), _next(c, end)][χN DTRt60 DTRb60; χNE] * env.C[3, r, _next(c, end)][χNE DTRt0 DTRb0; χaSE] *
        env.C[4, _next(r + 1, end), _prev(c, end)][χbSE DBLt300 DBLb300; χS] * env.C[5, _next(r + 1, end), _prev(c - 1, end)][χS DBLt240 DBLb240; χSW] * env.C[6, _next(r, end), _prev(c - 1, end)][χSW DBLt180 DBLb180; χaNW] *
        env.Eb[3, _next(r, end), c][χaSE DTRt300 DTRb300; χCSE] * env.Ea[3, _next(r, end), c][χCSE DBLt0 DBLb0; χbSE] *
        env.Eb[6, r, _prev(c, end)][χaNW DBLt120 DBLb120; χCNW] * env.Ea[6, r, _prev(c, end)][χCNW DTRt180 DTRb180; χbNW] *
        op[dbTR dbBL; dtTR dtBL]
end

function _contract_edges_120((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E, S, 2, 2}) where {P <: PEPSSandwichTriangular, E, S}
    O_r_c = network[r, c]
    O_rn_c = network[_next(r, end), c]
    return @tensor opt = true ket(O_r_c)[dtTL; DTLt120 DTLt60 DTLt0 DTLt300 DTLt240 DTLt180] * ket(O_rn_c)[dtBR; DTLt300 DBRt60 DBRt0 DBRt300 DBRt240 DBRt180] *
        conj(bra(O_r_c)[dbTL; DTLb120 DTLb60 DTLb0 DTLb300 DTLb240 DTLb180]) * conj(bra(O_rn_c)[dbBR; DTLb300 DBRb60 DBRb0 DBRb300 DBRb240 DBRb180]) *
        env.C[1, _prev(r, end), c][χNW DTLt120 DTLb120; χN] * env.C[2, _prev(r, end), _next(c, end)][χN DTLt60 DTLb60; χaNE] * env.C[3, _next(r, end), _next(c, end)][χbNE DBRt0 DBRb0; χSE] *
        env.C[4, _next(r + 1, end), c][χSE DBRt300 DBRb300; χS] * env.C[5, _next(r + 1, end), _prev(c, end)][χS DBRt240 DBRb240; χaSW] * env.C[6, r, _prev(c, end)][χbSW DTLt180 DTLb180; χNW] *
        env.Eb[2, r, _next(c, end)][χaNE DTLt0 DTLb0; χCNE] * env.Ea[2, r, _next(c, end)][χCNE DBRt60 DBRb60; χbNE] *
        env.Eb[5, _next(r, end), _prev(c, end)][χaSW DBRt180 DBRb180; χCSW] * env.Ea[5, _next(r, end), _prev(c, end)][χCSW DTLt240 DTLb240; χbSW] *
        op[dbTL dbBR; dtTL dtBR]
end

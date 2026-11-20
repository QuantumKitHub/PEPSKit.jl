function _contract_edges_0((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[_cellsite(network, r, c)...][DL180 DL240 DL300; DL120 DL60 DL0] * network[_cellsite(network, r, c + 1)...][DL0 DR240 DR300; DR120 DR60 DR0] *
        env.C[1, _cellsite(network, r - 1, c)...][χNW DL120; χaN] * env.C[2, _cellsite(network, r - 1, c + 1 + 1)...][χbN DR60; χNE] * env.C[3, _cellsite(network, r, c + 1 + 1)...][χNE DR0; χSE] *
        env.C[4, _cellsite(network, r + 1, c + 1)...][χSE DR300; χaS] * env.C[5, _cellsite(network, r + 1, c - 1)...][χbS DL240; χSW] * env.C[6, _cellsite(network, r, c - 1)...][χSW DL180; χNW] *
        env.Eb[1, _cellsite(network, r - 1, c + 1)...][χaN DL60; χCN] * env.Ea[1, _cellsite(network, r - 1, c + 1)...][χCN DR120; χbN] *
        env.Eb[4, _cellsite(network, r + 1, c)...][χaS DR240; χCS] * env.Ea[4, _cellsite(network, r + 1, c)...][χCS DL300; χbS]
end

function _contract_edges_60((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[_cellsite(network, r, c)...][DTR180 DBL60 DTR300; DTR120 DTR60 DTR0] * network[_cellsite(network, r + 1, c - 1)...][DBL180 DBL240 DBL300; DBL120 DBL60 DBL0] *
        env.C[1, _cellsite(network, r - 1, c)...][χbNW DTR120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN DTR60; χNE] * env.C[3, _cellsite(network, r, c + 1)...][χNE DTR0; χaSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c - 1)...][χbSE DBL300; χS] * env.C[5, _cellsite(network, r + 1 + 1, c - 1 - 1)...][χS DBL240; χSW] * env.C[6, _cellsite(network, r + 1, c - 1 - 1)...][χSW DBL180; χaNW] *
        env.Eb[3, _cellsite(network, r + 1, c)...][χaSE DTR300; χCSE] * env.Ea[3, _cellsite(network, r + 1, c)...][χCSE DBL0; χbSE] *
        env.Eb[6, _cellsite(network, r, c - 1)...][χaNW DBL120; χCNW] * env.Ea[6, _cellsite(network, r, c - 1)...][χCNW DTR180; χbNW]
end

function _contract_edges_120((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[_cellsite(network, r, c)...][DTL180 DTL240 DTL300; DTL120 DTL60 DTL0] * network[_cellsite(network, r + 1, c)...][DBR180 DBR240 DBR300; DTL300 DBR60 DBR0] *
        env.C[1, _cellsite(network, r - 1, c)...][χNW DTL120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN DTL60; χaNE] * env.C[3, _cellsite(network, r + 1, c + 1)...][χbNE DBR0; χSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c)...][χSE DBR300; χS] * env.C[5, _cellsite(network, r + 1 + 1, c - 1)...][χS DBR240; χaSW] * env.C[6, _cellsite(network, r, c - 1)...][χbSW DTL180; χNW] *
        env.Eb[2, _cellsite(network, r, c + 1)...][χaNE DTL0; χCNE] * env.Ea[2, _cellsite(network, r, c + 1)...][χCNE DBR60; χbNE] *
        env.Eb[5, _cellsite(network, r + 1, c - 1)...][χaSW DBR180; χCSW] * env.Ea[5, _cellsite(network, r + 1, c - 1)...][χCSW DTL240; χbSW]
end

function _contract_site_large((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[_cellsite(network, r - 1, c)...][DNW180 DW60 DNW300; DNW120 DNW60 DNW0] * network[_cellsite(network, r - 1, c + 1)...][DNW0 DNE240 DNE300; DNE120 DNE60 DNE0] *
        network[_cellsite(network, r, c + 1)...][DE180 DE240 DE300; DNE300 DE60 DE0] * network[_cellsite(network, r + 1, c)...][DSE180 DSE240 DSE300; DSE120 DE240 DSE0] * network[_cellsite(network, r + 1, c - 1)...][DSW180 DSW240 DSW300; DSW120 DSW60 DSE180] *
        network[_cellsite(network, r, c - 1)...][DW180 DW240 DSW120; DW120 DW60 DW0] * network[_cellsite(network, r, c)...][DW0 DSW60 DSE120; DNW300 DNE240 DE180] *
        env.C[1, _cellsite(network, r - 1 - 1, c)...][χaNW DNW120; χbN] * env.Eb[1, _cellsite(network, r - 1 - 1, c + 1)...][χbN DNW60; χCN] * env.Ea[1, _cellsite(network, r - 1 - 1, c + 1)...][χCN DNE120; χaN] *
        env.C[2, _cellsite(network, r - 1 - 1, c + 1 + 1)...][χaN DNE60; χbNE] * env.Eb[2, _cellsite(network, r - 1, c + 1 + 1)...][χbNE DNE0; χCNE] * env.Ea[2, _cellsite(network, r - 1, c + 1 + 1)...][χCNE DE60; χaNE] *
        env.C[3, _cellsite(network, r, c + 1 + 1)...][χaNE DE0; χbSE] * env.Eb[3, _cellsite(network, r + 1, c + 1)...][χbSE DE300; χCSE] * env.Ea[3, _cellsite(network, r + 1, c + 1)...][χCSE DSE0; χaSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c)...][χaSE DSE300; χbS] * env.Eb[4, _cellsite(network, r + 1 + 1, c - 1)...][χbS DSE240; χCS] * env.Ea[4, _cellsite(network, r + 1 + 1, c - 1)...][χCS DSW300; χaS] *
        env.C[5, _cellsite(network, r + 1 + 1, c - 1 - 1)...][χaS DSW240; χbSW] * env.Eb[5, _cellsite(network, r + 1, c - 1 - 1)...][χbSW DSW180; χCSW] * env.Ea[5, _cellsite(network, r + 1, c - 1 - 1)...][χCSW DW240; χaSW] *
        env.C[6, _cellsite(network, r, c - 1 - 1)...][χaSW DW180; χbNW] * env.Eb[6, _cellsite(network, r - 1, c - 1)...][χbNW DW120; χCNW] * env.Ea[6, _cellsite(network, r - 1, c - 1)...][χCNW DNW180; χaNW]
end

function _contract_corners((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true env.C[1, _cellsite(network, r - 1, c)...][χNW D120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN D60; χNE] * env.C[3, _cellsite(network, r, c + 1)...][χNE D0; χSE] *
        env.C[4, _cellsite(network, r + 1, c)...][χSE D300; χS] * env.C[5, _cellsite(network, r + 1, c - 1)...][χS D240; χSW] * env.C[6, _cellsite(network, r, c - 1)...][χSW D180; χNW] *
        network[_cellsite(network, r, c)...][D180 D240 D300; D120 D60 D0]
end

function _contract_edges_0((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    O_r_c = network[_cellsite(network, r, c)...]
    O_r_cn = network[_cellsite(network, r, c + 1)...]
    return @tensor opt = true ket(O_r_c)[dL; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(O_r_cn)[dR; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(O_r_c)[dL; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(O_r_cn)[dR; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1, _cellsite(network, r - 1, c)...][χNW DLt120 DLb120; χaN] * env.C[2, _cellsite(network, r - 1, c + 1 + 1)...][χbN DRt60 DRb60; χNE] * env.C[3, _cellsite(network, r, c + 1 + 1)...][χNE DRt0 DRb0; χSE] *
        env.C[4, _cellsite(network, r + 1, c + 1)...][χSE DRt300 DRb300; χaS] * env.C[5, _cellsite(network, r + 1, c - 1)...][χbS DLt240 DLb240; χSW] * env.C[6, _cellsite(network, r, c - 1)...][χSW DLt180 DLb180; χNW] *
        env.Eb[1, _cellsite(network, r - 1, c + 1)...][χaN DLt60 DLb60; χCN] * env.Ea[1, _cellsite(network, r - 1, c + 1)...][χCN DRt120 DRb120; χbN] *
        env.Eb[4, _cellsite(network, r + 1, c)...][χaS DRt240 DRb240; χCS] * env.Ea[4, _cellsite(network, r + 1, c)...][χCS DLt300 DLb300; χbS]
end

function _contract_edges_60((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    O_r_c = network[_cellsite(network, r, c)...]
    O_rn_cp = network[_cellsite(network, r + 1, c - 1)...]
    return @tensor opt = true ket(O_r_c)[dTR; DTRt120 DTRt60 DTRt0 DTRt300 DBLt60 DTRt180] * ket(O_rn_cp)[dBL; DBLt120 DBLt60 DBLt0 DBLt300 DBLt240 DBLt180] *
        conj(bra(O_r_c)[dTR; DTRb120 DTRb60 DTRb0 DTRb300 DBLb60 DTRb180]) * conj(bra(O_rn_cp)[dBL; DBLb120 DBLb60 DBLb0 DBLb300 DBLb240 DBLb180]) *
        env.C[1, _cellsite(network, r - 1, c)...][χbNW DTRt120 DTRb120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN DTRt60 DTRb60; χNE] * env.C[3, _cellsite(network, r, c + 1)...][χNE DTRt0 DTRb0; χaSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c - 1)...][χbSE DBLt300 DBLb300; χS] * env.C[5, _cellsite(network, r + 1 + 1, c - 1 - 1)...][χS DBLt240 DBLb240; χSW] * env.C[6, _cellsite(network, r + 1, c - 1 - 1)...][χSW DBLt180 DBLb180; χaNW] *
        env.Eb[3, _cellsite(network, r + 1, c)...][χaSE DTRt300 DTRb300; χCSE] * env.Ea[3, _cellsite(network, r + 1, c)...][χCSE DBLt0 DBLb0; χbSE] *
        env.Eb[6, _cellsite(network, r, c - 1)...][χaNW DBLt120 DBLb120; χCNW] * env.Ea[6, _cellsite(network, r, c - 1)...][χCNW DTRt180 DTRb180; χbNW]
end

function _contract_edges_120((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    O_r_c = network[_cellsite(network, r, c)...]
    O_rn_c = network[_cellsite(network, r + 1, c)...]
    return @tensor opt = true ket(O_r_c)[dTL; DTLt120 DTLt60 DTLt0 DTLt300 DTLt240 DTLt180] * ket(O_rn_c)[dBR; DTLt300 DBRt60 DBRt0 DBRt300 DBRt240 DBRt180] *
        conj(bra(O_r_c)[dTL; DTLb120 DTLb60 DTLb0 DTLb300 DTLb240 DTLb180]) * conj(bra(O_rn_c)[dBR; DTLb300 DBRb60 DBRb0 DBRb300 DBRb240 DBRb180]) *
        env.C[1, _cellsite(network, r - 1, c)...][χNW DTLt120 DTLb120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN DTLt60 DTLb60; χaNE] * env.C[3, _cellsite(network, r + 1, c + 1)...][χbNE DBRt0 DBRb0; χSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c)...][χSE DBRt300 DBRb300; χS] * env.C[5, _cellsite(network, r + 1 + 1, c - 1)...][χS DBRt240 DBRb240; χaSW] * env.C[6, _cellsite(network, r, c - 1)...][χbSW DTLt180 DTLb180; χNW] *
        env.Eb[2, _cellsite(network, r, c + 1)...][χaNE DTLt0 DTLb0; χCNE] * env.Ea[2, _cellsite(network, r, c + 1)...][χCNE DBRt60 DBRb60; χbNE] *
        env.Eb[5, _cellsite(network, r + 1, c - 1)...][χaSW DBRt180 DBRb180; χCSW] * env.Ea[5, _cellsite(network, r + 1, c - 1)...][χCSW DTLt240 DTLb240; χbSW]
end

function _contract_site_large((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    O_r_c = network[_cellsite(network, r, c)...]
    O_rp_c = network[_cellsite(network, r - 1, c)...]
    O_rp_cn = network[_cellsite(network, r - 1, c + 1)...]
    O_rn_cp = network[_cellsite(network, r + 1, c - 1)...]
    O_rn_c = network[_cellsite(network, r + 1, c)...]
    O_r_cn = network[_cellsite(network, r, c + 1)...]
    O_r_cp = network[_cellsite(network, r, c - 1)...]
    return @tensor opt = true ket(O_rp_c)[dNW; DNWt120 DNWt60 DNWt0 DNWt300 DWt60 DNWt180] * ket(O_rp_cn)[dNE; DNEt120 DNEt60 DNEt0 DNEt300 DNEt240 DNWt0] *
        ket(O_r_cn)[dE; DNEt300 DEt60 DEt0 DEt300 DEt240 DEt180] * ket(O_rn_c)[dSE; DSEt120 DEt240 DSEt0 DSEt300 DSEt240 DSEt180] * ket(O_rn_cp)[dSW; DSWt120 DSWt60 DSEt180 DSWt300 DSWt240 DSWt180] *
        ket(O_r_cp)[dW; DWt120 DWt60 DWt0 DSWt120 DWt240 DWt180] * ket(O_r_c)[dCenter; DNWt300 DNEt240 DEt180 DSEt120 DSWt60 DWt0] *
        conj(bra(O_rp_c)[dNW; DNWb120 DNWb60 DNWb0 DNWb300 DWb60 DNWb180]) * conj(bra(O_rp_cn)[dNE; DNEb120 DNEb60 DNEb0 DNEb300 DNEb240 DNWb0]) *
        conj(bra(O_r_cn)[dE; DNEb300 DEb60 DEb0 DEb300 DEb240 DEb180]) * conj(bra(O_rn_c)[dSE; DSEb120 DEb240 DSEb0 DSEb300 DSEb240 DSEb180]) * conj(bra(O_rn_cp)[dSW; DSWb120 DSWb60 DSEb180 DSWb300 DSWb240 DSWb180]) *
        conj(bra(O_r_cp)[dW; DWb120 DWb60 DWb0 DSWb120 DWb240 DWb180]) * conj(bra(O_r_c)[dCenter; DNWb300 DNEb240 DEb180 DSEb120 DSWb60 DWb0]) *
        env.C[1, _cellsite(network, r - 1 - 1, c)...][χaNW DNWt120 DNWb120; χbN] * env.Eb[1, _cellsite(network, r - 1 - 1, c + 1)...][χbN DNWt60 DNWb60; χCN] * env.Ea[1, _cellsite(network, r - 1 - 1, c + 1)...][χCN DNEt120 DNEb120; χaN] *
        env.C[2, _cellsite(network, r - 1 - 1, c + 1 + 1)...][χaN DNEt60 DNEb60; χbNE] * env.Eb[2, _cellsite(network, r - 1, c + 1 + 1)...][χbNE DNEt0 DNEb0; χCNE] * env.Ea[2, _cellsite(network, r - 1, c + 1 + 1)...][χCNE DEt60 DEb60; χaNE] *
        env.C[3, _cellsite(network, r, c + 1 + 1)...][χaNE DEt0 DEb0; χbSE] * env.Eb[3, _cellsite(network, r + 1, c + 1)...][χbSE DEt300 DEb300; χCSE] * env.Ea[3, _cellsite(network, r + 1, c + 1)...][χCSE DSEt0 DSEb0; χaSE] *
        env.C[4, _cellsite(network, r + 1 + 1, c)...][χaSE DSEt300 DSEb300; χbS] * env.Eb[4, _cellsite(network, r + 1 + 1, c - 1)...][χbS DSEt240 DSEb240; χCS] * env.Ea[4, _cellsite(network, r + 1 + 1, c - 1)...][χCS DSWt300 DSWb300; χaS] *
        env.C[5, _cellsite(network, r + 1 + 1, c - 1 - 1)...][χaS DSWt240 DSWb240; χbSW] * env.Eb[5, _cellsite(network, r + 1, c - 1 - 1)...][χbSW DSWt180 DSWb180; χCSW] * env.Ea[5, _cellsite(network, r + 1, c - 1 - 1)...][χCSW DWt240 DWb240; χaSW] *
        env.C[6, _cellsite(network, r, c - 1 - 1)...][χaSW DWt180 DWb180; χbNW] * env.Eb[6, _cellsite(network, r - 1, c - 1)...][χbNW DWt120 DWb120; χCNW] * env.Ea[6, _cellsite(network, r - 1, c - 1)...][χCNW DNWt180 DNWb180; χaNW]
end

function _contract_corners((r, c)::Tuple{Int, Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    O_r_c = network[_cellsite(network, r, c)...]
    return @tensor opt = true env.C[1, _cellsite(network, r - 1, c)...][χNW Dt120 Db120; χN] * env.C[2, _cellsite(network, r - 1, c + 1)...][χN Dt60 Db60; χNE] * env.C[3, _cellsite(network, r, c + 1)...][χNE Dt0 Db0; χSE] *
        env.C[4, _cellsite(network, r + 1, c)...][χSE Dt300 Db300; χS] * env.C[5, _cellsite(network, r + 1, c - 1)...][χS Dt240 Db240; χSW] * env.C[6, _cellsite(network, r, c - 1)...][χSW Dt180 Db180; χNW] *
        ket(O_r_c)[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] * conj(bra(O_r_c)[d; Db120 Db60 Db0 Db300 Db240 Db180])
end

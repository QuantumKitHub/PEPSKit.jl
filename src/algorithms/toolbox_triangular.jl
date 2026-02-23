function network_value(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular)
    return prod(Iterators.product(axes(network)...)) do (r, c)
        nw_corners = complex(_contract_corners((r,c), network, env))
        nw_full = complex(_contract_site_large((r,c), network, env))
        nw_0 = complex(_contract_edges_0((r,c), network, env))
        nw_60 = complex(_contract_edges_60((r,c), network, env))
        nw_120 = complex(_contract_edges_120((r,c), network, env))
        return (nw_full * nw_corners^2 / (nw_0 * nw_60 * nw_120))^(1 / 3)
    end
end

function _contract_edges_0((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[r,c][DL180 DL240 DL300; DL120 DL60 DL0] * network[r,_next(c,end)][DL0 DR240 DR300; DR120 DR60 DR0] *
        env.C[1,_prev(r,end),c][χNW DL120; χNa] * env.C[2,_prev(r,end),_next(c+1,end)][χNb DR60; χNE] * env.C[3,r,_next(c+1,end)][χNE DR0; χSE] *
        env.C[4,_next(r,end),_next(c,end)][χSE DR300; χSa] * env.C[5,_next(r,end),_prev(c,end)][χSb DL240; χSW] * env.C[6,r,_prev(c,end)][χSW DL180; χNW] *
        env.Eb[1,_prev(r,end),_next(c,end)][χNa DL60; χNC] * env.Ea[1,_prev(r,end),_next(c,end)][χNC DR120; χNb] *
        env.Eb[4,_next(r,end),c][χSa DR240; χSC] * env.Ea[4,_next(r,end),c][χSC DL300; χSb]
end

function _contract_edges_60((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[r,c][DTR180 DBL60 DTR300; DTR120 DTR60 DTR0] * network[_next(r,end),_prev(c,end)][DBL180 DBL240 DBL300; DBL120 DBL60 DBL0] *
        env.C[1,_prev(r,end),c][χNWb DTR120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN DTR60; χNE] * env.C[3,r,_next(c,end)][χNE DTR0; χSEa] *
        env.C[4,_next(r+1,end),_prev(c,end)][χSEb DBL300; χS] * env.C[5,_next(r+1,end),_prev(c-1,end)][χS DBL240; χSW] * env.C[6,_next(r,end),_prev(c-1,end)][χSW DBL180; χNWa] *
        env.Eb[3,_next(r,end),c][χSEa DTR300; χSEC] * env.Ea[3,_next(r,end),c][χSEC DBL0; χSEb] *
        env.Eb[6,r,_prev(c,end)][χNWa DBL120; χNWC] * env.Ea[6,r,_prev(c,end)][χNWC DTR180; χNWb]
end

function _contract_edges_120((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[r,c][DTL180 DTL240 DTL300; DTL120 DTL60 DTL0] * network[_next(r,end),c][DBR180 DBR240 DBR300; DTL300 DBR60 DBR0] *
        env.C[1,_prev(r,end),c][χNW DTL120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN DTL60; χNEa] * env.C[3,_next(r,end),_next(c,end)][χNEb DBR0; χSE] *
        env.C[4,_next(r+1,end),c][χSE DBR300; χS] * env.C[5,_next(r+1,end),_prev(c,end)][χS DBR240; χSWa] * env.C[6,r,_prev(c,end)][χSWb DTL180; χNW] *
        env.Eb[2,r,_next(c,end)][χNEa DTL0; χNEC] * env.Ea[2,r,_next(c,end)][χNEC DBR60; χNEb] *
        env.Eb[5,_next(r,end),_prev(c,end)][χSWa DBR180; χSWC] * env.Ea[5,_next(r,end),_prev(c,end)][χSWC DTL240; χSWb]
end

function _contract_site_large((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true network[_prev(r,end),c][DNW180 DW60 DNW300; DNW120 DNW60 DNW0] * network[_prev(r,end),_next(c,end)][DNW0 DNE240 DNE300; DNE120 DNE60 DNE0] *
        network[r,_next(c,end)][DE180 DE240 DE300; DNE300 DE60 DE0] * network[_next(r,end),c][DSE180 DSE240 DSE300; DSE120 DE240 DSE0] * network[_next(r,end),_prev(c,end)][DSW180 DSW240 DSW300; DSW120 DSW60 DSE180] *
        network[r,_prev(c,end)][DW180 DW240 DSW120; DW120 DW60 DW0] * network[r,c][DW0 DSW60 DSE120; DNW300 DNE240 DE180] *
        env.C[1,_prev(r-1,end),c][χNWa DNW120; χNb] * env.Eb[1,_prev(r-1,end),_next(c,end)][χNb DNW60; χNC] * env.Ea[1,_prev(r-1,end),_next(c,end)][χNC DNE120; χNa] *
        env.C[2,_prev(r-1,end),_next(c+1,end)][χNa DNE60; χNEb] * env.Eb[2,_prev(r,end),_next(c+1,end)][χNEb DNE0; χNEC] * env.Ea[2,_prev(r,end),_next(c+1,end)][χNEC DE60; χNEa] *
        env.C[3,r,_next(c+1,end)][χNEa DE0; χSEb] * env.Eb[3,_next(r,end),_next(c,end)][χSEb DE300; χSEC] * env.Ea[3,_next(r,end),_next(c,end)][χSEC DSE0; χSEa] *
        env.C[4,_next(r+1,end),c][χSEa DSE300; χSb] * env.Eb[4,_next(r+1,end),_prev(c,end)][χSb DSE240; χSC] * env.Ea[4,_next(r+1,end),_prev(c,end)][χSC DSW300; χSa] *
        env.C[5,_next(r+1,end),_prev(c-1,end)][χSa DSW240; χSWb] * env.Eb[5,_next(r,end),_prev(c-1,end)][χSWb DSW180; χSWC] * env.Ea[5,_next(r,end),_prev(c-1,end)][χSWC DW240; χSWa] *
        env.C[6,r,_prev(c-1,end)][χSWa DW180; χNWb] * env.Eb[6,_prev(r,end),_prev(c,end)][χNWb DW120; χNWC] * env.Ea[6,_prev(r,end),_prev(c,end)][χNWC DNW180; χNWa]
end

function _contract_corners((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PFTensorTriangular}
    return @tensor opt = true env.C[1,_prev(r,end),c][χNW D120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN D60; χNE] * env.C[3,r,_next(c,end)][χNE D0; χSE] *
        env.C[4,_next(r,end),c][χSE D300; χS] * env.C[5,_next(r,end),_prev(c,end)][χS D240; χSW] * env.C[6,r,_prev(c,end)][χSW D180; χNW] *
        network[r,c][D180 D240 D300; D120 D60 D0]
end

### For 

function _contract_edges_0((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    return @tensor opt = true ket(network[r,c])[dL; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(network[r,_next(c,end)])[dR; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(network[r,c])[dL; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(network[r,_next(c,end)])[dR; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1,_prev(r,end),c][χNW DLt120 DLb120; χNa] * env.C[2,_prev(r,end),_next(c+1,end)][χNb DRt60 DRb60; χNE] * env.C[3,r,_next(c+1,end)][χNE DRt0 DRb0; χSE] *
        env.C[4,_next(r,end),_next(c,end)][χSE DRt300 DRb300; χSa] * env.C[5,_next(r,end),_prev(c,end)][χSb DLt240 DLb240; χSW] * env.C[6,r,_prev(c,end)][χSW DLt180 DLb180; χNW] *
        env.Eb[1,_prev(r,end),_next(c,end)][χNa DLt60 DLb60; χNC] * env.Ea[1,_prev(r,end),_next(c,end)][χNC DRt120 DRb120; χNb] *
        env.Eb[4,_next(r,end),c][χSa DRt240 DRb240; χSC] * env.Ea[4,_next(r,end),c][χSC DLt300 DLb300; χSb]
end

function _contract_edges_0((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E,S,2,2}) where {P <: PEPSSandwichTriangular, E, S}
    return @tensor opt = true ket(network[r,c])[dLt; DLt120 DLt60 DLt0 DLt300 DLt240 DLt180] * ket(network[r,_next(c,end)])[dRt; DRt120 DRt60 DRt0 DRt300 DRt240 DLt0] *
        conj(bra(network[r,c])[dLb; DLb120 DLb60 DLb0 DLb300 DLb240 DLb180]) * conj(bra(network[r,_next(c,end)])[dRb; DRb120 DRb60 DRb0 DRb300 DRb240 DLb0]) *
        env.C[1,_prev(r,end),c][χNW DLt120 DLb120; χNa] * env.C[2,_prev(r,end),_next(c+1,end)][χNb DRt60 DRb60; χNE] * env.C[3,r,_next(c+1,end)][χNE DRt0 DRb0; χSE] *
        env.C[4,_next(r,end),_next(c,end)][χSE DRt300 DRb300; χSa] * env.C[5,_next(r,end),_prev(c,end)][χSb DLt240 DLb240; χSW] * env.C[6,r,_prev(c,end)][χSW DLt180 DLb180; χNW] *
        env.Eb[1,_prev(r,end),_next(c,end)][χNa DLt60 DLb60; χNC] * env.Ea[1,_prev(r,end),_next(c,end)][χNC DRt120 DRb120; χNb] *
        env.Eb[4,_next(r,end),c][χSa DRt240 DRb240; χSC] * env.Ea[4,_next(r,end),c][χSC DLt300 DLb300; χSb] * 
        op[dLb dRb; dLt dRt]
end

function _contract_edges_60((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    return @tensor opt = true ket(network[r,c])[dTR; DTRt120 DTRt60 DTRt0 DTRt300 DBLt60 DTRt180] * ket(network[_next(r,end),_prev(c,end)])[dBL; DBLt120 DBLt60 DBLt0 DBLt300 DBLt240 DBLt180] *
        conj(bra(network[r,c])[dTR; DTRb120 DTRb60 DTRb0 DTRb300 DBLb60 DTRb180]) * conj(bra(network[_next(r,end),_prev(c,end)])[dBL; DBLb120 DBLb60 DBLb0 DBLb300 DBLb240 DBLb180]) *
        env.C[1,_prev(r,end),c][χNWb DTRt120 DTRb120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN DTRt60 DTRb60; χNE] * env.C[3,r,_next(c,end)][χNE DTRt0 DTRb0; χSEa] *
        env.C[4,_next(r+1,end),_prev(c,end)][χSEb DBLt300 DBLb300; χS] * env.C[5,_next(r+1,end),_prev(c-1,end)][χS DBLt240 DBLb240; χSW] * env.C[6,_next(r,end),_prev(c-1,end)][χSW DBLt180 DBLb180; χNWa] *
        env.Eb[3,_next(r,end),c][χSEa DTRt300 DTRb300; χSEC] * env.Ea[3,_next(r,end),c][χSEC DBLt0 DBLb0; χSEb] *
        env.Eb[6,r,_prev(c,end)][χNWa DBLt120 DBLb120; χNWC] * env.Ea[6,r,_prev(c,end)][χNWC DTRt180 DTRb180; χNWb]
end

function _contract_edges_120((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    return @tensor opt = true ket(network[r,c])[dTL; DTLt120 DTLt60 DTLt0 DTLt300 DTLt240 DTLt180] * ket(network[_next(r,end),c])[dBR; DTLt300 DBRt60 DBRt0 DBRt300 DBRt240 DBRt180] *
        conj(bra(network[r,c])[dTL; DTLb120 DTLb60 DTLb0 DTLb300 DTLb240 DTLb180]) * conj(bra(network[_next(r,end),c])[dBR; DTLb300 DBRb60 DBRb0 DBRb300 DBRb240 DBRb180]) *
        env.C[1,_prev(r,end),c][χNW DTLt120 DTLb120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN DTLt60 DTLb60; χNEa] * env.C[3,_next(r,end),_next(c,end)][χNEb DBRt0 DBRb0; χSE] *
        env.C[4,_next(r+1,end),c][χSE DBRt300 DBRb300; χS] * env.C[5,_next(r+1,end),_prev(c,end)][χS DBRt240 DBRb240; χSWa] * env.C[6,r,_prev(c,end)][χSWb DTLt180 DTLb180; χNW] *
        env.Eb[2,r,_next(c,end)][χNEa DTLt0 DTLb0; χNEC] * env.Ea[2,r,_next(c,end)][χNEC DBRt60 DBRb60; χNEb] *
        env.Eb[5,_next(r,end),_prev(c,end)][χSWa DBRt180 DBRb180; χSWC] * env.Ea[5,_next(r,end),_prev(c,end)][χSWC DTLt240 DTLb240; χSWb]
end

function _contract_site_large((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    return @tensor opt = true ket(network[_prev(r,end),c])[dNW; DNWt120 DNWt60 DNWt0 DNWt300 DWt60 DNWt180] * ket(network[_prev(r,end),_next(c,end)])[dNE; DNEt120 DNEt60 DNEt0 DNEt300 DNEt240 DNWt0] *
        ket(network[r,_next(c,end)])[dE; DNEt300 DEt60 DEt0 DEt300 DEt240 DEt180] * ket(network[_next(r,end),c])[dSE; DSEt120 DEt240 DSEt0 DSEt300 DSEt240 DSEt180] * ket(network[_next(r,end),_prev(c,end)])[dSW; DSWt120 DSWt60 DSEt180 DSWt300 DSWt240 DSWt180] *
        ket(network[r,_prev(c,end)])[dW; DWt120 DWt60 DWt0 DSWt120 DWt240 DWt180] * ket(network[r,c])[dCenter; DNWt300 DNEt240 DEt180 DSEt120 DSWt60 DWt0] *
        conj(bra(network[_prev(r,end),c])[dNW; DNWb120 DNWb60 DNWb0 DNWb300 DWb60 DNWb180]) * conj(bra(network[_prev(r,end),_next(c,end)])[dNE; DNEb120 DNEb60 DNEb0 DNEb300 DNEb240 DNWb0]) *
        conj(bra(network[r,_next(c,end)])[dE; DNEb300 DEb60 DEb0 DEb300 DEb240 DEb180]) * conj(bra(network[_next(r,end),c])[dSE; DSEb120 DEb240 DSEb0 DSEb300 DSEb240 DSEb180]) * conj(bra(network[_next(r,end),_prev(c,end)])[dSW; DSWb120 DSWb60 DSEb180 DSWb300 DSWb240 DSWb180]) *
        conj(bra(network[r,_prev(c,end)])[dW; DWb120 DWb60 DWb0 DSWb120 DWb240 DWb180]) * conj(bra(network[r,c])[dCenter; DNWb300 DNEb240 DEb180 DSEb120 DSWb60 DWb0]) *
        env.C[1,_prev(r-1,end),c][χNWa DNWt120 DNWb120; χNb] * env.Eb[1,_prev(r-1,end),_next(c,end)][χNb DNWt60 DNWb60; χNC] * env.Ea[1,_prev(r-1,end),_next(c,end)][χNC DNEt120 DNEb120; χNa] *
        env.C[2,_prev(r-1,end),_next(c+1,end)][χNa DNEt60 DNEb60; χNEb] * env.Eb[2,_prev(r,end),_next(c+1,end)][χNEb DNEt0 DNEb0; χNEC] * env.Ea[2,_prev(r,end),_next(c+1,end)][χNEC DEt60 DEb60; χNEa] *
        env.C[3,r,_next(c+1,end)][χNEa DEt0 DEb0; χSEb] * env.Eb[3,_next(r,end),_next(c,end)][χSEb DEt300 DEb300; χSEC] * env.Ea[3,_next(r,end),_next(c,end)][χSEC DSEt0 DSEb0; χSEa] *
        env.C[4,_next(r+1,end),c][χSEa DSEt300 DSEb300; χSb] * env.Eb[4,_next(r+1,end),_prev(c,end)][χSb DSEt240 DSEb240; χSC] * env.Ea[4,_next(r+1,end),_prev(c,end)][χSC DSWt300 DSWb300; χSa] *
        env.C[5,_next(r+1,end),_prev(c-1,end)][χSa DSWt240 DSWb240; χSWb] * env.Eb[5,_next(r,end),_prev(c-1,end)][χSWb DSWt180 DSWb180; χSWC] * env.Ea[5,_next(r,end),_prev(c-1,end)][χSWC DWt240 DWb240; χSWa] *
        env.C[6,r,_prev(c-1,end)][χSWa DWt180 DWb180; χNWb] * env.Eb[6,_prev(r,end),_prev(c,end)][χNWb DWt120 DWb120; χNWC] * env.Ea[6,_prev(r,end),_prev(c,end)][χNWC DNWt180 DNWb180; χNWa]
end

function _contract_corners((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular) where {P <: PEPSSandwichTriangular}
    return @tensor opt = true env.C[1,_prev(r,end),c][χNW Dt120 Db120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN Dt60 Db60; χNE] * env.C[3,r,_next(c,end)][χNE Dt0 Db0; χSE] *
        env.C[4,_next(r,end),c][χSE Dt300 Db300; χS] * env.C[5,_next(r,end),_prev(c,end)][χS Dt240 Db240; χSW] * env.C[6,r,_prev(c,end)][χSW Dt180 Db180; χNW] *
        ket(network[r,c])[d; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] * conj(bra(network[r,c])[d; Db120 Db60 Db0 Db300 Db240 Db180])
end

function _contract_corners((r,c)::Tuple{Int,Int}, network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, op::AbstractTensorMap{E,S,1,1}) where {P <: PEPSSandwichTriangular, E, S}
    return @tensor opt = true env.C[1,_prev(r,end),c][χNW Dt120 Db120; χN] * env.C[2,_prev(r,end),_next(c,end)][χN Dt60 Db60; χNE] * env.C[3,r,_next(c,end)][χNE Dt0 Db0; χSE] *
        env.C[4,_next(r,end),c][χSE Dt300 Db300; χS] * env.C[5,_next(r,end),_prev(c,end)][χS Dt240 Db240; χSW] * env.C[6,r,_prev(c,end)][χSW Dt180 Db180; χNW] *
        ket(network[r,c])[dt; Dt120 Dt60 Dt0 Dt300 Dt240 Dt180] * conj(bra(network[r,c])[db; Db120 Db60 Db0 Db300 Db240 Db180]) * op[db; dt]
end

function energy(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, onesite_op, twosite_op) where {P <: PEPSSandwichTriangular}
    return sum(Iterators.product(axes(network)...)) do (r, c)
        expval_onesite = _contract_corners((r,c), network, env, onesite_op) / _contract_corners((r,c), network, env)
        expval_twosite = _contract_edges_0((r,c), network, env, twosite_op) / _contract_edges_0((r,c), network, env)
        return expval_onesite + expval_twosite
    end
end

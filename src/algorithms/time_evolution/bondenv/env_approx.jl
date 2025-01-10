"""
Use CTMRG to approximately contract 
a cluster around a bond to construct bond environment
"""
@kwdef struct CTMBondEnv <: BondEnvAlgorithm
    pad::Int = 1
end

function bondenv_ctm(row::Int, col::Int, peps::InfinitePEPS, alg::CTMBondEnv)
    return error("Not implemented")
end

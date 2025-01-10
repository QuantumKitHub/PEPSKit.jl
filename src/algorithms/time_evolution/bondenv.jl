"""
    BondEnvAlgorithm

Abstract super type for all algorithms to construct the environment of a bond in the InfinitePEPS.
"""
abstract type BondEnvAlgorithm end

const Hair{S} = AbstractTensorMap{S,1,1} where {S<:ElementarySpace}
const BondEnv{S} = AbstractTensorMap{S,2,2} where {S<:ElementarySpace}
const PEPSOrth{S} = AbstractTensor{S,4} where {S<:ElementarySpace}

include("bondenv/env_ctm.jl")
include("bondenv/optimize.jl")

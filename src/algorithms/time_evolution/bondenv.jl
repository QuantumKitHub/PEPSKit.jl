"""
    BondEnvAlgorithm

Abstract super type for all algorithms to construct the environment of a bond in the InfinitePEPS.
"""
abstract type BondEnvAlgorithm end

const Hair{S} = AbstractTensor{S,2} where {S<:ElementarySpace}
const BondEnv{S} = AbstractTensorMap{S,2,2} where {S<:ElementarySpace}
const PEPSOrth{S} = AbstractTensor{S,4} where {S<:ElementarySpace}
const BondPhys{S} = AbstractTensor{S,3} where {S<:ElementarySpace}
const BondPhys2{S} = AbstractTensor{S,4} where {S<:ElementarySpace}

include("bondenv/ntutools.jl")
include("bondenv/env_ntu.jl")
include("bondenv/env_ctm.jl")
include("bondenv/eat.jl")
include("bondenv/optimize.jl")

const BondEnv{T, S} = AbstractTensorMap{T, S, 2, 2} where {T <: Number, S <: ElementarySpace}
const PEPSOrth{T, S} = AbstractTensor{T, S, 4} where {T <: Number, S <: ElementarySpace}
const PEPOOrth{T, S} = AbstractTensor{T, S, 5} where {T <: Number, S <: ElementarySpace}

const BondEnv{T, S} = AbstractTensorMap{T, S, 2, 2} where {T <: Number, S <: ElementarySpace}
# Orthogonal tensors obtained PEPSTensor/PEPOTensor
# with one physical leg being factored out by `_qr_bond`
const PEPSOrth{T, S} = AbstractTensor{T, S, 4} where {T <: Number, S <: ElementarySpace}
const PEPOOrth{T, S} = AbstractTensor{T, S, 5} where {T <: Number, S <: ElementarySpace}
const StateTensor = Union{PEPSTensor, PEPOTensor, PEPSOrth, PEPOOrth}

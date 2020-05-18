const PEPSType{S} = MPSKit.GenericMPSTensor{S,4} where {S<:EuclideanSpace}
const EffM{S} = TensorKit.AbstractTensorMap{S,5,5} where {S<:EuclideanSpace}

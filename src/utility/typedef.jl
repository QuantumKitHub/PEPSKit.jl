const PEPSType{S} = GenericMPSTensor{S,4} where {S<:EuclideanSpace}
const EffM{S} = AbstractTensorMap{S,5,5} where {S<:EuclideanSpace}

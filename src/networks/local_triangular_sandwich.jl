#
# Interface for local effective-rank-6 tensor sandwiches
#

# route all virtualspace getters through a single method for convenience
northwest_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[1]
northeast_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[2]
east_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[3]
southeast_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[4]
southwest_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[5]
west_virtualspace(O::AbstractTensorMap{E, S, 1, 6}, args...) where {E, S} = domain(O)[6]

## Rotations

# generic local interface
_rotl60_localsandwich(O) = rotl60.(O)
_rotr60_localsandwich(O) = rotr60.(O)

## PartitionFunction

# specialized local rotation interface
_rotl60_localsandwich(O::PFTensorTriangular) = rotl60(O)
_rotr60_localsandwich(O::PFTensorTriangular) = rotr60(O)
_rot180_localsandwich(O::PFTensorTriangular) = rot180(O)

# specialized local math interface
_add_localsandwich(O1::PFTensorTriangular, O2::PFTensorTriangular) = O1 + O2
_subtract_localsandwich(O1::PFTensorTriangular, O2::PFTensorTriangular) = O1 - O2
_mul_localsandwich(α::Number, O::PFTensorTriangular) = α * O
_isapprox_localsandwich(O1::PFTensorTriangular, O2::PFTensorTriangular; kwargs...) = isapprox(O1, O2; kwargs...)

## PEPS

const PEPSSandwichTriangular{T <: PEPSTensorTriangular} = Tuple{T, T}

ket(O::PEPSSandwichTriangular) = O[1]
bra(O::PEPSSandwichTriangular) = O[2]

function virtualspace(O::PEPSSandwichTriangular, dir)
    return virtualspace(ket(O), dir) ⊗ virtualspace(bra(O), dir)'
end

TensorKit.spacetype(::Type{P}) where {P <: PEPSSandwichTriangular} = spacetype(eltype(P))

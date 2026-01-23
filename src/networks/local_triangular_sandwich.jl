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
_rotl60_localsandwich(O::PFTriaTensor) = rotl60(O)
_rotr60_localsandwich(O::PFTriaTensor) = rotr60(O)
_rot180_localsandwich(O::PFTriaTensor) = rot180(O)

# specialized local math interface
_add_localsandwich(O1::PFTriaTensor, O2::PFTriaTensor) = O1 + O2
_subtract_localsandwich(O1::PFTriaTensor, O2::PFTriaTensor) = O1 - O2
_mul_localsandwich(α::Number, O::PFTriaTensor) = α * O
_isapprox_localsandwich(O1::PFTriaTensor, O2::PFTriaTensor; kwargs...) = isapprox(O1, O2; kwargs...)

## PEPS

const PEPSTriaSandwich{T <: PEPSTriaTensor} = Tuple{T, T}

ket(O::PEPSTriaSandwich) = O[1]
bra(O::PEPSTriaSandwich) = O[2]

function virtualspace(O::PEPSTriaSandwich, dir)
    return virtualspace(ket(O), dir) ⊗ virtualspace(bra(O), dir)'
end

TensorKit.spacetype(::Type{P}) where {P <: PEPSTriaSandwich} = spacetype(eltype(P))

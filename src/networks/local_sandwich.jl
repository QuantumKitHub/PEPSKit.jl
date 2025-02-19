#
# Interface for local effective-rank-4 tensor sandwiches
#

## Space utils

_elementwise_dual(S::ElementarySpace) = dual(S)
_elementwise_dual(P::ProductSpace) = ProductSpace(dual.(P)...)

# route all virtualspace getters through a single method for convenience
north_virtualspace(O, args...) = virtualspace(O, args..., NORTH)
east_virtualspace(O, args...) = virtualspace(O, args..., EAST)
south_virtualspace(O, args...) = virtualspace(O, args..., SOUTH)
west_virtualspace(O, args...) = virtualspace(O, args..., WEST)

MPSKit.left_virtualspace(O, args...) = west_virtualspace(O, args...)
function MPSKit.right_virtualspace(O, args...)
    return _elementwise_dual(east_virtualspace(O, args...))
end # follow MPSKit convention: right vspace gets a dual by default

## PartitionFunction

const PFSandwich{T<:PFTensor} = Tuple{T}

tensor(O::PFSandwich) = O[1]

function virtualspace(O::PFSandwich, dir)
    return virtualspace(tensor(O), dir)
end

## PEPS

const PEPSSandwich{T<:PEPSTensor} = Tuple{T,T}

ket(O::PEPSSandwich) = O[1]
bra(O::PEPSSandwich) = O[2]

function virtualspace(O::PEPSSandwich, dir)
    return virtualspace(ket(O), dir) ⊗ virtualspace(bra(O), dir)'
end

## PEPO

const PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,T,Vararg{P,N}}

ket(O::PEPOSandwich) = O[1]
bra(O::PEPOSandwich) = O[2]
pepo(O::PEPOSandwich) = O[3:end]
pepo(O::PEPOSandwich, i::Int) = O[2 + i]

function virtualspace(O::PEPOSandwich, dir)
    return prod([
        virtualspace(ket(O), dir),
        virtualspace.(pepo(O), Ref(dir))...,
        virtualspace(bra(O), dir)',
    ])
end

## Rotations

_rotl90_localsandwich(O) = rotl90.(O)
_rotr90_localsandwich(O) = rotr90.(O)
_rot180_localsandwich(O) = rot180.(O)

## Math (for Zygote accumulation)

_add_localsandwich(O1, O2) = O1 .+ O2
_subtract_localsandwich(O1, O2) = O1 .- O2
_mul_localsandwich(α, O) = α .* O

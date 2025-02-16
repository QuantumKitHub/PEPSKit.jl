#
# Interface for local effective-rank-4 tensor sandwiches
#

## Space utils

_elementwise_dual(S::ElementarySpace) = dual(S)
_elementwise_dual(P::ProductSpace) = ProductSpace(dual.(P)...)

north_virtualspace(O, args...) = virtualspace(O, args..., NORTH)
east_virtualspace(O, args...) = virtualspace(O, args..., EAST)
south_virtualspace(O, args...) = virtualspace(O, args..., SOUTH)
west_virtualspace(O, args...) = virtualspace(O, args..., WEST)

MPSKit.left_virtualspace(O, args...) = west_virtualspace(O, args...)
function MPSKit.right_virtualspace(O, args...)
    return _elementwise_dual(east_virtualspace(O, args...))
end # follow MPSKit convention: right vspace gets a dual by default

## PartitionFunction

_rotl90_localsandwich(O::PFTensor) = rotl90(O)
_rotr90_localsandwich(O::PFTensor) = rotr90(O)
_rot180_localsandwich(O::PFTensor) = rot180(O)

_add_localsandwich(O1::PFTensor, O2::PFTensor) = O1 + O2
_subtract_localsandwich(O1::PFTensor, O2::PFTensor) = O1 - O2
_mul_localsandwich(α::PFTensor, O::PFTensor) = α * O

## PEPS

const PEPSSandwich{T<:PEPSTensor} = Tuple{T,T}

ket(O::PEPSSandwich) = O[1]
bra(O::PEPSSandwich) = O[2]

function virtualspace(O::PEPSSandwich, dir)
    return virtualspace(ket(O), dir) ⊗ virtualspace(bra(O), dir)'
end

_rotl90_localsandwich(O::PEPSSandwich) = rotl90.(O)
_rotr90_localsandwich(O::PEPSSandwich) = rotr90.(O)
_rot180_localsandwich(O::PEPSSandwich) = rot180.(O)

## PEPO

const PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,T,Tuple{Vararg{P,N}}}

ket(O::PEPOSandwich) = O[1]
bra(O::PEPOSandwich) = O[2]
pepo(O::PEPOSandwich) = O[3]
pepo(O::PEPOSandwich, i::Int) = O[3][i]

function virtualspace(O::PEPOSandwich, dir)
    return prod([
        virtualspace(ket(O), dir),
        virtualspace.(pepo(O), Ref(dir))...,
        virtualspace(bra(O), dir)',
    ])
end

_rotl90_localsandwich(O::PEPOSandwich) = (rotl90(ket(O)), rotl90(bra(O)), rotl90.(pepo(O)))
_rotr90_localsandwich(O::PEPOSandwich) = (rotr90(ket(O)), rotr90(bra(O)), rotr90.(pepo(O)))
_rot180_localsandwich(O::PEPOSandwich) = (rot180(ket(O)), rot180(bra(O)), rot180.(pepo(O)))

## Math (for Zygote accumulation)

_add_localsandwich(O1::PEPSSandwich, O2::PEPSSandwich) = O1 .+ O2
_subtract_localsandwich(O1::PEPSSandwich, O2::PEPSSandwich) = O1 .- O2
_mul_localsandwich(α::Number, O::PEPSSandwich) = α .* O

## Chainrules

function ChainRulesCore.rrule(::typeof(ket), O::PEPSSandwich)
    k = ket(O)
    ket_pullback(Δk) = NoTangent(), (unthunk(Δk), zerovector(bra(O)))
    return k, ket_pullback
end
function ChainRulesCore.rrule(::typeof(bra), O::PEPSSandwich)
    b = bra(O)
    bra_pullback(Δb) = NoTangent(), (zerovector(ket(O)), unthunk(Δb))
    return b, bra_pullback
end

function ChainRulesCore.rrule(::typeof(ket), O::PEPOSandwich)
    k = ket(O)
    ket_pullback(Δk) = NoTangent(), (unthunk(Δk), zerovector(bra(O)), zerovector.(pepo(O)))
    return k, ket_pullback
end
function ChainRulesCore.rrule(::typeof(bra), O::PEPOSandwich)
    b = bra(O)
    bra_pullback(Δb) = NoTangent(), (zerovector(ket(O)), unthunk(Δb), zerovector.(pepo(O)))
    return b, bra_pullback
end
function ChainRulesCore.rrule(::typeof(pepo), O::PEPOSandwich)
    p = pepo(O)
    pepo_pullback(Δp) = NoTangent(), (zerovector(ket(O)), zerovector(bra(O)), unthunk(Δp))
    return p, pepo_pullback
end

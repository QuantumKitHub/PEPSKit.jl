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

## Rotations

# generic local interface
_rotl90_localsandwich(O) = rotl90.(O)
_rotr90_localsandwich(O) = rotr90.(O)
_rot180_localsandwich(O) = rot180.(O)

## Math (for Zygote accumulation)

# generic local interface
_add_localsandwich(O1, O2) = O1 .+ O2
_subtract_localsandwich(O1, O2) = O1 .- O2
_mul_localsandwich(α::Number, O) = α .* O

## PartitionFunction

# specialized local rotation interface
_rotl90_localsandwich(O::PFTensor) = rotl90(O)
_rotr90_localsandwich(O::PFTensor) = rotr90(O)
_rot180_localsandwich(O::PFTensor) = rot180(O)

# specialized local math interface
_add_localsandwich(O1::PFTensor, O2::PFTensor) = O1 + O2
_subtract_localsandwich(O1::PFTensor, O2::PFTensor) = O1 - O2
_mul_localsandwich(α::Number, O::PFTensor) = α * O

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

## Only PEPO layers
# In a CTMRG contraction, the top physical leg of the top PEPOTensor is contracted with the bottom physical leg of the bottom PEPOTensor

const PEPOLayersSandwich{N,P<:PEPOTensor} = Tuple{Vararg{P,N}}

pepo(O::PEPOLayersSandwich) = O[1:end]
pepo(O::PEPOLayersSandwich, i::Int) = O[i]

function virtualspace(O::PEPOLayersSandwich, dir)
    return prod([virtualspace.(pepo(O), Ref(dir))...])
end

#
# Expressions for mpotensor
#

function _pepo_fuser_tensor_expr(tensorname, H::Int, dir, args...;)
    return tensorexpr(
        tensorname,
        (virtuallabel(dir, :fuser, args...),),
        (
            virtuallabel(dir, :top),
            ntuple(h -> virtuallabel(_virtual_labels(dir, :mid, h, args...)...), H)...,
            virtuallabel(dir, :bot),
        ),
    )
end

function _pepolayers_fuser_tensor_expr(tensorname, H::Int, dir, args...;)
    return tensorexpr(
        tensorname,
        (virtuallabel(dir, :fuser, args...),),
        (ntuple(h -> virtuallabel(_virtual_labels(dir, :mid, h, args...)...), H)...,),
    )
end

@generated function _mpotensor_contraction(
    F_north::AbstractTensorMap{T,S},
    F_east::AbstractTensorMap{T,S},
    F_south::AbstractTensorMap{T,S},
    F_west::AbstractTensorMap{T,S},
    O::PEPOSandwich{H},
) where {T,S,H}
    fuser_north = _pepo_fuser_tensor_expr(:F_north, H, :N)
    fuser_east = _pepo_fuser_tensor_expr(:F_east, H, :E)
    fuser_south = _pepo_fuser_tensor_expr(:F_south, H, :S)
    fuser_west = _pepo_fuser_tensor_expr(:F_west, H, :W)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    result = tensorexpr(:res, (:D_W_fuser, :D_S_fuser), (:D_N_fuser, :D_E_fuser))

    rhs = Expr(
        :call,
        :*,
        Expr(:call, :conj, fuser_north),
        Expr(:call, :conj, fuser_east),
        fuser_south,
        fuser_west,
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )
    return macroexpand(@__MODULE__, :(return @autoopt @tensor $result := $rhs))
end

@generated function _mpotensor_contraction(
    F_north::AbstractTensorMap{T,S},
    F_east::AbstractTensorMap{T,S},
    F_south::AbstractTensorMap{T,S},
    F_west::AbstractTensorMap{T,S},
    O::PEPOLayersSandwich{H},
) where {T,S,H}
    fuser_north = _pepolayers_fuser_tensor_expr(:F_north, H, :N)
    fuser_east = _pepolayers_fuser_tensor_expr(:F_east, H, :E)
    fuser_south = _pepolayers_fuser_tensor_expr(:F_south, H, :S)
    fuser_west = _pepolayers_fuser_tensor_expr(:F_west, H, :W)
    pepo_es = _pepolayers_sandwich_expr(:O, H)

    result = tensorexpr(:res, (:D_W_fuser, :D_S_fuser), (:D_N_fuser, :D_E_fuser))

    rhs = Expr(
        :call,
        :*,
        Expr(:call, :conj, fuser_north),
        Expr(:call, :conj, fuser_east),
        fuser_south,
        fuser_west,
        pepo_es...,
    )
    return macroexpand(@__MODULE__, :(return @autoopt @tensor $result := $rhs))
end

# not overloading MPOTensor because that defines AbstractTensorMap{<:Any,S,2,2}(::PEPSTensor, ::PEPSTensor)
# ie type piracy
mpotensor(top::PEPSTensor) = mpotensor((top, top))
function mpotensor(network::PEPOSandwich{H}) where {H}
    @assert virtualspace(ket(network), NORTH) == dual(virtualspace(ket(network), SOUTH)) &&
        virtualspace(bra(network), NORTH) == dual(virtualspace(bra(network), SOUTH)) &&
        virtualspace(ket(network), EAST) == dual(virtualspace(ket(network), WEST)) &&
        virtualspace(bra(network), EAST) == dual(virtualspace(bra(network), WEST)) &&
        isdual(virtualspace(ket(network), NORTH)) &&
        isdual(virtualspace(bra(network), NORTH)) &&
        isdual(virtualspace(ket(network), EAST)) &&
        isdual(virtualspace(bra(network), EAST)) "Method not yet implemented for given virtual spaces"
    for h in 1:H
        @assert virtualspace(network[h], NORTH) == dual(virtualspace(network[h], SOUTH)) &&
            virtualspace(network[h], EAST) == dual(virtualspace(network[h], WEST)) &&
            isdual(virtualspace(network[h], NORTH)) &&
            isdual(virtualspace(network[h], EAST)) "Method not yet implemented for given virtual spaces"
    end
    F_west = isomorphism(
        storagetype(network[1]),
        fuse(virtualspace(network, WEST)),
        virtualspace(network, WEST),
    )
    F_south = isomorphism(
        storagetype(network[1]),
        fuse(virtualspace(network, SOUTH)),
        virtualspace(network, SOUTH),
    )
    return _mpotensor_contraction(
        F_south, F_west, twist(F_south, H + 3), twist(F_west, H + 3), network
    )
end

function mpotensor(network::PEPOLayersSandwich{H}) where {H}
    for h in 1:H
        @assert virtualspace(network[h], NORTH) == dual(virtualspace(network[h], SOUTH)) &&
            virtualspace(network[h], EAST) == dual(virtualspace(network[h], WEST)) &&
            isdual(virtualspace(network[h], NORTH)) &&
            isdual(virtualspace(network[h], EAST)) "Method not yet implemented for given virtual spaces"
    end
    F_west = isomorphism(
        storagetype(network[1]),
        fuse(virtualspace(network, WEST)),
        virtualspace(network, WEST),
    )
    F_south = isomorphism(
        storagetype(network[1]),
        fuse(virtualspace(network, SOUTH)),
        virtualspace(network, SOUTH),
    )
    return _mpotensor_contraction(F_south, F_west, F_south, F_west, network)
end

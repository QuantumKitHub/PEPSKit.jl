import MPSKit: Multiline, MultilineEnvironments

# implementation of PEPS transfer matrices in terms of MPSKit.InfiniteMPO

#
# Effective MPOTensor interface
#


## Space utils

_elementwise_dual(S::ElementarySpace) = S
_elementwise_dual(P::ProductSpace) = prod(dual.(P))

north_space(O, args...) = virtual_space(O, NORTH, args...)
east_space(O, args...) = virtual_space(O, EAST, args...)
south_space(O, args...) = virtual_space(O, SOUTH, args...)
west_space(O, args...) = virtual_space(O, WEST, args...)


## PEPS

const PEPSSandwich{T<:PEPSTensor} = Tuple{T,T}

ket(p::PEPSSandwich) = p[1]
bra(p::PEPSSandwich) = p[2]

function virtual_space(p::PEPSSandwich, dir)
    return space(ket(p), dir + 1) ⊗ space(bra(p), dir + 1)'
end


## PEPO

const PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,T,Tuple{Vararg{P,N}}}

ket(p::PEPOSandwich) = p[1]
bra(p::PEPOSandwich) = p[2]
pepo(p::PEPOSandwich) = p[3]
pepo(p::PEPOSandwich, i::Int) = p[3][i]

function virtual_space(p::PEPOSandwich, i, dir)
    return prod([
        space(ket(p), dir + 1),
        space.(pepo(p), Ref(dir + 2))...,
        space(bra(parentindices), dir + 1)',
    ])
end


## Common

MPSKit.left_virtualspace(p::Union{PEPSSandwich,PEPOSandwich}) = west_space(p)
function MPSKit.right_virtualspace(p::Union{PEPSSandwich,PEPOSandwich})
    return _elementwise_dual(east_space(p))
end # follow MPSKit convention: right vspace gets a dual by default


#
# PEPS
#

"""
    InfiniteTransferPEPS{T}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the overlap between 'ket' and 'bra' `InfinitePEPS` states.
"""
const InfiniteTransferPEPS{T<:PEPSTensor} = InfiniteMPO{PEPSSandwich{T}}

function InfiniteTransferPEPS(
    top::PeriodicArray{T,1}, bot::PeriodicArray{T,1}
) where {T<:PEPSTensor}
    return InfiniteMPO(map(Tuple, zip(top, bot)))
end

InfiniteTransferPEPS(top) = InfiniteTransferPEPS(top, top)

"""
    InfiniteTransferPEPS(T::InfinitePEPS, dir, row)

Constructs a transfer operator corresponding to a single row of a partition function
representing the norm of the state `T`. The partition function is first rotated such that
the direction `dir` faces north, after which its `row`th row from the north is selected.
"""
function InfiniteTransferPEPS(T::InfinitePEPS, dir, row)
    T = rotate_north(T, dir)
    return InfiniteTransferPEPS(PeriodicArray(T[row, :]))
end

"""
    const MultilineTransferPEPS = MPSKit.Multiline{<:InfiniteTransferPEPS}

Type that represents a multi-line transfer operator, where each line each corresponds to a
row of a partition function encoding the overlap between 'ket' and 'bra' `InfinitePEPS`
states.
"""
const MultilineTransferPEPS = MPSKit.Multiline{<:InfiniteTransferPEPS} # TODO: do we actually need this?
Base.convert(::Type{MultilineTransferPEPS}, O::InfiniteTransferPEPS) = MPSKit.Multiline([O])

"""
    MultilineTransferPEPS(T::InfinitePEPS, dir)

Construct a multi-row transfer operator corresponding to the partition function representing
the norm of the state `T`. The partition function is first rotated such
that the direction `dir` faces north.
"""
function MultilineTransferPEPS(T::InfinitePEPS, dir)
    rowsize = size(T, mod1(dir, 2))  # depends on dir
    return MPSKit.Multiline(map(cr -> InfiniteTransferPEPS(T, dir, cr), 1:rowsize))
end


#
# PEPO
#

"""
    InfiniteTransferPEPO{T,O}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the expectation value of an `InfinitePEPO` between 'ket' and
'bra' `InfinitePEPS` states.
"""
const InfiniteTransferPEPO{T<:PEPSTensor,O<:PEPOTensor} = InfiniteMPO{PEPOSandwich{<:Any,T,O}}

function InfiniteTransferPEPO(
    top::PeriodicArray{T,1}, mid::PeriodicArray{O,2}, bot::PeriodicArray{T,1}
) where {T,O}
    size(top, 1) == size(bot, 1) == size(mid, 1) ||
        throw(ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have length"))
    return InfiniteMPO(map(Tuple, zip(top, bot, Iterators.map(Tuple, eachcol(mid)))))
end

InfiniteTransferPEPO(top, mid) = InfiniteTransferPEPO(top, mid, top)

"""
    InfiniteTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir, row)

Constructs a transfer operator corresponding to a single row of a partition function
representing the expectation value of `O` for the state `T`. The partition function is first
rotated such that the direction `dir` faces north, after which its `row`th row from the
north is selected.
"""
function InfiniteTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir, row)
    T = rotate_north(T, dir)
    O = rotate_north(O, dir)
    return InfiniteTransferPEPO(PeriodicArray(T[row, :]), PeriodicArray(O[row, :, :]))
end

"""
    const MultilineTransferPEPO = MPSKit.Multiline{<:InfiniteTransferPEPO}

Type that represents a multi-line transfer operator, where each line each corresponds to a
row of a partition function encoding the overlap of an `InfinitePEPO` between 'ket' and
'bra' `InfinitePEPS` states.
"""
const MultilineTransferPEPO = MPSKit.Multiline{<:InfiniteTransferPEPO}
Base.convert(::Type{MultilineTransferPEPO}, O::InfiniteTransferPEPO) = MPSKit.Multiline([O])

"""
    MultilineTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir)

Construct a multi-row transfer operator corresponding to the partition function representing
the expectation value of `O` for the state `T`. The partition function is first rotated such
that the direction `dir` faces north.
"""
function MultilineTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir)
    rowsize = size(T, mod1(dir, 2))  # depends on dir
    return MPSKit.Multiline(map(cr -> InfiniteTransferPEPO(T, O, dir, cr), 1:rowsize))
end

#
# Common interface
#

const InfiniteTransferMatrix = Union{InfiniteTransferPEPS,InfiniteTransferPEPO}
const MultilineTransferMatrix = Union{MultilineTransferPEPS,MultilineTransferPEPO}

virtual_space(O::InfiniteTransferMatrix, i, dir) = virtual_space(O[i], dir)

"""
    initializeMPS(
        O::Union{InfiniteTransferPEPS,InfiniteTransferPEPO},
        virtualspaces::AbstractArray{<:ElementarySpace,1}
    )
    initializeMPS(
        O::Union{MultilineTransferPEPS,MultilineTransferPEPO},
        virtualspaces::AbstractArray{<:ElementarySpace,2}
    )

Inialize a boundary MPS for the transfer operator `O` by specifying an array of virtual
spaces consistent with the unit cell.
"""
function initializeMPS(
    O::InfiniteTransferMatrix, virtualspaces::AbstractArray{S,1}
) where {S}
    return InfiniteMPS([
        randn(
            scalartype(O),
            virtualspaces[_prev(i, end)] * _elementwise_dual(north_space(O, i)),
            virtualspaces[mod1(i, end)],
        ) for i in 1:length(O)
    ])
end
function initializeMPS(O::InfiniteTransferMatrix, χ::Int)
    return InfiniteMPS([
        randn(calartype(O), ℂ^χ * _elementwise_dual(north_space(O, i)), ℂ^χ) for
        i in 1:length(O)
    ])
end
function initializeMPS(
    O::MultilineTransferMatrix, virtualspaces::AbstractArray{S,2}
) where {S}
    mpss = map(1:size(O, 1)) do r
        return initializeMPS(O[r], virtualspaces[r, :])
    end
    return MPSKit.Multiline(mpss)
end
function initializeMPS(
    O::MultilineTransferMatrix, virtualspaces::AbstractArray{S,1}
) where {S}
    return initializeMPS(O, repeat(virtualspaces, length(O), 1))
end
function initializeMPS(O::MultilineTransferMatrix, V::ElementarySpace)
    return initializeMPS(O, repeat([V], length(O), length(O[1])))
end
function initializeMPS(O::MultilineTransferMatrix, χ::Int)
    return initializeMPS(O, repeat([ℂ^χ], length(O), length(O[1])))
end

@doc """
    MPSKit.expectation_value(st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO})
    MPSKit.expectation_value(st::MultilineMPS, op::Union{MultilineTransferPEPS,MultilineTransferPEPO})

Compute expectation value of the transfer operator `op` for the state `st` for each site in
the unit cell.
""" MPSKit.expectation_value(st, op)

@doc """
    MPSKit.leading_boundary(
        st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO}, alg, [envs]
    )
    MPSKit.leading_boundary(
        st::MPSMulitline, op::Union{MultilineTransferPEPS,MultilineTransferPEPO}, alg, [envs]
    )

Approximate the leading boundary MPS eigenvector for the transfer operator `op` using `st`
as initial guess.
""" MPSKit.leading_boundary(st, op, alg)

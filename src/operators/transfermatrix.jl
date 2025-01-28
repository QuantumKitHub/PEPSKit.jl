import MPSKit: Multiline, MultilineEnvironments

# TODO: rewrite in terms of InfiniteMPO{O} where O is an 'effective MPO type' once type
# restrictions in MPSKit are relaxed

#
# PEPS
#

"""
    InfiniteTransferPEPS{T}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the overlap between 'ket' and 'bra' `InfinitePEPS` states.
"""
struct InfiniteTransferPEPS{T}
    top::PeriodicArray{T,1}
    bot::PeriodicArray{T,1}
    function InfiniteTransferPEPS(
        top::PeriodicArray{T,1}, bot::PeriodicArray{T,1}
    ) where {T}
        size(top) == size(bot) ||
            throw(ArgumentError("Top and bottom PEPS rows should have equal sizes."))
        return new{T}(top, bot)
    end
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

Base.size(transfer::InfiniteTransferPEPS) = size(transfer.top)
Base.size(transfer::InfiniteTransferPEPS, args...) = size(transfer.top, args...)
Base.length(transfer::InfiniteTransferPEPS) = size(transfer, 1)
Base.getindex(O::InfiniteTransferPEPS, i) = (O.top[i], O.bot[i])

Base.iterate(O::InfiniteTransferPEPS, i=1) = i > length(O) ? nothing : (O[i], i + 1)

VectorInterface.scalartype(::Type{InfiniteTransferPEPS{T}}) where {T} = scalartype(T)

function virtual_space(O::InfiniteTransferPEPS, i, dir)
    return prod([space(O.top[i], dir + 1), space(O.bot[i], dir + 1)'])
end

"""
    const MultilineTransferPEPS = MPSKit.Multiline{<:InfiniteTransferPEPS}

Type that represents a multi-line transfer operator, where each line each corresponds to a
row of a partition function encoding the overlap between 'ket' and 'bra' `InfinitePEPS`
states.
"""
const MultilineTransferPEPS = MPSKit.Multiline{<:InfiniteTransferPEPS}
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
struct InfiniteTransferPEPO{T,O}
    top::PeriodicArray{T,1}
    mid::PeriodicArray{O,2}
    bot::PeriodicArray{T,1}
    function InfiniteTransferPEPO(
        top::PeriodicArray{T,1}, mid::PeriodicArray{O,2}, bot::PeriodicArray{T,1}
    ) where {T,O}
        size(top, 1) == size(bot, 1) == size(mid, 1) ||
            throw(ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have length"))
        return new{T,O}(top, mid, bot)
    end
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

Base.size(transfer::InfiniteTransferPEPO) = size(transfer.top)
Base.size(transfer::InfiniteTransferPEPO, args...) = size(transfer.top, args...)
Base.length(transfer::InfiniteTransferPEPO) = size(transfer, 1)
height(transfer::InfiniteTransferPEPO) = size(transfer.mid, 2)
Base.getindex(O::InfiniteTransferPEPO, i) = (O.top[i], O.bot[i], Tuple(O.mid[i, :]))

Base.iterate(O::InfiniteTransferPEPO, i=1) = i > length(O) ? nothing : (O[i], i + 1)

VectorInterface.scalartype(::Type{InfiniteTransferPEPO{T,O}}) where {T,O} = scalartype(T)

function virtual_space(O::InfiniteTransferPEPO, i, dir)
    return prod([
        space(O.top[i], dir + 1),
        space.(O.mid[i, :], Ref(dir + 2))...,
        space(O.bot[i], dir + 1)',
    ])
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

_elementwise_dual(S::ElementarySpace) = S
_elementwise_dual(P::ProductSpace) = prod(dual.(P))

north_space(O::InfiniteTransferMatrix, i) = virtual_space(O, i, NORTH)
east_space(O::InfiniteTransferMatrix, i) = virtual_space(O, i, EAST)
south_space(O::InfiniteTransferMatrix, i) = virtual_space(O, i, SOUTH)
west_space(O::InfiniteTransferMatrix, i) = virtual_space(O, i, WEST)

MPSKit.left_virtualspace(O::InfiniteTransferMatrix, i) = west_space(O, i)
function MPSKit.right_virtualspace(O::InfiniteTransferMatrix, i)
    return _elementwise_dual(east_space(O, i))
end # follow MPSKit convention: right vspace gets a dual by default

function Base.getindex(t::MultilineTransferMatrix, ::Colon, j::Int)
    return Base.getindex.(parent(t), j)
end
Base.getindex(t::MultilineTransferMatrix, i::Int, j) = Base.getindex(t[i], j)

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

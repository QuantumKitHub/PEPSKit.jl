# Get next and previous directional CTMRG environment index, respecting periodicity
_next(i, total) = mod1(i + 1, total)
_prev(i, total) = mod1(i - 1, total)

"""
Shift the first of `inds` into the unit cell
(with size `unitcell`) by a lattice translation.
"""
function _shift_into_unitcell!(
        inds::Vector{CartesianIndex{N}}, unitcell::NTuple{N, Int}
    ) where {N}
    I1 = first(inds)
    I1_mod = CartesianIndex(mod1.(Tuple(I1), unitcell))
    inds .-= (I1 - I1_mod)
    return inds
end

@inline _periodic_inds(data, J::NTuple{N, Int}) where {N} =
    ntuple(i -> mod1(J[i], size(data, i)), Val(N))

"""
    periodic_getindex(A, data, I::Tuple)

Periodic indexing of `A[I...]` backed by `data`.
For `I::Vararg{Int}`, this yields `data[mod1.(I, size(data))...]`.
Mixed index arguments (`Int` / `CartesianIndex`) are supported by flattening with `Base.to_indices`.

!!! warning
    Linear indexing schemes are not supported.
"""
@inline function periodic_getindex(A, data, I::Tuple)
    J = to_indices(data, I)
    return _periodic_getindex_dispatch(A, data, J)
end

# Dispatch helper
_periodic_getindex_dispatch(A, data, J::Tuple{Vararg{Int}}) = @inbounds data[_periodic_inds(data, J)...]
_periodic_getindex_dispatch(A, data, J::Tuple) = throw(ArgumentError("Invalid periodic indexing type $(typeof(J))"))

"""
    periodic_setindex!(A, data, v, I::Tuple)

Periodic indexing assignment `A[I...] = v` backed by `data`, returning `v` so that the
syntax `A[I...] = v` evaluates to `v` in line with Base Julia conventions.
For `I::Vararg{Int}`, this yields `data[mod1.(I, size(data))...] = v`.
Mixed index arguments (`Int` / `CartesianIndex`) are supported by flattening with `Base.to_indices`.

!!! warning
    Linear indexing schemes are not supported.
"""
@inline function periodic_setindex!(A, data, v, I::Tuple)
    J = to_indices(data, I)
    return _periodic_setindex_dispatch!(A, data, v, J)
end

# Dispatch helper
function _periodic_setindex_dispatch!(A, data, v, J::Tuple{Vararg{Int}})
    @inbounds data[_periodic_inds(data, J)...] = v
    return v
end
_periodic_setindex_dispatch!(_A, data, v, J::Tuple) =
    throw(ArgumentError("Invalid periodic indexing type $(typeof(J))"))

# Get next and previous coordinate (direction, row, column), given a direction and going around the environment clockwise
function _next_coordinate((dir, row, col), rowsize, colsize)
    if dir == 1
        return (_next(dir, 4), row, _next(col, colsize))
    elseif dir == 2
        return (_next(dir, 4), _next(row, rowsize), col)
    elseif dir == 3
        return (_next(dir, 4), row, _prev(col, colsize))
    elseif dir == 4
        return (_next(dir, 4), _prev(row, rowsize), col)
    end
end
function _prev_coordinate((dir, row, col), rowsize, colsize)
    if dir == 1
        return (_prev(dir, 4), _next(row, rowsize), col)
    elseif dir == 2
        return (_prev(dir, 4), row, _prev(col, colsize))
    elseif dir == 3
        return (_prev(dir, 4), _prev(row, rowsize), col)
    elseif dir == 4
        return (_prev(dir, 4), row, _next(col, colsize))
    end
end

# iterator over each coordinates
"""
    eachcoordinate(x, [dirs=1:4])

Enumerate all (dir, row, col) pairs.
"""
function eachcoordinate end

@non_differentiable eachcoordinate(args...)

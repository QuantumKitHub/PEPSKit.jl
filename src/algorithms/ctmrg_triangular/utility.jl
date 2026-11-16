# lattice offsets `(δr, δc)` of the six neighbors in the directions NW, NE, E, SE, SW, W
const _TRIANGULAR_OFFSETS = ((-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1))

"""
    _coordinates(dir, rot, r, c, cell) -> (dir′, r′, c′)

Return `mod1(dir, 6)` together with the unit-cell representative of the neighbor of site
`[r, c]` in direction `dir + rot`, for a [`TriangularUnitCell`](@ref) `cell` (or a
rectangular unit-cell size).
"""
function _coordinates(dir::Int, rot::Int, r::Int, c::Int, cell::TriangularUnitCell)
    δr, δc = _TRIANGULAR_OFFSETS[mod1(dir + rot, 6)]
    return (mod1(dir, 6), _site(cell, r + δr, c + δc)...)
end
function _coordinates(dir::Int, rot::Int, r::Int, c::Int, unitcell::Tuple{Int, Int})
    return _coordinates(dir, rot, r, c, TriangularUnitCell(unitcell))
end
function _coordinates(dir::Int, r::Int, c::Int, cell)
    return _coordinates(dir, 0, r, c, cell)[2:3]
end
@non_differentiable _coordinates(args...)

function _truncway(trunctype::Symbol, D::E) where {E <: ElementarySpace}
    if trunctype == :truncdim
        return truncdim(dim(D))
    else
        return notrunc()
    end
end

function rotr60(A::T, dir::Int) where {E, S, T <: AbstractTensorMap{E, S, 1, 6}}
    if dir == 0
        return A
    else
        return rotr60(permute(A, ((1,), (7, 2, 3, 4, 5, 6))), dir - 1)
    end
end

function rotr60(A::T, dir::Int) where {E, S, T <: AbstractTensorMap{E, S, 3, 3}}
    if dir == 0
        return A
    else
        return rotr60(permute(A, ((2, 3, 6), (1, 4, 5))), dir - 1)
    end
end

function rotl60(A::T, dir::Int) where {E, S, T <: AbstractTensorMap{E, S, 1, 6}}
    if dir == 0
        return A
    else
        return rotl60(permute(A, ((1,), (3, 4, 5, 6, 7, 2))), dir - 1)
    end
end

function rotl60(A::T, dir::Int) where {E, S, T <: AbstractTensorMap{E, S, 3, 3}}
    if dir == 0
        return A
    else
        return rotl60(permute(A, ((4, 1, 2), (5, 6, 3))), dir - 1)
    end
end

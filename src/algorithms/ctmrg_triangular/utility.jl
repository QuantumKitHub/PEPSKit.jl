function _coordinates(dir::Int, rot::Int, r::Int, c::Int, unitcell::Tuple{Int, Int})
    rows, cols = unitcell
    if mod1(dir + rot, 6) == 1
        return (mod1(dir, 6), mod1(r - 1, rows), mod1(c, cols))
    elseif mod1(dir + rot, 6) == 2
        return (mod1(dir, 6), mod1(r - 1, rows), mod1(c + 1, cols))
    elseif mod1(dir + rot, 6) == 3
        return (mod1(dir, 6), mod1(r, rows), mod1(c + 1, cols))
    elseif mod1(dir + rot, 6) == 4
        return (mod1(dir, 6), mod1(r + 1, rows), mod1(c, cols))
    elseif mod1(dir + rot, 6) == 5
        return (mod1(dir, 6), mod1(r + 1, rows), mod1(c - 1, cols))
    elseif mod1(dir + rot, 6) == 6
        return (mod1(dir, 6), mod1(r, rows), mod1(c - 1, cols))
    end
end

function _coordinates(dir::Int, r::Int, c::Int, unitcell::Tuple{Int, Int})
    return _coordinates(dir, 0, r, c; unitcell)[2:3]
end

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

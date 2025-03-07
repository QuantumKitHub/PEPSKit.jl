"""
    struct InfinitePartitionFunction{T<:PartitionFunctionTensor}

Represents an infinite partition function on a 2D square lattice.
"""
struct InfinitePartitionFunction{T<:PartitionFunctionTensor}
    A::Matrix{T}
    function InfinitePartitionFunction{T}(A::Matrix{T}) where {T<:PartitionFunctionTensor}
        return new{T}(A)
    end
    function InfinitePartitionFunction(A::Matrix{T}) where {T<:PartitionFunctionTensor}
        for (d, w) in Tuple.(CartesianIndices(A))
            north_virtualspace(A[d, w]) == south_virtualspace(A[_prev(d, end), w])' ||
                throw(
                    SpaceMismatch("North virtual space at site $((d, w)) does not match.")
                )
            east_virtualspace(A[d, w]) == west_virtualspace(A[d, _next(w, end)])' ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
            dim(space(A[d, w])) > 0 || @warn "no fusion channels at site ($d, $w)"
        end
        return new{T}(A)
    end
end

const InfinitePF{T} = InfinitePartitionFunction{T}

## Constructors

"""
    InfinitePartitionFunction(A::AbstractMatrix{T})

Create an `InfinitePartitionFunction` by specifying a matrix containing the PEPS tensors at each site in
the unit cell.
"""
function InfinitePartitionFunction(A::AbstractMatrix{T}) where {T<:PartitionFunctionTensor}
    return InfinitePartitionFunction(Array(deepcopy(A))) # TODO: find better way to copy
end

"""
    InfinitePartitionFunction(
        f=randn, T=ComplexF64, Pspaces::A, Nspaces::A, [Espaces::A]
    ) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Create an `InfinitePartitionFunction` by specifying the physical, north virtual and east virtual spaces
of the PEPS tensor at each site in the unit cell as a matrix. Each individual space can be
specified as either an `Int` or an `ElementarySpace`.
"""
function InfinitePartitionFunction(
    Nspaces::A, Espaces::A
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    return InfinitePartitionFunction(randn, ComplexF64, Nspaces, Espaces)
end
function InfinitePartitionFunction(
    f, T, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:ElementarySpaceLike}}
    size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (-1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, 1)))

    A = map(Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return PartitionFunctionTensor(f, T, N, E, S, W)
    end

    return InfinitePartitionFunction(A)
end

"""
    InfinitePartitionFunction(A; unitcell=(1, 1))

Create an `InfinitePartitionFunction` by specifying a tensor and unit cell.

The unit cell is labeled as a matrix which means that any tensor in the unit cell,
regardless if partition function tensor or environment tensor, is obtained by shifting the row
and column index `[r, c]` by one, respectively:
```
   |            |          |
---C[r-1,c-1]---T[r-1,c]---T[r-1,c+1]---
   |            |          |
---T[r,c-1]-----AA[r,c]----AA[r,c+1]----
   |            |          |
---T[r+1,c-1]---AA[r+1,c]--AA[r+1,c+1]--
   |            |          |
```
The unit cell has periodic boundary conditions, so `[r, c]` is indexed modulo the
size of the unit cell.
"""
function InfinitePartitionFunction(
    A::T; unitcell::Tuple{Int,Int}=(1, 1)
) where {T<:PartitionFunctionTensor}
    return InfinitePartitionFunction(fill(A, unitcell))
end

"""
    InfinitePartitionFunction(f=randn, T=ComplexF64, Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePartitionFunction by specifying its physical, north and east spaces and unit cell.
Spaces can be specified either via `Int` or via `ElementarySpace`.
"""
function InfinitePartitionFunction(
    Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpaceLike}
    return InfinitePartitionFunction(
        randn, ComplexF64, fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end
function InfinitePartitionFunction(
    f, T, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpaceLike}
    return InfinitePartitionFunction(f, T, fill(Nspace, unitcell), fill(Espace, unitcell))
end

## Unit cell interface

unitcell(t::InfinitePartitionFunction) = t.A
Base.size(A::InfinitePartitionFunction, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePartitionFunction) = length(unitcell(A))
Base.eltype(::Type{InfinitePartitionFunction{T}}) where {T} = T
Base.eltype(A::InfinitePartitionFunction) = eltype(typeof(A))

Base.copy(A::InfinitePartitionFunction) = InfinitePartitionFunction(copy(unitcell(A)))
function Base.similar(A::InfinitePartitionFunction, args...)
    return InfinitePartitionFunction(map(t -> similar(t, args...), unitcell(A)))
end
function Base.repeat(A::InfinitePartitionFunction, counts...)
    return InfinitePartitionFunction(repeat(unitcell(A), counts...))
end

Base.getindex(A::InfinitePartitionFunction, args...) = Base.getindex(unitcell(A), args...)
function Base.setindex!(A::InfinitePartitionFunction, args...)
    return (Base.setindex!(unitcell(A), args...); A)
end
Base.axes(A::InfinitePartitionFunction, args...) = axes(unitcell(A), args...)
eachcoordinate(A::InfinitePartitionFunction) = collect(Iterators.product(axes(A)...))
function eachcoordinate(A::InfinitePartitionFunction, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces

virtualspace(n::InfinitePartitionFunction, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(state::InfinitePartitionFunction)
    return InfiniteSquareNetwork(unitcell(state))
end

## (Approximate) equality
function Base.:(==)(A₁::InfinitePartitionFunction, A₂::InfinitePartitionFunction)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(
    A₁::InfinitePartitionFunction, A₂::InfinitePartitionFunction; kwargs...
)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations

function Base.rotl90(A::InfinitePartitionFunction)
    return InfinitePartitionFunction(rotl90(rotl90.(unitcell(A))))
end
function Base.rotr90(A::InfinitePartitionFunction)
    return InfinitePartitionFunction(rotr90(rotr90.(unitcell(A))))
end
function Base.rot180(A::InfinitePartitionFunction)
    return InfinitePartitionFunction(rot180(rot180.(unitcell(A))))
end

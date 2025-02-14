"""
    struct InfinitePartitionFunction{T<:PartitionFunctionTensor}

Represents an infinite partition function on a 2D square lattice.
"""
struct InfinitePartitionFunction{T<:PartitionFunctionTensor} <: InfiniteGridNetwork{T,2}
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

## InfiniteGridNetwork interface

unitcell(t::InfinitePartitionFunction) = t.A

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

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(state::InfinitePartitionFunction)
    return InfiniteSquareNetwork(unitcell(state))
end

function ChainRulesCore.rrule(
    ::Type{InfiniteSquareNetwork}, state::InfinitePartitionFunction, bot::InfinitePEPS
)
    network = InfiniteSquareNetwork(state)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δstate = InfinitePartitionFunction(unitcell(Δnetwork))
        return NoTangent(), Δstate
    end
    return network, InfiniteSquareNetwork_pullback
end

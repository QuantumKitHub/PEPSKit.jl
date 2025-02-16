"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePEPS{T<:PEPSTensor} <: InfiniteGridNetwork{T,2}
    A::Matrix{T}
    InfinitePEPS{T}(A::Matrix{T}) where {T<:PEPSTensor} = new{T}(A)
    function InfinitePEPS(A::Array{T,2}) where {T<:PEPSTensor}
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

## InfiniteGridNetwork interface

unitcell(t::InfinitePEPS) = t.A

## Constructors

const ElementarySpaceLike = Union{Int,ElementarySpace}

"""
    InfinitePEPS(A::AbstractMatrix{T})

Create an `InfinitePEPS` by specifying a matrix containing the PEPS tensors at each site in
the unit cell.
"""
function InfinitePEPS(A::AbstractMatrix{<:PEPSTensor})
    return InfinitePEPS(Array(deepcopy(A))) # TODO: find better way to copy
end

"""
    InfinitePEPS(
        f=randn, T=ComplexF64, Pspaces::A, Nspaces::A, [Espaces::A]
    ) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Create an `InfinitePEPS` by specifying the physical, north virtual and east virtual spaces
of the PEPS tensor at each site in the unit cell as a matrix. Each individual space can be
specified as either an `Int` or an `ElementarySpace`.
"""
function InfinitePEPS(
    Pspaces::A, Nspaces::A, Espaces::A
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:ElementarySpaceLike}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (-1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, 1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return PEPSTensor(f, T, P, N, E, S, W)
    end

    return InfinitePEPS(A)
end

"""
    InfinitePEPS(A; unitcell=(1, 1))

Create an `InfinitePEPS` by specifying a tensor and unit cell.

The unit cell is labeled as a matrix which means that any tensor in the unit cell,
regardless if PEPS tensor or environment tensor, is obtained by shifting the row
and column index `[r, c]` by one, respectively:
```
   |            |          |
---C[r-1,c-1]---T[r-1,c]---T[r-1,c+1]---
   |            ||         ||
---T[r,c-1]=====AA[r,c]====AA[r,c+1]====
   |            ||         ||
---T[r+1,c-1]===AA[r+1,c]==AA[r+1,c+1]==
   |            ||         ||
```
The unit cell has periodic boundary conditions, so `[r, c]` is indexed modulo the
size of the unit cell.
"""
function InfinitePEPS(A::T; unitcell::Tuple{Int,Int}=(1, 1)) where {T<:PEPSTensor}
    return InfinitePEPS(fill(A, unitcell))
end

"""
    InfinitePEPS(f=randn, T=ComplexF64, Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePEPS by specifying its physical, north and east spaces and unit cell.
Spaces can be specified either via `Int` or via `ElementarySpace`.
"""
function InfinitePEPS(
    Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpaceLike}
    return InfinitePEPS(
        randn,
        ComplexF64,
        fill(Pspace, unitcell),
        fill(Nspace, unitcell),
        fill(Espace, unitcell),
    )
end
function InfinitePEPS(
    f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpaceLike}
    return InfinitePEPS(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(top::InfinitePEPS, bot::InfinitePEPS=top)
    size(top) == size(bot) ||
        throw(ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have length"))
    return InfiniteSquareNetwork(map(Tuple, zip(unitcell(top), unitcell(bot))))
end

function ChainRulesCore.rrule(
    ::Type{InfiniteSquareNetwork}, top::InfinitePEPS, bot::InfinitePEPS
)
    network = InfiniteSquareNetwork(top, bot)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δtop = InfinitePEPS(map(ket, unitcell(Δnetwork)))
        Δbot = InfinitePEPS(map(bra, unitcell(Δnetwork)))
        return NoTangent(), Δtop, Δbot
    end
    return network, InfiniteSquareNetwork_pullback
end

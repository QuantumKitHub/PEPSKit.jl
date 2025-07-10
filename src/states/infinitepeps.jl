"""
    struct InfinitePEPS{T<:AbstractMatrix{<:PEPSTensor}}

Represents an infinite projected entangled-pair state on a 2D square lattice.

## Fields

$(TYPEDFIELDS)
"""
struct InfinitePEPS{T<:PEPSTensor}
    A::InfiniteTiledArray{T,2}
    function InfinitePEPS(A::InfiniteTiledArray{T,2}) where {T<:PEPSTensor}
        Iv = CartesianIndex(1, 0)
        Ih = CartesianIndex(0, 1)
        for I in CartesianIndices(A)
            north_virtualspace(A[I]) == south_virtualspace(A[I - Iv])' ||
                throw(SpaceMismatch("North virtual space at site $I does not match."))
            east_virtualspace(A[I]) == west_virtualspace(A[I + Ih])' ||
                throw(SpaceMismatch("East virtual space at site $I does not match."))
            dim(space(A[I])) > 0 || @warn "no fusion channels at site $I"
        end
        return new{T}(A)
    end
end

## Constructors

"""
    InfinitePEPS(A::AbstractMatrix{T})

Create an `InfinitePEPS` by specifying a matrix containing the PEPS tensors at each site in
the unit cell.
"""
function InfinitePEPS(A::AbstractMatrix{T}) where {T<:PEPSTensor}
    return InfinitePEPS(InfiniteTiledArray(A))
end

const ElementarySpaceLike = Union{Int,ElementarySpace}

"""
    InfinitePEPS([f=randn, T=ComplexF64,] Pspaces::A, Nspaces::A, [Espaces::A]) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Create an `InfinitePEPS` by specifying the physical, north virtual and east virtual spaces
of the PEPS tensor at each site in the unit cell as a matrix. Each individual space can be
specified as either an `Int` or an `ElementarySpace`.
"""
function InfinitePEPS(
    Pspaces::A, Nspaces::A, Espaces::A=Nspaces
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:InfiniteTiledArray{<:ElementarySpaceLike,2}}
    tiling(Pspaces) == tiling(Nspaces) == tiling(Espaces) ||
        throw(DimensionMismatch("Input spaces do not have matching tilings"))

    Sspaces = tiledmap(adjoint, circshift(Nspaces, (-1, 0)))
    Wspaces = tiledmap(adjoint, circshift(Espaces, (0, 1)))

    tensors = map(eachtilingindex(Pspaces)) do I
        P, N, E, S, W = getindex.((Pspaces, Nspaces, Espaces, Sspaces, Wspaces), Ref(I))
        return PEPSTensor(f, T, P, N, E, S, W)
    end
    A = InfiniteTiledArray(reshape(tensors, :), tiling(Pspaces))

    return InfinitePEPS(A)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:ElementarySpaceLike}}
    return InfinitePEPS(
        f,
        T,
        InfiniteTiledArray(Pspaces),
        InfiniteTiledArray(Nspaces),
        InfiniteTiledArray(Espaces),
    )
end

"""
    InfinitePEPS(A::PEPSTensor; unitcell=(1, 1))

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
# TODO: consider replacing `unitcell` with `tiling` and updating meaning.

"""
    InfinitePEPS([f=randn, T=ComplexF64,] Pspace, Nspace, [Espace]; unitcell=(1,1))

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

## Unit cell interface

unitcell(t::InfinitePEPS) = t.A
Base.size(A::InfinitePEPS, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePEPS) = length(unitcell(A))
Base.eltype(::Type{InfinitePEPS{T}}) where {T} = T
Base.eltype(A::InfinitePEPS) = eltype(typeof(A))

TiledArrays.tiling(A::InfinitePEPS) = tiling(unitcell(A))

Base.copy(A::InfinitePEPS) = InfinitePEPS(copy(unitcell(A)))
function Base.similar(A::InfinitePEPS, T::Type{TorA}=scalartype(A)) where {TorA}
    return InfinitePEPS(map(t -> similar(t, T), unitcell(A)))
end
Base.repeat(A::InfinitePEPS, counts...) = InfinitePEPS(repeat(unitcell(A), counts...))

Base.getindex(A::InfinitePEPS, args...) = Base.getindex(unitcell(A), args...)
Base.setindex!(A::InfinitePEPS, args...) = (Base.setindex!(unitcell(A), args...); A)
Base.axes(A::InfinitePEPS, args...) = axes(unitcell(A), args...)
eachcoordinate(A::InfinitePEPS) = collect(Iterators.product(axes(A)...))
function eachcoordinate(A::InfinitePEPS, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces

TensorKit.sectortype(t::InfinitePEPS) = sectortype(typeof(t))
TensorKit.sectortype(::Type{T}) where {T<:InfinitePEPS} = sectortype(eltype(T))
TensorKit.spacetype(t::InfinitePEPS) = spacetype(typeof(t))
TensorKit.spacetype(::Type{T}) where {T<:InfinitePEPS} = spacetype(eltype(T))
virtualspace(n::InfinitePEPS, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)
physicalspace(n::InfinitePEPS, r::Int, c::Int) = physicalspace(n[r, c])

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(top::InfinitePEPS, bot::InfinitePEPS=top)
    tiling(top) == tiling(bot) || throw(
        DimensionMismatch(
            "Top PEPS, bottom PEPS and PEPO rows should have the same tiling"
        ),
    )
    return InfiniteSquareNetwork(tiledmap(tuple, unitcell(top), unitcell(bot)))
end

## Vector interface

VI.scalartype(::Type{NT}) where {NT<:InfinitePEPS} = scalartype(eltype(NT))
VI.zerovector(A::InfinitePEPS) = InfinitePEPS(zerovector(unitcell(A)))

function VI.scale(ψ::InfinitePEPS, α::Number)
    _scale = Base.Fix2(scale, α)
    return InfinitePEPS(map(_scale, unitcell(ψ)))
end
function VI.scale!(ψ::InfinitePEPS, α::Number)
    _scale! = Base.Fix2(scale!, α)
    foreach(_scale!, unitcell(ψ))
    return ψ
end
function VI.scale!(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    foreach(_scale!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.scale!!(ψ::InfinitePEPS, α::Number) = scale!(ψ, α)
VI.scale!!(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS, α::Number) = scale!(ψ₁, ψ₂, α)

function VI.add(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return InfinitePEPS(map(_add, unitcell(ψ₁), unitcell(ψ₂)))
end
function VI.add!(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.add!!(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS, α::Number, β::Number) = add!(ψ₁, ψ₂, α, β)

## Math

function Base.:+(A₁::InfinitePEPS, A₂::InfinitePEPS)
    return InfinitePEPS(unitcell(A₁) + unitcell(A₂))
end
function Base.:-(A₁::InfinitePEPS, A₂::InfinitePEPS)
    return InfinitePEPS(unitcell(A₁) - unitcell(A₂))
end
Base.:*(α::Number, A::InfinitePEPS) = InfinitePEPS(α * unitcell(A))
Base.:*(A::InfinitePEPS, α::Number) = α * A
Base.:/(A::InfinitePEPS, α::Number) = InfinitePEPS(unitcell(A) / α)
LinearAlgebra.dot(A₁::InfinitePEPS, A₂::InfinitePEPS) = dot(unitcell(A₁), unitcell(A₂))
LinearAlgebra.norm(A::InfinitePEPS) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfinitePEPS, A₂::InfinitePEPS)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfinitePEPS, A₂::InfinitePEPS; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations

Base.rotl90(A::InfinitePEPS) = InfinitePEPS(rotl90(rotl90.(unitcell(A))))
Base.rotr90(A::InfinitePEPS) = InfinitePEPS(rotr90(rotr90.(unitcell(A))))
Base.rot180(A::InfinitePEPS) = InfinitePEPS(rot180(rot180.(unitcell(A))))

## OptimKit optimization backwards compatibility (v0.4 uses VectorInterface)

function LinearAlgebra.rmul!(A::InfinitePEPS, α::Number) # Used in _scale during OptimKit.optimize
    rmul!.(unitcell(A), α)
    return A
end
function LinearAlgebra.axpy!(α::Number, A₁::InfinitePEPS, A₂::InfinitePEPS) # Used in _add during OptimKit.optimize
    axpy!.(α, unitcell(A₁), unitcell(A₂))
    return A₂
end

## FiniteDifferences vectorization

"""
    to_vec(A::InfinitePEPS) -> vec, state_from_vec

Vectorize an `InfinitePEPS` into a vector of real numbers. A vectorized infinite PEPS can
retrieved again as an `InfinitePEPS` by application of the `state_from_vec` map.
"""
function FiniteDifferences.to_vec(A::InfinitePEPS)
    vec, back = FiniteDifferences.to_vec(unitcell(A))
    function state_from_vec(vec)
        return NWType(back(vec))
    end
    return vec, state_from_vec
end

## Chainrules

function ChainRulesCore.rrule(::typeof(Base.getindex), network::InfinitePEPS, args...)
    tensor = network[args...]

    function getindex_pullback(Δtensor_)
        Δtensor = unthunk(Δtensor_)
        Δnetwork = zerovector(network)
        Δnetwork[args...] = Δtensor
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return tensor, getindex_pullback
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

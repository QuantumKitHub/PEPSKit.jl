"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.

## Fields

$(TYPEDFIELDS)
"""
struct InfinitePEPS{T <: PEPSTensor}
    A::Matrix{T}
    InfinitePEPS{T}(A::Matrix{T}) where {T <: PEPSTensor} = new{T}(A)
    function InfinitePEPS(A::Array{T, 2}) where {T <: PEPSTensor}
        bosonic_braiding = BraidingStyle(sectortype(T)) === Bosonic()
        for (d, w) in Tuple.(CartesianIndices(A))
            (bosonic_braiding || !isdual(physicalspace(A[d, w]))) ||
                throw(ArgumentError("Dual physical spaces for symmetry sectors with non-trivial twists are not allowed (for now)."))
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

## Constructors

"""
    InfinitePEPS(A::AbstractMatrix{T})

Create an `InfinitePEPS` by specifying a matrix containing the PEPS tensors at each site in
the unit cell.
"""
function InfinitePEPS(A::AbstractMatrix{<:PEPSTensor})
    return InfinitePEPS(Array(deepcopy(A))) # TODO: find better way to copy
end

"""
    InfinitePEPS([f=randn, T=ComplexF64,] Pspaces::A, Nspaces::A, [Espaces::A]) where {A<:AbstractMatrix{ElementarySpace}}

Create an `InfinitePEPS` by specifying the physical, north virtual and east virtual spaces
of the PEPS tensor at each site in the unit cell as a matrix.
"""
function InfinitePEPS(
        f, T::Type{<:Number}, Pspaces::M, Nspaces::M, Espaces::M = Nspaces
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (-1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, 1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return PEPSTensor(f, T, P, N, E, S, W)
    end

    return InfinitePEPS(A)
end
function InfinitePEPS(
        Pspaces::A, virtual_spaces...; kwargs...
    ) where {A <: Union{AbstractMatrix{<:ElementarySpace}, ElementarySpace}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, virtual_spaces...; kwargs...)
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
function InfinitePEPS(A::T; unitcell::Tuple{Int, Int} = (1, 1)) where {T <: PEPSTensor}
    return InfinitePEPS(fill(A, unitcell))
end

# expand PEPS spaces to unit cell size
function _fill_state_physical_spaces(
        Pspace::S; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return fill(Pspace, unitcell)
end
function _fill_state_virtual_spaces(
        Nspace::S, Espace::S = Nspace; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return fill(Nspace, unitcell), fill(Espace, unitcell)
end

"""
    InfinitePEPS([f=randn, T=ComplexF64,] Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePEPS by specifying its physical, north and east spaces and unit cell.
"""
function InfinitePEPS(
        f, T::Type{<:Number}, Pspace::S, vspaces...; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return InfinitePEPS(
        f, T,
        _fill_state_physical_spaces(Pspace; unitcell),
        _fill_state_virtual_spaces(vspaces...; unitcell)...,
    )
end

## Unit cell interface

unitcell(t::InfinitePEPS) = t.A
Base.size(A::InfinitePEPS, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePEPS) = length(unitcell(A))
Base.eltype(::Type{InfinitePEPS{T}}) where {T} = T
Base.eltype(A::InfinitePEPS) = eltype(typeof(A))

Base.copy(A::InfinitePEPS) = InfinitePEPS(copy(unitcell(A)))
function Base.similar(A::InfinitePEPS, T::Type{TorA} = scalartype(A)) where {TorA}
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

TensorKit.spacetype(::Type{T}) where {T <: InfinitePEPS} = spacetype(eltype(T))
virtualspace(n::InfinitePEPS, dir) = virtualspace.(unitcell(n), dir)
virtualspace(n::InfinitePEPS, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)
physicalspace(n::InfinitePEPS) = physicalspace.(unitcell(n))
physicalspace(n::InfinitePEPS, r::Int, c::Int) = physicalspace(n[r, c])

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(top::InfinitePEPS, bot::InfinitePEPS = top)
    size(top) == size(bot) || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have the same length")
    )
    return InfiniteSquareNetwork(map(tuple, unitcell(top), unitcell(bot)))
end

## Vector interface

VI.scalartype(::Type{NT}) where {NT <: InfinitePEPS} = scalartype(eltype(NT))
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

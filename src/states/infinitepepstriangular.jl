"""
    struct InfinitePEPSTriangular{T<:PEPSTensorTriangularTriangular}

Represents an infinite projected entangled-pair state on a 2D triangular lattice.

## Fields

$(TYPEDFIELDS)
"""
struct InfinitePEPSTriangular{T <: PEPSTensorTriangular}
    A::Matrix{T}
    InfinitePEPSTriangular{T}(A::Matrix{T}) where {T <: PEPSTensorTriangular} = new{T}(A)
    function InfinitePEPSTriangular(A::Array{T, 2}) where {T <: PEPSTensorTriangular}
        bosonic_braiding = BraidingStyle(sectortype(T)) === Bosonic()
        # for (d, w) in Tuple.(CartesianIndices(A))
        #     (bosonic_braiding || !isdual(physicalspace(A[d, w]))) ||
        #         throw(ArgumentError("Dual physical spaces for symmetry sectors with non-trivial twists are not allowed (for now)."))
        #     north_virtualspace(A[d, w]) == south_virtualspace(A[_prev(d, end), w])' ||
        #         throw(
        #         SpaceMismatch("North virtual space at site $((d, w)) does not match.")
        #     )
        #     east_virtualspace(A[d, w]) == west_virtualspace(A[d, _next(w, end)])' ||
        #         throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        #     dim(space(A[d, w])) > 0 || @warn "no fusion channels at site ($d, $w)"
        # end
        return new{T}(A)
    end
end

## Constructors

"""
    InfinitePEPSTriangular(A::AbstractMatrix{T})

Create an `InfinitePEPSTriangular` by specifying a matrix containing the PEPS tensors at each site in
the unit cell.
"""
function InfinitePEPSTriangular(A::AbstractMatrix{<:PEPSTensorTriangular})
    return InfinitePEPSTriangular(Array(deepcopy(A))) # TODO: find better way to copy
end

"""
    InfinitePEPSTriangular([f=randn, T=ComplexF64,] Pspaces::A, Nspaces::A, [Espaces::A]) where {A<:AbstractMatrix{ElementarySpace}}

Create an `InfinitePEPSTriangular` by specifying the physical, north virtual and east virtual spaces
of the PEPS tensor at each site in the unit cell as a matrix.
"""
function InfinitePEPSTriangular(
        f, T::Type{<:Number}, Pspaces::M, NWspaces::M, NEspaces::M = NWspaces, Espaces::M = NWspaces
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
    size(Pspaces) == size(NEspaces) == size(NEspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    SEspaces = adjoint.(circshift(NWspaces, (-1, 0)))
    SWspaces = adjoint.(NEspaces)
    Wspaces = adjoint.(circshift(Espaces, (0, 1)))

    A = map(Pspaces, NWspaces, NEspaces, Espaces, SEspaces, SWspaces, Wspaces) do NW, NE, E, SE, SW, W
        return PEPSTensorTriangular(f, T, P, NW, NE, E, SE, SW, W)
    end

    return InfinitePEPSTriangular(A)
end
function InfinitePEPSTriangular(
        Pspaces::A, virtual_spaces...; kwargs...
    ) where {A <: Union{AbstractMatrix{<:ElementarySpace}, ElementarySpace}}
    return InfinitePEPSTriangular(randn, ComplexF64, Pspaces, virtual_spaces...; kwargs...)
end

"""
    InfinitePEPSTriangular(A::PEPSTensorTriangular; unitcell=(1, 1))

Create an `InfinitePEPSTriangular` by specifying a tensor and unit cell.

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
function InfinitePEPSTriangular(A::T; unitcell::Tuple{Int, Int} = (1, 1)) where {T <: PEPSTensorTriangular}
    return InfinitePEPSTriangular(fill(A, unitcell))
end

# expand PEPS spaces to unit cell size
function _fill_state_virtual_spaces_triangular(
        NWspace::S, NEspace::S = NWspace, Espace::S = NWspace; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return fill(NWspace, unitcell), fill(NEspace, unitcell), fill(Espace, unitcell)
end

"""
    InfinitePEPSTriangular([f=randn, T=ComplexF64,] Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePEPSTriangular by specifying its physical, north and east spaces and unit cell.
"""
function InfinitePEPSTriangular(
        f, T::Type{<:Number}, Pspace::S, vspaces...; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return InfinitePEPSTriangular(
        f, T,
        _fill_state_physical_spaces(Pspace; unitcell),
        _fill_state_virtual_spaces_triangular(vspaces...; unitcell)...,
    )
end

## Unit cell interface

unitcell(t::InfinitePEPSTriangular) = t.A
Base.size(A::InfinitePEPSTriangular, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePEPSTriangular) = length(unitcell(A))
Base.eltype(::Type{InfinitePEPSTriangular{T}}) where {T} = T
Base.eltype(A::InfinitePEPSTriangular) = eltype(typeof(A))

Base.copy(A::InfinitePEPSTriangular) = InfinitePEPSTriangular(copy(unitcell(A)))
function Base.similar(A::InfinitePEPSTriangular, T::Type{TorA} = scalartype(A)) where {TorA}
    return InfinitePEPSTriangular(map(t -> similar(t, T), unitcell(A)))
end
Base.repeat(A::InfinitePEPSTriangular, counts...) = InfinitePEPSTriangular(repeat(unitcell(A), counts...))

Base.getindex(A::InfinitePEPSTriangular, args...) = Base.getindex(unitcell(A), args...)
Base.setindex!(A::InfinitePEPSTriangular, args...) = (Base.setindex!(unitcell(A), args...); A)
Base.axes(A::InfinitePEPSTriangular, args...) = axes(unitcell(A), args...)
eachcoordinate(A::InfinitePEPSTriangular) = collect(Iterators.product(axes(A)...))
function eachcoordinate(A::InfinitePEPSTriangular, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces

TensorKit.spacetype(::Type{T}) where {T <: InfinitePEPSTriangular} = spacetype(eltype(T))
virtualspace(n::InfinitePEPSTriangular, dir) = virtualspace.(unitcell(n), dir)
function virtualspace(n::InfinitePEPSTriangular, r::Int, c::Int, dir)
    Nr, Nc = size(n)
    return virtualspace(n[mod1(r, Nr), mod1(c, Nc)], dir)
end
physicalspace(n::InfinitePEPSTriangular) = physicalspace.(unitcell(n))
function physicalspace(n::InfinitePEPSTriangular, r::Int, c::Int)
    Nr, Nc = size(n)
    return physicalspace(n[mod1(r, Nr), mod1(c, Nc)])
end

## InfiniteTriangularNetwork interface

function InfiniteTriangularNetwork(top::InfinitePEPSTriangular, bot::InfinitePEPSTriangular = top)
    size(top) == size(bot) || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO rows should have the same length")
    )
    return InfiniteTriangularNetwork(map(tuple, unitcell(top), unitcell(bot)))
end

## Vector interface

VI.scalartype(::Type{NT}) where {NT <: InfinitePEPSTriangular} = scalartype(eltype(NT))
VI.zerovector(A::InfinitePEPSTriangular) = InfinitePEPSTriangular(zerovector(unitcell(A)))

function VI.scale(ψ::InfinitePEPSTriangular, α::Number)
    _scale = Base.Fix2(scale, α)
    return InfinitePEPSTriangular(map(_scale, unitcell(ψ)))
end
function VI.scale!(ψ::InfinitePEPSTriangular, α::Number)
    _scale! = Base.Fix2(scale!, α)
    foreach(_scale!, unitcell(ψ))
    return ψ
end
function VI.scale!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    foreach(_scale!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.scale!!(ψ::InfinitePEPSTriangular, α::Number) = scale!(ψ, α)
VI.scale!!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number) = scale!(ψ₁, ψ₂, α)

function VI.add(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return InfinitePEPSTriangular(map(_add, unitcell(ψ₁), unitcell(ψ₂)))
end
function VI.add!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.add!!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number) = add!(ψ₁, ψ₂, α, β)

## Math

function Base.:+(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    return InfinitePEPSTriangular(unitcell(A₁) + unitcell(A₂))
end
function Base.:-(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    return InfinitePEPSTriangular(unitcell(A₁) - unitcell(A₂))
end
Base.:*(α::Number, A::InfinitePEPSTriangular) = InfinitePEPSTriangular(α * unitcell(A))
Base.:*(A::InfinitePEPSTriangular, α::Number) = α * A
Base.:/(A::InfinitePEPSTriangular, α::Number) = InfinitePEPSTriangular(unitcell(A) / α)
LinearAlgebra.dot(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular) = dot(unitcell(A₁), unitcell(A₂))
LinearAlgebra.norm(A::InfinitePEPSTriangular) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations

# rotl60(A::InfinitePEPSTriangular) = InfinitePEPSTriangular(rotl60(_rotl60_localsandwich.(unitcell(A))))
# rotr60(A::InfinitePEPSTriangular) = InfinitePEPSTriangular(rotr60(_rotl60_localsandwich.(unitcell(A))))
# Base.rot180(A::InfinitePEPSTriangular) = InfinitePEPSTriangular(rot180(rot180.(unitcell(A))))

## FiniteDifferences vectorization

"""
    to_vec(A::InfinitePEPSTriangular) -> vec, state_from_vec

Vectorize an `InfinitePEPSTriangular` into a vector of real numbers. A vectorized infinite PEPS can
retrieved again as an `InfinitePEPSTriangular` by application of the `state_from_vec` map.
"""
function FiniteDifferences.to_vec(A::InfinitePEPSTriangular)
    vec, back = FiniteDifferences.to_vec(unitcell(A))
    function state_from_vec(vec)
        return NWType(back(vec))
    end
    return vec, state_from_vec
end

## Chainrules

function ChainRulesCore.rrule(::typeof(Base.getindex), network::InfinitePEPSTriangular, args...)
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
        ::Type{InfiniteTriangularNetwork}, top::InfinitePEPSTriangular, bot::InfinitePEPSTriangular
    )
    network = InfiniteTriangularNetwork(top, bot)

    function InfiniteTriangularNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δtop = InfinitePEPSTriangular(map(ket, unitcell(Δnetwork)))
        Δbot = InfinitePEPSTriangular(map(bra, unitcell(Δnetwork)))
        return NoTangent(), Δtop, Δbot
    end
    return network, InfiniteTriangularNetwork_pullback
end

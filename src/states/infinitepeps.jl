"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePEPS{T<:PEPSTensor} <: AbstractPEPS
    A::Matrix{T}
    InfinitePEPS{T}(A::Matrix{T}) where {T<:PEPSTensor} = new{T}(A)
    function InfinitePEPS(A::Array{T,2}) where {T<:PEPSTensor}
        for (d, w) in Tuple.(CartesianIndices(A))
            space(A[d, w], 2) == space(A[_prev(d, end), w], 4)' || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            space(A[d, w], 3) == space(A[d, _next(w, end)], 5)' ||
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
function InfinitePEPS(A::AbstractMatrix{T}) where {T<:PEPSTensor}
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
) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
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
) where {S<:Union{ElementarySpace,Int}}
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
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePEPS(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

## Shape and size
Base.size(T::InfinitePEPS) = size(T.A)
Base.size(T::InfinitePEPS, i) = size(T.A, i)
Base.length(T::InfinitePEPS) = length(T.A)
Base.eltype(T::InfinitePEPS) = eltype(typeof(T))
Base.eltype(::Type{<:InfinitePEPS{T}}) where {T} = T
VectorInterface.scalartype(::Type{T}) where {T<:InfinitePEPS} = scalartype(eltype(T))

## Copy
Base.copy(T::InfinitePEPS) = InfinitePEPS(copy(T.A))
Base.similar(T::InfinitePEPS, args...) = InfinitePEPS(similar(T.A, args...))
Base.repeat(T::InfinitePEPS, counts...) = InfinitePEPS(repeat(T.A, counts...))

Base.getindex(T::InfinitePEPS, args...) = Base.getindex(T.A, args...)
Base.setindex!(T::InfinitePEPS, args...) = (Base.setindex!(T.A, args...); T)
Base.axes(T::InfinitePEPS, args...) = axes(T.A, args...)
function eachcoordinate(x::InfinitePEPS)
    return collect(Iterators.product(axes(x)...))
end
function eachcoordinate(x::InfinitePEPS, dirs)
    return collect(Iterators.product(dirs, axes(x, 1), axes(x, 2)))
end
TensorKit.space(t::InfinitePEPS, i, j) = space(t[i, j], 1)

## Math
Base.:+(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A + ψ₂.A)
Base.:-(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A - ψ₂.A)
Base.:*(α::Number, ψ::InfinitePEPS) = InfinitePEPS(α * ψ.A)
Base.:/(ψ::InfinitePEPS, α::Number) = InfinitePEPS(ψ.A / α)
LinearAlgebra.dot(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = dot(ψ₁.A, ψ₂.A)
LinearAlgebra.norm(ψ::InfinitePEPS) = norm(ψ.A)

## (Approximate) equality
function Base.:(==)(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS)
    return all(zip(ψ₁.A, ψ₂.A)) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS; kwargs...)
    return all(zip(ψ₁.A, ψ₂.A)) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

# Used in _scale during OptimKit.optimize
function LinearAlgebra.rmul!(ψ::InfinitePEPS, α::Number)
    rmul!.(ψ.A, α)
    return ψ
end

# Used in _add during OptimKit.optimize
function LinearAlgebra.axpy!(α::Number, ψ₁::InfinitePEPS, ψ₂::InfinitePEPS)
    axpy!.(α, ψ₁.A, ψ₂.A)
    return ψ₂
end

# VectorInterface
VectorInterface.zerovector(x::InfinitePEPS) = InfinitePEPS(zerovector(x.A))

# Rotations
Base.rotl90(t::InfinitePEPS) = InfinitePEPS(rotl90(rotl90.(t.A)))
Base.rotr90(t::InfinitePEPS) = InfinitePEPS(rotr90(rotr90.(t.A)))
Base.rot180(t::InfinitePEPS) = InfinitePEPS(rot180(rot180.(t.A)))

# Chainrules
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), state::InfinitePEPS, row::Int, col::Int
)
    pepstensor = state[row, col]

    function getindex_pullback(Δpepstensor)
        Δstate = zerovector(state)
        Δstate[row, col] = Δpepstensor
        return NoTangent(), Δstate, NoTangent(), NoTangent()
    end
    return pepstensor, getindex_pullback
end

function ChainRulesCore.rrule(::Type{<:InfinitePEPS}, A::Matrix{T}) where {T<:PEPSTensor}
    peps = InfinitePEPS(A)
    function InfinitePEPS_pullback(Δpeps)
        return NoTangent(), Δpeps.A
    end
    return peps, InfinitePEPS_pullback
end

function ChainRulesCore.rrule(::typeof(rotl90), peps::InfinitePEPS)
    peps′ = rotl90(peps)
    function rotl90_pullback(Δpeps)
        return NoTangent(), rotr90(Δpeps)
    end
    return peps′, rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), peps::InfinitePEPS)
    peps′ = rotr90(peps)
    function rotr90_pullback(Δpeps)
        return NoTangent(), rotl90(Δpeps)
    end
    return peps′, rotr90_pullback
end

# FiniteDifferences
# Makes use of tensors already having a to_vec method
function FiniteDifferences.to_vec(state::InfinitePEPS)
    vec, back = FiniteDifferences.to_vec(state.A)
    function state_from_vec(vec)
        return InfinitePEPS(back(vec))
    end
    return vec, state_from_vec
end

"""
    struct InfinitePartitionFunction{T<:PartitionFunctionTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePartitionFunction{T<:PartitionFunctionTensor} <: AbstractPartitionFunction
    A::Matrix{T}
    function InfinitePartitionFunction{T}(A::Matrix{T}) where {T<:PartitionFunctionTensor}
        return new{T}(A)
    end
    function InfinitePartitionFunction(A::Array{T,2}) where {T<:PartitionFunctionTensor}
        for (d, w) in Tuple.(CartesianIndices(A))
            space(A[d, w], 1) == space(A[_prev(d, end), w], 4)' || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            space(A[d, w], 2) == space(A[d, _next(w, end)], 3)' ||
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
) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    return InfinitePartitionFunction(randn, ComplexF64, Nspaces, Espaces)
end
function InfinitePartitionFunction(
    f, T, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
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
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePartitionFunction(
        randn, ComplexF64, fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end
function InfinitePartitionFunction(
    f, T, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePartitionFunction(f, T, fill(Nspace, unitcell), fill(Espace, unitcell))
end

## Shape and size
Base.size(T::InfinitePartitionFunction) = size(T.A)
Base.size(T::InfinitePartitionFunction, i) = size(T.A, i)
Base.length(T::InfinitePartitionFunction) = length(T.A)
Base.eltype(T::InfinitePartitionFunction) = eltype(typeof(T))
Base.eltype(::Type{<:InfinitePartitionFunction{T}}) where {T} = T
function VectorInterface.scalartype(::Type{T}) where {T<:InfinitePartitionFunction}
    return scalartype(eltype(T))
end

## Copy
Base.copy(T::InfinitePartitionFunction) = InfinitePartitionFunction(copy(T.A))
function Base.similar(T::InfinitePartitionFunction, args...)
    return InfinitePartitionFunction(similar(T.A, args...))
end
function Base.repeat(T::InfinitePartitionFunction, counts...)
    return InfinitePartitionFunction(repeat(T.A, counts...))
end

Base.getindex(T::InfinitePartitionFunction, args...) = Base.getindex(T.A, args...)
Base.setindex!(T::InfinitePartitionFunction, args...) = (Base.setindex!(T.A, args...); T)
Base.axes(T::InfinitePartitionFunction, args...) = axes(T.A, args...)
function eachcoordinate(x::InfinitePartitionFunction)
    return collect(Iterators.product(axes(x)...))
end
function eachcoordinate(x::InfinitePartitionFunction, dirs)
    return collect(Iterators.product(dirs, axes(x, 1), axes(x, 2)))
end
TensorKit.space(t::InfinitePartitionFunction, i, j) = space(t[i, j], 1)

## Math
function Base.:+(ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction)
    return InfinitePartitionFunction(ψ₁.A + ψ₂.A)
end
function Base.:-(ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction)
    return InfinitePartitionFunction(ψ₁.A - ψ₂.A)
end
Base.:*(α::Number, ψ::InfinitePartitionFunction) = InfinitePartitionFunction(α * ψ.A)
Base.:/(ψ::InfinitePartitionFunction, α::Number) = InfinitePartitionFunction(ψ.A / α)
function LinearAlgebra.dot(ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction)
    return dot(ψ₁.A, ψ₂.A)
end
LinearAlgebra.norm(ψ::InfinitePartitionFunction) = norm(ψ.A)

## (Approximate) equality
function Base.:(==)(ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction)
    return all(zip(ψ₁.A, ψ₂.A)) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(
    ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction; kwargs...
)
    return all(zip(ψ₁.A, ψ₂.A)) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

# Used in _scale during OptimKit.optimize
function LinearAlgebra.rmul!(ψ::InfinitePartitionFunction, α::Number)
    rmul!(ψ.A, α)
    return ψ
end

# Used in _add during OptimKit.optimize
function LinearAlgebra.axpy!(
    α::Number, ψ₁::InfinitePartitionFunction, ψ₂::InfinitePartitionFunction
)
    axpy!.(α, ψ₁.A, ψ₂.A)
    return ψ₂
end

# VectorInterface
function VectorInterface.zerovector(x::InfinitePartitionFunction)
    return InfinitePartitionFunction(zerovector(x.A))
end

# Rotations
Base.rotl90(t::InfinitePartitionFunction) = InfinitePartitionFunction(rotl90(rotl90.(t.A)))
Base.rotr90(t::InfinitePartitionFunction) = InfinitePartitionFunction(rotr90(rotr90.(t.A)))
Base.rot180(t::InfinitePartitionFunction) = InfinitePartitionFunction(rot180(rot180.(t.A)))

# Chainrules
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), state::InfinitePartitionFunction, row::Int, col::Int
)
    PartitionFunctionTensor = state[row, col]

    function getindex_pullback(ΔPartitionFunction)
        Δstate = zerovector(state)
        Δstate[row, col] = ΔPartitionFunction
        return NoTangent(), Δstate, NoTangent(), NoTangent()
    end
    return PartitionFunctionTensor, getindex_pullback
end

function ChainRulesCore.rrule(
    ::Type{<:InfinitePartitionFunction}, A::Matrix{T}
) where {T<:PartitionFunctionTensor}
    peps = InfinitePartitionFunction(A)
    function InfinitePartitionFunction_pullback(Δpeps)
        return NoTangent(), Δpeps.A
    end
    return peps, InfinitePartitionFunction_pullback
end

function ChainRulesCore.rrule(::typeof(rotl90), peps::InfinitePartitionFunction)
    peps′ = rotl90(peps)
    function rotl90_pullback(Δpeps)
        return NoTangent(), rotr90(Δpeps)
    end
    return peps′, rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), peps::InfinitePartitionFunction)
    peps′ = rotr90(peps)
    function rotr90_pullback(Δpeps)
        return NoTangent(), rotl90(Δpeps)
    end
    return peps′, rotr90_pullback
end

# FiniteDifferences
# Makes use of tensors already having a to_vec method
function FiniteDifferences.to_vec(state::InfinitePartitionFunction)
    vec, back = FiniteDifferences.to_vec(state.A)
    function state_from_vec(vec)
        return InfinitePartitionFunction(back(vec))
    end
    return vec, state_from_vec
end

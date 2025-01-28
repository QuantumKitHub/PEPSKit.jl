"""
    InfiniteSquareNetwork{T,N}

Abstract infinite tensor network consisting of a translationally invariant unit cell
on a square lattice.
"""
abstract type InfiniteSquareNetwork{T,N} end

## Shape and size
function unitcell(::InfiniteSquareNetwork) end  # Return array of constituent tensors
Base.size(A::InfiniteSquareNetwork, args...) = size(unitcell(A), args...)
Base.length(A::InfiniteSquareNetwork) = length(unitcell(A))
Base.eltype(::Type{<:InfiniteSquareNetwork{T}}) where {T} = T
Base.eltype(A::InfiniteSquareNetwork) = eltype(typeof(A))

## Copy
Base.copy(A::NWType) where {NWType<:InfiniteSquareNetwork} = NWType(copy(unitcell(A)))
function Base.similar(A::NWType, args...) where {NWType<:InfiniteSquareNetwork}
    return NWType(similar(unitcell(A), args...))
end
function Base.repeat(A::NWType, counts...) where {NWType<:InfiniteSquareNetwork}
    return NWType(repeat(unitcell(A), counts...))
end

## Indexing
Base.getindex(A::InfiniteSquareNetwork, args...) = Base.getindex(unitcell(A), args...)
function Base.setindex!(A::InfiniteSquareNetwork, args...)
    return (Base.setindex!(unitcell(A), args...); A)
end
Base.axes(A::InfiniteSquareNetwork, args...) = axes(unitcell(A), args...)
function eachcoordinate(A::InfiniteSquareNetwork)
    return collect(Iterators.product(axes(A)...))
end
function eachcoordinate(A::InfiniteSquareNetwork, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Vector interface
function VectorInterface.scalartype(::Type{NWType}) where {NWType<:InfiniteSquareNetwork}
    return scalartype(eltype(NWType))
end
function VectorInterface.zerovector(A::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(zerovector(unitcell(A)))
end

## Math
function Base.:+(A₁::NWType, A₂::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A₁) + unitcell(A₂))
end
function Base.:-(A₁::NWType, A₂::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A₁) - unitcell(A₂))
end
function Base.:*(α::Number, A::NWType) where {NWType<:InfiniteSquareNetwork}
    return NWType(α * unitcell(A))
end
function Base.:/(A::NWType, α::Number) where {NWType<:InfiniteSquareNetwork}
    return NWType(unitcell(A) / α)
end
function LinearAlgebra.dot(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return dot(unitcell(A₁), unitcell(A₂))
end
LinearAlgebra.norm(A::InfiniteSquareNetwork) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations
function Base.rotl90(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,2}} # Rotations of matrix unit cells
    return NWType(rotl90(rotl90.(unitcell(A))))
end
function Base.rotr90(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,2}}
    return NWType(rotr90(rotr90.(unitcell(A))))
end
function Base.rot180(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,2}}
    return NWType(rot180(rot180.(unitcell(A))))
end
function Base.rotl90(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,3}} # Rotations of cubic unit cells along z-axis
    return NWType(stack(rotl90, eachslice(unitcell(A); dims=3)))
end
function Base.rotr90(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,3}}
    return NWType(stack(rotr90, eachslice(unitcell(A); dims=3)))
end
function Base.rot180(A::NWType) where {NWType<:InfiniteSquareNetwork{<:Any,3}}
    return NWType(stack(rot180, eachslice(unitcell(A); dims=3)))
end

## OptimKit optimization compatibility
function LinearAlgebra.rmul!(A::InfiniteSquareNetwork, α::Number) # Used in _scale during OptimKit.optimize
    rmul!.(unitcell(A), α)
    return A
end
function LinearAlgebra.axpy!(
    α::Number, A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork
) # Used in _add during OptimKit.optimize
    axpy!.(α, unitcell(A₁), unitcell(A₂))
    return A₂
end

## FiniteDifferences vectorization
function FiniteDifferences.to_vec(A::NWType) where {NWType<:InfiniteSquareNetwork}
    vec, back = FiniteDifferences.to_vec(unitcell(A))
    function state_from_vec(vec)
        return NWType(back(vec))
    end
    return vec, state_from_vec
end

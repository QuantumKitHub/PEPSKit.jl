# To be deprecated(?): supertype for physics-inspired networks which defines an interface
# for optimization over these kinds of networks.

"""
    InfiniteGridNetwork{T,N}

Abstract infinite tensor network consisting of a translationally invariant unit cell
on a hypercubic lattice.
"""
abstract type InfiniteGridNetwork{T,N} end

## Shape and size
function unitcell(::InfiniteGridNetwork) end  # Return array of constituent tensors
Base.size(A::InfiniteGridNetwork, args...) = size(unitcell(A), args...)
Base.length(A::InfiniteGridNetwork) = length(unitcell(A))
Base.eltype(::Type{<:InfiniteGridNetwork{T}}) where {T} = T
Base.eltype(A::InfiniteGridNetwork) = eltype(typeof(A))

## Copy
Base.copy(A::NWType) where {NWType<:InfiniteGridNetwork} = NWType(copy(unitcell(A)))
function Base.similar(A::NWType, args...) where {NWType<:InfiniteGridNetwork}
    return NWType(similar(unitcell(A), args...))
end
function Base.repeat(A::NWType, counts...) where {NWType<:InfiniteGridNetwork}
    return NWType(repeat(unitcell(A), counts...))
end

## Indexing
Base.getindex(A::InfiniteGridNetwork, args...) = Base.getindex(unitcell(A), args...)
function Base.setindex!(A::InfiniteGridNetwork, args...)
    return (Base.setindex!(unitcell(A), args...); A)
end
Base.axes(A::InfiniteGridNetwork, args...) = axes(unitcell(A), args...)
function eachcoordinate(A::InfiniteGridNetwork)
    return collect(Iterators.product(axes(A)...))
end
function eachcoordinate(A::InfiniteGridNetwork, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces
virtualspace(n::InfiniteGridNetwork, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)
physicalspace(n::InfiniteGridNetwork, r::Int, c::Int) = physicalspace(n[r, c])
function virtualspace(n::InfiniteGridNetwork, r::Int, c::Int, h::Int, dir)
    return virtualspace(n[r, c, h], dir)
end
physicalspace(n::InfiniteGridNetwork, r::Int, c::Int, h::Int) = physicalspace(n[r, c, h])

## Vector interface
function VectorInterface.scalartype(::Type{NWType}) where {NWType<:InfiniteGridNetwork}
    return scalartype(eltype(NWType))
end
function VectorInterface.zerovector(A::NWType) where {NWType<:InfiniteGridNetwork}
    return NWType(zerovector(unitcell(A)))
end

## Math
function Base.:+(A₁::NWType, A₂::NWType) where {NWType<:InfiniteGridNetwork}
    return NWType(unitcell(A₁) + unitcell(A₂))
end
function Base.:-(A₁::NWType, A₂::NWType) where {NWType<:InfiniteGridNetwork}
    return NWType(unitcell(A₁) - unitcell(A₂))
end
function Base.:*(α::Number, A::NWType) where {NWType<:InfiniteGridNetwork}
    return NWType(α * unitcell(A))
end
function Base.:/(A::NWType, α::Number) where {NWType<:InfiniteGridNetwork}
    return NWType(unitcell(A) / α)
end
function LinearAlgebra.dot(A₁::InfiniteGridNetwork, A₂::InfiniteGridNetwork)
    return dot(unitcell(A₁), unitcell(A₂))
end
LinearAlgebra.norm(A::InfiniteGridNetwork) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfiniteGridNetwork, A₂::InfiniteGridNetwork)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfiniteGridNetwork, A₂::InfiniteGridNetwork; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations
function Base.rotl90(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,2}} # Rotations of matrix unit cells
    return NWType(rotl90(rotl90.(unitcell(A))))
end
function Base.rotr90(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,2}}
    return NWType(rotr90(rotr90.(unitcell(A))))
end
function Base.rot180(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,2}}
    return NWType(rot180(rot180.(unitcell(A))))
end
function Base.rotl90(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,3}} # Rotations of cubic unit cells along z-axis
    return NWType(stack(rotl90, eachslice(unitcell(A); dims=3)))
end
function Base.rotr90(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,3}}
    return NWType(stack(rotr90, eachslice(unitcell(A); dims=3)))
end
function Base.rot180(A::NWType) where {NWType<:InfiniteGridNetwork{<:Any,3}}
    return NWType(stack(rot180, eachslice(unitcell(A); dims=3)))
end

## OptimKit optimization compatibility
function LinearAlgebra.rmul!(A::InfiniteGridNetwork, α::Number) # Used in _scale during OptimKit.optimize
    rmul!.(unitcell(A), α)
    return A
end
function LinearAlgebra.axpy!(α::Number, A₁::InfiniteGridNetwork, A₂::InfiniteGridNetwork) # Used in _add during OptimKit.optimize
    axpy!.(α, unitcell(A₁), unitcell(A₂))
    return A₂
end

## FiniteDifferences vectorization
function FiniteDifferences.to_vec(A::NWType) where {NWType<:InfiniteGridNetwork}
    vec, back = FiniteDifferences.to_vec(unitcell(A))
    function state_from_vec(vec)
        return NWType(back(vec))
    end
    return vec, state_from_vec
end

## Chainrules
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), network::InfiniteGridNetwork, args...
)
    tensor = network[args...]

    function getindex_pullback(Δtensor_)
        Δtensor = unthunk(Δtensor_)
        Δnetwork = zerovector(network)
        Δnetwork[args...] = Δtensor
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return tensor, getindex_pullback
end

# TODO: no longer used?
function ChainRulesCore.rrule(::Type{NWType}, A::Array) where {NWType<:InfiniteGridNetwork}
    network = NWType(A)
    function InfiniteGridNetwork_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), unnitcell(Δnetwork)
    end
    return network, InfiniteGridNetwork_pullback
end

# TODO: no longer used?
function ChainRulesCore.rrule(::typeof(rotl90), network::InfiniteGridNetwork)
    network´ = rotl90(network)
    function rotl90_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), rotr90(Δnetwork)
    end
    return network´, rotl90_pullback
end

# TODO: no longer used?
function ChainRulesCore.rrule(::typeof(rotr90), network::InfiniteGridNetwork)
    network´ = rotr90(network)
    function rotr90_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), rotl90(Δnetwork)
    end
    return network´, rotr90_pullback
end

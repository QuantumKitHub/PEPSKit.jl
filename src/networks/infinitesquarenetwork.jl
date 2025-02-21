"""
    InfiniteSquareNetwork{O}

Contractible square network. Wraps a matrix of 'rank-4-tensor-like' objects.
"""
struct InfiniteSquareNetwork{O}
    A::Matrix{O}
    InfiniteSquareNetwork{O}(A::Matrix{O}) where {O} = new{O}(A)
    function InfiniteSquareNetwork(A::Matrix)
        for I in eachindex(IndexCartesian(), A)
            d, w = Tuple(I)
            north_virtualspace(A[d, w]) ==
            _elementwise_dual(south_virtualspace(A[_prev(d, end), w])) || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            east_virtualspace(A[d, w]) ==
            _elementwise_dual(west_virtualspace(A[d, _next(w, end)])) ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        end
        return InfiniteSquareNetwork{eltype(A)}(A)
    end
end
InfiniteSquareNetwork(n::InfiniteSquareNetwork) = n

## Unit cell interface

unitcell(n::InfiniteSquareNetwork) = n.A
Base.size(n::InfiniteSquareNetwork, args...) = size(unitcell(n), args...)
Base.length(n::InfiniteSquareNetwork) = length(unitcell(n))
Base.eltype(n::InfiniteSquareNetwork) = eltype(typeof(n))
Base.eltype(::Type{InfiniteSquareNetwork{O}}) where {O} = O

Base.copy(n::InfiniteSquareNetwork) = InfiniteSquareNetwork(copy(unitcell(n)))
function Base.similar(n::InfiniteSquareNetwork, args...)
    return InfiniteSquareNetwork(similar(unitcell(n), args...))
end
function Base.repeat(n::InfiniteSquareNetwork, counts...)
    return InfiniteSquareNetwork(repeat(unitcell(n), counts...))
end

## Indexing
Base.getindex(n::InfiniteSquareNetwork, args...) = Base.getindex(unitcell(n), args...)
function Base.setindex!(n::InfiniteSquareNetwork, args...)
    return (Base.setindex!(unitcell(n), args...); n)
end
Base.axes(n::InfiniteSquareNetwork, args...) = axes(unitcell(n), args...)
eachcoordinate(n::InfiniteSquareNetwork) = collect(Iterators.product(axes(n)...))
function eachcoordinate(n::InfiniteSquareNetwork, dirs)
    return collect(Iterators.product(dirs, axes(n, 1), axes(n, 2)))
end

## Spaces

virtualspace(n::InfiniteSquareNetwork, r::Int, c::Int, dir) = virtualspace(n[r, c], dir)

## Vector interface
function VectorInterface.scalartype(::Type{T}) where {T<:InfiniteSquareNetwork}
    return scalartype(eltype(T))
end
function VectorInterface.zerovector(A::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(zerovector(unitcell(A)))
end

## Math (for Zygote accumulation)

function Base.:+(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(_add_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:-(A₁::InfiniteSquareNetwork, A₂::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(_subtract_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:*(α::Number, A::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(_mul_localsandwich.(α, unitcell(A)))
end
Base.:*(A::InfiniteSquareNetwork, α::Number) = α * A
function Base.:/(A::InfiniteSquareNetwork, α::Number)
    return A * inv(α)
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

function Base.rotl90(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rotl90(_rotl90_localsandwich.(unitcell(n))))
end
function Base.rotr90(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rotr90(_rotr90_localsandwich.(unitcell(n))))
end
function Base.rot180(n::InfiniteSquareNetwork)
    return InfiniteSquareNetwork(rot180(_rot180_localsandwich.(unitcell(n))))
end

## Chainrules

# generic implementation
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), network::InfiniteSquareNetwork, r::Int, c::Int
)
    O = network[r, c]

    function getindex_pullback(ΔO_)
        ΔO = map(unthunk, ΔO_)
        if ΔO isa Tangent
            ΔO = ChainRulesCore.construct(typeof(O), ChainRulesCore.backing(ΔO))
        end
        Δnetwork = zerovector(network)
        Δnetwork[r, c] = ΔO
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return O, getindex_pullback
end

# specialized PFTensor implementation
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), network::InfiniteSquareNetwork{<:PFTensor}, r::Int, c::Int
)
    O = network[r, c]

    function getindex_pullback(ΔO_)
        ΔO = unthunk(ΔO_)
        Δnetwork = zerovector(network)
        Δnetwork[r, c] = ΔO
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return O, getindex_pullback
end

function ChainRulesCore.rrule(::typeof(rotl90), network::InfiniteSquareNetwork)
    network´ = rotl90(network)
    function rotl90_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        return NoTangent(), rotr90(Δnetwork)
    end
    return network´, rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), network::InfiniteSquareNetwork)
    network´ = rotr90(network)
    function rotr90_pullback(Δnetwork)
        Δnetwork = unthunk(Δnetwork)
        return NoTangent(), rotl90(Δnetwork)
    end
    return network´, rotr90_pullback
end

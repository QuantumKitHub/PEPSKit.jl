const PEPSTensorTriangular{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 1, 6}

"""
$(TYPEDEF)

Contractible square network. Wraps a matrix of 'rank-6-tensor-like' objects.

## Fields

$(TYPEDFIELDS)
"""
struct InfiniteTriangularNetwork{O}
    A::Matrix{O}
    InfiniteTriangularNetwork{O}(A::Matrix{O}) where {O} = new{O}(A)
    function InfiniteTriangularNetwork(A::Matrix)
        # for I in eachindex(IndexCartesian(), A)
        #     d, w = Tuple(I)
        #     northwest_virtualspace(A[d, w]) ==
        #         _elementwise_dual(southeast_virtualspace(A[_prev(d, end), w])) || throw(
        #         SpaceMismatch("North virtual space at site $((d, w)) does not match.")
        #     )
        #     northeast_virtualspace(A[d, w]) ==
        #         _elementwise_dual(southwest_virtualspace(A[d, _next(w, end)])) ||
        #         throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        #     east_virtualspace(A[d, w]) ==
        #         _elementwise_dual(west_virtualspace(A[d, _next(w, end)])) ||
        #         throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        # end
        return InfiniteTriangularNetwork{eltype(A)}(A)
    end
end
InfiniteTriangularNetwork(n::InfiniteTriangularNetwork) = n

## Unit cell interface

unitcell(n::InfiniteTriangularNetwork) = n.A
Base.size(n::InfiniteTriangularNetwork, args...) = size(unitcell(n), args...)
Base.length(n::InfiniteTriangularNetwork) = length(unitcell(n))
Base.eltype(n::InfiniteTriangularNetwork) = eltype(typeof(n))
Base.eltype(::Type{InfiniteTriangularNetwork{O}}) where {O} = O

Base.copy(n::InfiniteTriangularNetwork) = InfiniteTriangularNetwork(copy(unitcell(n)))
function Base.similar(n::InfiniteTriangularNetwork, T::Type{TorA} = scalartype(n)) where {TorA}
    return InfiniteTriangularNetwork(map(t -> similar(t, T), unitcell(n)))
end
function Base.repeat(n::InfiniteTriangularNetwork, counts...)
    return InfiniteTriangularNetwork(repeat(unitcell(n), counts...))
end

## Indexing
Base.getindex(n::InfiniteTriangularNetwork, args...) = Base.getindex(unitcell(n), args...)
function Base.setindex!(n::InfiniteTriangularNetwork, args...)
    return (Base.setindex!(unitcell(n), args...); n)
end
Base.axes(n::InfiniteTriangularNetwork, args...) = axes(unitcell(n), args...)
eachcoordinate(n::InfiniteTriangularNetwork) = collect(Iterators.product(axes(n)...))
function eachcoordinate(n::InfiniteTriangularNetwork, dirs)
    return collect(Iterators.product(dirs, axes(n, 1), axes(n, 2)))
end

## Spaces

TensorKit.spacetype(::Type{T}) where {T <: InfiniteTriangularNetwork} = spacetype(eltype(T))
function virtualspace(n::InfiniteTriangularNetwork, r::Int, c::Int, dir)
    Nr, Nc = size(n)
    return virtualspace(n[mod1(r, Nr), mod1(c, Nc)], dir)
end

## Vector interface

function VectorInterface.scalartype(::Type{T}) where {T <: InfiniteTriangularNetwork}
    return scalartype(eltype(T))
end
function VectorInterface.zerovector(A::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(zerovector(unitcell(A)))
end

## Math (for Zygote accumulation)

function Base.:+(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(_add_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:-(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(_subtract_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:*(α::Number, A::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(_mul_localsandwich.(α, unitcell(A)))
end
Base.:*(A::InfiniteTriangularNetwork, α::Number) = α * A
function Base.:/(A::InfiniteTriangularNetwork, α::Number)
    return A * inv(α)
end
function LinearAlgebra.dot(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    return dot(unitcell(A₁), unitcell(A₂))
end
LinearAlgebra.norm(A::InfiniteTriangularNetwork) = norm(unitcell(A))

## (Approximate) equality

function Base.:(==)(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return _isapprox_localsandwich(p₁, p₂; kwargs...)
    end
end

## Rotations

function rotl60(n::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(rotl60(_rotl60_localsandwich.(unitcell(n))))
end
function rotr60(n::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(rotr60(_rotr60_localsandwich.(unitcell(n))))
end
function Base.rot180(n::InfiniteTriangularNetwork)
    return InfiniteTriangularNetwork(rot180(_rot180_localsandwich.(unitcell(n))))
end

## Chainrules

# generic implementation
function ChainRulesCore.rrule(
        ::typeof(Base.getindex), network::InfiniteTriangularNetwork, r::Int, c::Int
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
        ::typeof(Base.getindex), network::InfiniteTriangularNetwork{<:PFTensor}, r::Int, c::Int
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

# function ChainRulesCore.rrule(::typeof(rotl90), network::InfiniteTriangularNetwork)
#     network´ = rotl90(network)
#     function rotl90_pullback(Δnetwork_)
#         Δnetwork = unthunk(Δnetwork_)
#         return NoTangent(), rotr90(Δnetwork)
#     end
#     return network´, rotl90_pullback
# end

# function ChainRulesCore.rrule(::typeof(rotr90), network::InfiniteTriangularNetwork)
#     network´ = rotr90(network)
#     function rotr90_pullback(Δnetwork)
#         Δnetwork = unthunk(Δnetwork)
#         return NoTangent(), rotl90(Δnetwork)
#     end
#     return network´, rotr90_pullback
# end

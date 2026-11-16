const PEPSTensorTriangular{S <: ElementarySpace} = AbstractTensorMap{<:Any, S, 1, 6}

"""
$(TYPEDEF)

Contractible triangular network. Wraps a matrix of 'rank-6-tensor-like' objects, stored in
a unit cell that is periodic under the translations `(0, Ncols)` and `(Nrows, shift)`,
see [`TriangularUnitCell`](@ref).

## Fields

$(TYPEDFIELDS)
"""
struct InfiniteTriangularNetwork{O}
    A::Matrix{O}
    shift::Int
    function InfiniteTriangularNetwork{O}(A::Matrix{O}, shift::Integer = 0) where {O}
        return new{O}(A, TriangularUnitCell(size(A), shift).shift)
    end
    function InfiniteTriangularNetwork(A::Matrix; shift::Integer = 0)
        cell = TriangularUnitCell(size(A), shift)
        site(r, c) = A[_site(cell, r, c)...]
        for I in eachindex(IndexCartesian(), A)
            r, c = Tuple(I)
            virtualspace(A[r, c], 1) ==
                _elementwise_dual(virtualspace(site(r - 1, c), 4)) ||
                throw(SpaceMismatch("Northwest virtual space at site $((r, c)) does not match."))
            virtualspace(A[r, c], 2) ==
                _elementwise_dual(virtualspace(site(r - 1, c + 1), 5)) ||
                throw(SpaceMismatch("Northeast virtual space at site $((r, c)) does not match."))
            virtualspace(A[r, c], 3) ==
                _elementwise_dual(virtualspace(site(r, c + 1), 6)) ||
                throw(SpaceMismatch("East virtual space at site $((r, c)) does not match."))
        end
        return new{eltype(A)}(A, cell.shift)
    end
end
InfiniteTriangularNetwork(n::InfiniteTriangularNetwork) = n

## Unit cell interface

unitcell(n::InfiniteTriangularNetwork) = n.A
_cell(n::InfiniteTriangularNetwork) = TriangularUnitCell(size(n.A), n.shift)
@non_differentiable _cell(n::InfiniteTriangularNetwork)
_rewrap(n::InfiniteTriangularNetwork, A::AbstractMatrix) = InfiniteTriangularNetwork(A; shift = n.shift)

# representative of lattice site `[r, c]` in the unit cell of `n`
_cellsite(n, r::Int, c::Int) = _site(_cell(n), r, c)
@non_differentiable _cellsite(args...)
Base.size(n::InfiniteTriangularNetwork, args...) = size(unitcell(n), args...)
Base.length(n::InfiniteTriangularNetwork) = length(unitcell(n))
Base.eltype(n::InfiniteTriangularNetwork) = eltype(typeof(n))
Base.eltype(::Type{InfiniteTriangularNetwork{O}}) where {O} = O

Base.copy(n::InfiniteTriangularNetwork) = _rewrap(n, copy(unitcell(n)))
function Base.similar(n::InfiniteTriangularNetwork, T::Type{TorA} = scalartype(n)) where {TorA}
    return _rewrap(n, map(t -> similar(t, T), unitcell(n)))
end
function Base.repeat(n::InfiniteTriangularNetwork, counts...)
    iszero(n.shift) || throw(ArgumentError("repeating a sheared unit cell is not supported"))
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
    return virtualspace(n[_cellsite(n, r, c)...], dir)
end

## Vector interface

function VectorInterface.scalartype(::Type{T}) where {T <: InfiniteTriangularNetwork}
    return scalartype(eltype(T))
end
function VectorInterface.zerovector(A::InfiniteTriangularNetwork)
    return _rewrap(A, zerovector(unitcell(A)))
end

## Math (for Zygote accumulation)

function Base.:+(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    _check_cells(A₁, A₂)
    return _rewrap(A₁, _add_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:-(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork)
    _check_cells(A₁, A₂)
    return _rewrap(A₁, _subtract_localsandwich.(unitcell(A₁), unitcell(A₂)))
end
function Base.:*(α::Number, A::InfiniteTriangularNetwork)
    return _rewrap(A, _mul_localsandwich.(α, unitcell(A)))
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
    _cell(A₁) == _cell(A₂) || return false
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfiniteTriangularNetwork, A₂::InfiniteTriangularNetwork; kwargs...)
    _cell(A₁) == _cell(A₂) || return false
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return _isapprox_localsandwich(p₁, p₂; kwargs...)
    end
end

## Rotations

# rotating a sheared unit cell does not correspond to rotating its storage matrix
function _check_unsheared(n::InfiniteTriangularNetwork)
    iszero(n.shift) || throw(ArgumentError("rotating a sheared unit cell is not supported"))
    return nothing
end

function rotl60(n::InfiniteTriangularNetwork)
    _check_unsheared(n)
    return InfiniteTriangularNetwork(rotl60(_rotl60_localsandwich.(unitcell(n))))
end
function rotr60(n::InfiniteTriangularNetwork)
    _check_unsheared(n)
    return InfiniteTriangularNetwork(rotr60(_rotr60_localsandwich.(unitcell(n))))
end
function Base.rot180(n::InfiniteTriangularNetwork)
    _check_unsheared(n)
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

# specialized PFTensorTriangular implementation
function ChainRulesCore.rrule(
        ::typeof(Base.getindex), network::InfiniteTriangularNetwork{<:PFTensorTriangular}, r::Int, c::Int
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

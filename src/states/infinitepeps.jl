# not everything is a PeriodicArray anymore
_next(i, total) = mod1(i + 1, total)
_prev(i, total) = mod1(i - 1, total)

"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePEPS{T<:PEPSTensor} <: AbstractPEPS
    A::Array{T,2} # TODO: switch back to PeriodicArray?

    function InfinitePEPS(A::Array{T,2}) where {T<:PEPSTensor}
        for (d, w) in Tuple.(CartesianIndices(A))
            space(A[d, w], 2) == space(A[_prev(d, end), w], 4)' || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            space(A[d, w], 3) == space(A[d, _next(w, end)], 5)' ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
        end
        return new{T}(A)
    end
end

## Constructors
"""
    InfinitePEPS(A::AbstractArray{T, 2})

Allow users to pass in an array of tensors.
"""
function InfinitePEPS(A::AbstractArray{T,2}) where {T<:PEPSTensor}
    return InfinitePEPS(Array(deepcopy(A)))
end

"""
    InfinitePEPS(Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function InfinitePEPS(
    Pspaces::AbstractArray{S,2},
    Nspaces::AbstractArray{S,2},
    Espaces::AbstractArray{S,2}=Nspaces,
) where {S<:ElementarySpace}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return TensorMap(rand, ComplexF64, P ← N * E * S * W)
    end

    return InfinitePEPS(A)
end

"""
    InfinitePEPS(A)

Allow users to pass in single tensor.
"""
function InfinitePEPS(A::T) where {T<:PEPSTensor}
    As = Array{T,2}(undef, (1, 1))
    As[1, 1] = A
    return InfinitePEPS(As)
end

"""
    InfinitePEPS(Pspace, Nspace, Espace)

Allow users to pass in single space.
"""
function InfinitePEPS(Pspace::S, Nspace::S, Espace::S=Nspace) where {S<:ElementarySpace}
    Pspaces = Array{S,2}(undef, (1, 1))
    Pspaces[1, 1] = Pspace
    Nspaces = Array{S,2}(undef, (1, 1))
    Nspaces[1, 1] = Nspace
    Espaces = Array{S,2}(undef, (1, 1))
    Espaces[1, 1] = Espace
    return InfinitePEPS(Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPS(d, D)

Allow users to pass in integers.
"""
function InfinitePEPS(d::Integer, D::Integer)
    T = TensorMap(rand, ComplexF64, ℂ^d ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')
    return InfinitePEPS(T)
end

"""
    InfinitePEPS(d, D, L)
    InfinitePEPS(d, D, (Lx, Ly)))

Allow users to pass in integers and specify unit cell.
"""
function InfinitePEPS(d::Integer, D::Integer, L::Integer)
    return InfinitePEPS(d, D, (L, L))
end
function InfinitePEPS(d::Integer, D::Integer, Ls::NTuple{2,Integer})
    T = [TensorMap(rand, ComplexF64, ℂ^d ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')]
    return InfinitePEPS(Array(repeat(T, Ls...)))
end

## Shape and size
Base.size(T::InfinitePEPS) = size(T.A)
Base.size(T::InfinitePEPS, i) = size(T.A, i)
Base.length(T::InfinitePEPS) = length(T.A)
Base.eltype(T::InfinitePEPS) = eltype(T.A)
VectorInterface.scalartype(T::InfinitePEPS) = scalartype(T.A)

## Copy
Base.copy(T::InfinitePEPS) = InfinitePEPS(copy(T.A))
Base.similar(T::InfinitePEPS) = InfinitePEPS(similar(T.A))
Base.repeat(T::InfinitePEPS, counts...) = InfinitePEPS(repeat(T.A, counts...))

Base.getindex(T::InfinitePEPS, args...) = Base.getindex(T.A, args...)
Base.setindex!(T::InfinitePEPS, args...) = (Base.setindex!(T.A, args...); T)
Base.axes(T::InfinitePEPS, args...) = axes(T.A, args...)
TensorKit.space(t::InfinitePEPS, i, j) = space(t[i, j], 1)

Base.rotl90(t::InfinitePEPS) = InfinitePEPS(rotl90(rotl90.(t.A)));

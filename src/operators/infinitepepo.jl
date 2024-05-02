"""
    struct InfinitePEPO{T<:PEPOTensor}

Represents an infinte projected entangled-pair operator (PEPO) on a 3D cubic lattice.
"""
struct InfinitePEPO{T<:PEPOTensor} <: AbstractPEPO
    A::Array{T,3}

    function InfinitePEPO(A::Array{T,3}) where {T<:PEPOTensor}
        # space checks
        for (d, w, h) in Tuple.(CartesianIndices(A))
            space(A[d, w, h], 1) == space(A[d, w, _next(h, end)], 2)' ||
                throw(SpaceMismatch("Physical space at site $((d, w, h)) does not match."))
            space(A[d, w, h], 3) == space(A[_prev(d, end), w, h], 5)' || throw(
                SpaceMismatch("North virtual space at site $((d, w, h)) does not match."),
            )
            space(A[d, w, h], 4) == space(A[d, _next(w, end), h], 6)' || throw(
                SpaceMismatch("East virtual space at site $((d, w, h)) does not match.")
            )
        end
        return new{T}(A)
    end
end

## Constructors
"""
    InfinitePEPO(A::AbstractArray{T, 2})

Allow users to pass in an array of tensors.
"""
function InfinitePEPO(A::AbstractArray{T,3}) where {T<:PEPOTensor}
    return InfinitePEPO(Array(deepcopy(A)))
end

"""
    InfinitePEPO(Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function InfinitePEPO(
    Pspaces::AbstractArray{S,3},
    Nspaces::AbstractArray{S,3},
    Espaces::AbstractArray{S,3}=Nspaces,
) where {S<:ElementarySpace}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1, 0)))
    Ppspaces = adjoint.(circshift(Pspaces, (0, 0, -1)))

    A = map(Pspaces, Ppspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, Pp, N, E, S, W
        return TensorMap(rand, ComplexF64, P * Pp ← N * E * S * W)
    end

    return InfinitePEPO(A)
end

"""
    InfinitePEPO(Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces, single layer special case.
"""
function InfinitePEPO(
    Pspaces::AbstractArray{S,2},
    Nspaces::AbstractArray{S,2},
    Espaces::AbstractArray{S,2}=Nspaces,
) where {S<:ElementarySpace}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Pspaces = reshape(Pspaces, (size(Pspaces)..., 1))
    Nspaces = reshape(Pspaces, (size(Nspaces)..., 1))
    Espaces = reshape(Pspaces, (size(Espaces)..., 1))

    return InfinitePEPO(Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPO(A)

Allow users to pass in single tensor.
"""
function InfinitePEPO(A::T) where {T<:PEPOTensor}
    As = Array{T,3}(undef, (1, 1, 1))
    As[1, 1, 1] = A
    return InfinitePEPO(As)
end

"""
    InfinitePEPO(Pspace, Nspace, Espace)

Allow users to pass in single space.
"""
function InfinitePEPO(Pspace::S, Nspace::S, Espace::S=Nspace) where {S<:ElementarySpace}
    Pspaces = Array{S,3}(undef, (1, 1, 1))
    Pspaces[1, 1] = Pspace
    Nspaces = Array{S,3}(undef, (1, 1, 1))
    Nspaces[1, 1] = Nspace
    Espaces = Array{S,3}(undef, (1, 1, 1))
    Espaces[1, 1] = Espace
    return InfinitePEPO(Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPO(d, D)

Allow users to pass in integers.
"""
function InfinitePEPO(d::Integer, D::Integer)
    T = TensorMap(rand, ComplexF64, ℂ^d ⊗ (ℂ^d)' ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')
    return InfinitePEPO(T)
end

"""
    InfinitePEPO(d, D, L)
    InfinitePEPO(d, D, (Lx, Ly, Lz)))

Allow users to pass in integers and specify unit cell.
"""
function InfinitePEPO(d::Integer, D::Integer, L::Integer)
    return InfinitePEPO(d, D, (L, L, L))
end
function InfinitePEPO(d::Integer, D::Integer, Ls::NTuple{3,Integer})
    T = [TensorMap(rand, ComplexF64, ℂ^d ⊗ (ℂ^d)' ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')]
    return InfinitePEPO(Array(repeat(T, Ls...)))
end

## Shape and size
Base.size(T::InfinitePEPO) = size(T.A)
Base.size(T::InfinitePEPO, i) = size(T.A, i)
Base.length(T::InfinitePEPO) = length(T.A)
Base.eltype(T::InfinitePEPO) = eltype(T.A)
VectorInterface.scalartype(T::InfinitePEPO) = scalartype(T.A)

## Copy
Base.copy(T::InfinitePEPO) = InfinitePEPO(copy(T.A))
Base.similar(T::InfinitePEPO) = InfinitePEPO(similar(T.A))
Base.repeat(T::InfinitePEPO, counts...) = InfinitePEPO(repeat(T.A, counts...))

Base.getindex(T::InfinitePEPO, args...) = Base.getindex(T.A, args...)
Base.axes(T::InfinitePEPO, args...) = axes(T.A, args...)
TensorKit.space(T::InfinitePEPO, i, j) = space(T[i, j, end], 1)

Base.rotl90(T::InfinitePEPO) = InfinitePEPO(rotl90(rotl90.(T.A)));

function initializePEPS(
    T::InfinitePEPO{<:PEPOTensor{S}}, vspace::S
) where {S<:ElementarySpace}
    Pspaces = Array{S,2}(undef, size(T, 1), size(T, 2))
    for (i, j) in product(1:size(T, 1), 1:size(T, 2))
        Pspaces[i, j] = space(T, i, j)
    end
    Nspaces = repeat([vspace], size(T, 1), size(T, 2))
    Espaces = repeat([vspace], size(T, 1), size(T, 2))
    return InfinitePEPS(Pspaces, Nspaces, Espaces)
end

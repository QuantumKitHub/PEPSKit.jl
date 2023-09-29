"""
    struct PeriodicPEPO{T<:PEPOTensor}

Represents an periodic PEPO on a 2D square lattice.
"""
struct PeriodicPEPO{T<:PEPOTensor} <: AbstractPEPO
    A::PeriodicArray{T,2}

    function PeriodicPEPO(A::PeriodicArray{T,2}) where {T<:PEPOTensor}
        Ivertical = CartesianIndex(-1, 0)
        Ihorizontal = CartesianIndex(0, 1)
        for I in CartesianIndices(A)
            space(A[I], 3) == space(A[I + Ivertical], 5)' || throw(
                SpaceMismatch("North virtual space at site $(Tuple(I)) does not match.")
            )
            space(A[I], 4) == space(A[I + Ihorizontal], 6)' || throw(
                SpaceMismatch("East virtual space at site $(Tuple(I)) does not match.")
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
function PeriodicPEPO(A::AbstractArray{T,2}) where {T<:PEPOTensor}
    return PeriodicPEPO(PeriodicArray(deepcopy(A)))
end

"""
    InfinitePEPO(Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function PeriodicPEPO(
    Pspaces::AbstractArray{S,2},
    Nspaces::AbstractArray{S,2},
    Espaces::AbstractArray{S,2}=Nspaces,
) where {S<:ElementarySpace}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1)))
    Ppspaces = adjoint.(Pspaces)

    A = map(Pspaces, Ppspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, Pp, N, E, S, W
        return TensorMap(rand, ComplexF64, P * Pp ← N * E * S * W)
    end

    return PeriodicPEPO(A)
end

"""
    PeriodicPEPO(Pspace, Nspace, Espace)

Allow users to pass in single space.
"""
function PeriodicPEPO(Pspace::S, Nspace::S, Espace::S=Nspace) where {S<:ElementarySpace}
    Pspaces = Array{S,2}(undef, (1, 1))
    Pspaces[1, 1] = Pspace
    Nspaces = Array{S,2}(undef, (1, 1))
    Nspaces[1, 1] = Nspace
    Espaces = Array{S,2}(undef, (1, 1))
    Espaces[1, 1] = Espace
    return PeriodicPEPO(Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPO(d, D)

Allow users to pass in integers.
"""
function PeriodicPEPO(d::Integer, D::Integer)
    T = [TensorMap(rand, ComplexF64, ℂ^d ⊗ ℂ^d ← ℂ^D ⊗ ℂ^D ⊗ (ℂ^D)' ⊗ (ℂ^D)')]
    return PeriodicPEPO(PeriodicArray(reshape(T, (1, 1))))
end

PeriodicPEPO(d::Integer, D::Integer, Ls::Tuple{Integer}) = repeat(PeriodicPEPO(d, D), Ls...)

## Shape and size
Base.size(T::PeriodicPEPO) = size(T.A)
Base.size(T::PeriodicPEPO, i) = size(T.A, i)
Base.length(T::PeriodicPEPO) = length(T.A)

## Copy
Base.copy(T::PeriodicPEPO) = PeriodicPEPO(copy(T.A))
Base.similar(T::PeriodicPEPO) = PeriodicPEPO(similar(T.A))
Base.repeat(T::PeriodicPEPO, counts...) = PeriodicPEPO(repeat(T.A, counts...))

Base.getindex(T::PeriodicPEPO, args...) = getindex(T.A, args...);
TensorKit.space(t::PeriodicPEPO, i, j) = space(t[i, j], 1)

Base.rotl90(t::PeriodicPEPO) = PeriodicPEPO(rotl90(rotl90.(t.A)));

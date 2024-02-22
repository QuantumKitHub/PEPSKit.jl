"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePEPS{T<:PEPSTensor} <: AbstractPEPS
    A::Matrix{T}

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
    return InfinitePEPS(Array(deepcopy(A))) # TODO: find better way to copy
end

"""
    InfinitePEPS(f=randn, T=ComplexF64, Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function InfinitePEPS(
    Pspaces::A, Nspaces::A, Espaces::A
) where {A<:AbstractArray{<:ElementarySpace,2}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractArray{<:ElementarySpace,2}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1)))

    A = map(Pspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, N, E, S, W
        return PEPSTensor(f, T, P, N, E, S, W)
    end

    return InfinitePEPS(A)
end

"""
    InfinitePEPS(A; unitcell=(1, 1))

Create an `InfinitePEPS` by specifying a tensor and unit cell.
"""
function InfinitePEPS(A::T; unitcell::Tuple{Int,Int}=(1, 1)) where {T<:PEPSTensor}
    return InfinitePEPS(fill(A, unitcell))
end

"""
    InfinitePEPS(Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePEPS by specifying its spaces and unit cell. Spaces can be specified
either via `Int` or via `ElementarySpace`.
"""
function InfinitePEPS(
    Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePEPS(
        fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

## Shape and size
Base.size(T::InfinitePEPS) = size(T.A)
Base.size(T::InfinitePEPS, i) = size(T.A, i)
Base.length(T::InfinitePEPS) = length(T.A)
Base.eltype(T::InfinitePEPS) = eltype(T.A)
VectorInterface.scalartype(T::InfinitePEPS) = scalartype(T.A)

## Copy
Base.copy(T::InfinitePEPS) = InfinitePEPS(copy(T.A))
Base.similar(T::InfinitePEPS, args...) = InfinitePEPS(similar(T.A, args...))
Base.repeat(T::InfinitePEPS, counts...) = InfinitePEPS(repeat(T.A, counts...))

Base.getindex(T::InfinitePEPS, args...) = Base.getindex(T.A, args...)
Base.setindex!(T::InfinitePEPS, args...) = (Base.setindex!(T.A, args...); T)
Base.axes(T::InfinitePEPS, args...) = axes(T.A, args...)
TensorKit.space(t::InfinitePEPS, i, j) = space(t[i, j], 1)

Base.rotl90(t::InfinitePEPS) = InfinitePEPS(rotl90(rotl90.(t.A)))

## Math
Base.:+(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A + ψ₂.A)
Base.:-(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A - ψ₂.A)
LinearAlgebra.norm(ψ::InfinitePEPS) = norm(ψ.A)
LinearAlgebra.dot(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = dot(ψ₁.A, ψ₂.A)

# For _scale! during optimization
function LinearAlgebra.rmul!(ψ::InfinitePEPS, β::Number)
    rmul!(ψ.A, β)
    return ψ
end

# For _add! during optimization
function LinearAlgebra.axpy!(β::Number, ψ::InfinitePEPS, η::InfinitePEPS)
    ψ.A .+= η.A .* β
    return ψ
end
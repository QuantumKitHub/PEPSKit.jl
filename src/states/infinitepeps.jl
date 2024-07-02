"""
    struct InfinitePEPS{T<:PEPSTensor}

Represents an infinite projected entangled-pair state on a 2D square lattice.
"""
struct InfinitePEPS{T<:PEPSTensor} <: AbstractPEPS
    A::Matrix{T}
    InfinitePEPS{T}(A::Matrix{T}) where {T<:PEPSTensor} = new{T}(A)
    function InfinitePEPS(A::Array{T,2}) where {T<:PEPSTensor}
        for (d, w) in Tuple.(CartesianIndices(A))
            space(A[d, w], 2) == space(A[_prev(d, end), w], 4)' || throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match.")
            )
            space(A[d, w], 3) == space(A[d, _next(w, end)], 5)' ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match."))
            dim(space(A[d, w])) > 0 || @warn "no fusion channels at site ($d, $w)"
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
) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    return InfinitePEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
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
    InfinitePEPS(f=randn, T=ComplexF64, Pspace, Nspace, [Espace]; unitcell=(1,1))

Create an InfinitePEPS by specifying its spaces and unit cell. Spaces can be specified
either via `Int` or via `ElementarySpace`.
"""
function InfinitePEPS(
    Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePEPS(
        randn,
        ComplexF64,
        fill(Pspace, unitcell),
        fill(Nspace, unitcell),
        fill(Espace, unitcell),
    )
end
function InfinitePEPS(
    f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:Union{ElementarySpace,Int}}
    return InfinitePEPS(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
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
# Base.similar(T::InfinitePEPS) = InfinitePEPS(similar(T.A))  # TODO: This is incompatible with inner constructor
Base.repeat(T::InfinitePEPS, counts...) = InfinitePEPS(repeat(T.A, counts...))

Base.getindex(T::InfinitePEPS, args...) = Base.getindex(T.A, args...)
Base.setindex!(T::InfinitePEPS, args...) = (Base.setindex!(T.A, args...); T)
Base.axes(T::InfinitePEPS, args...) = axes(T.A, args...)
TensorKit.space(t::InfinitePEPS, i, j) = space(t[i, j], 1)

## Math
Base.:+(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A + ψ₂.A)
Base.:-(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = InfinitePEPS(ψ₁.A - ψ₂.A)
Base.:*(α::Number, ψ::InfinitePEPS) = InfinitePEPS(α * ψ.A)
LinearAlgebra.dot(ψ₁::InfinitePEPS, ψ₂::InfinitePEPS) = dot(ψ₁.A, ψ₂.A)
LinearAlgebra.norm(ψ::InfinitePEPS) = norm(ψ.A)

# Used in _scale during OptimKit.optimize
function LinearAlgebra.rmul!(ψ::InfinitePEPS, α::Number)
    rmul!(ψ.A, α)
    return ψ
end

# Used in _add during OptimKit.optimize
function LinearAlgebra.axpy!(α::Number, ψ₁::InfinitePEPS, ψ₂::InfinitePEPS)
    ψ₂.A .+= α * ψ₁.A
    return ψ₂
end

# VectorInterface
VectorInterface.zerovector(x::InfinitePEPS) = InfinitePEPS(zerovector(x.A))

# Rotations
Base.rotl90(t::InfinitePEPS) = InfinitePEPS(rotl90(rotl90.(t.A)))
Base.rotr90(t::InfinitePEPS) = InfinitePEPS(rotr90(rotr90.(t.A)))
Base.rot180(t::InfinitePEPS) = InfinitePEPS(rot180(rot180.(t.A)))

# Chain rules
function ChainRulesCore.rrule(
    ::typeof(Base.getindex), state::InfinitePEPS, row::Int, col::Int
)
    pepstensor = state[row, col]

    function getindex_pullback(Δpepstensor)
        Δstate = zerovector(state)
        Δstate[row, col] = Δpepstensor
        return NoTangent(), Δstate, NoTangent(), NoTangent()
    end
    return pepstensor, getindex_pullback
end

function ChainRulesCore.rrule(::Type{<:InfinitePEPS}, A::Matrix{T}) where {T<:PEPSTensor}
    peps = InfinitePEPS(A)
    function InfinitePEPS_pullback(Δpeps)
        return NoTangent(), Δpeps.A
    end
    return peps, InfinitePEPS_pullback
end

function ChainRulesCore.rrule(::typeof(rotl90), peps::InfinitePEPS)
    peps′ = rotl90(peps)
    function rotl90_pullback(Δpeps)
        return NoTangent(), rotr90(Δpeps)
    end
    return peps′, rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), peps::InfinitePEPS)
    peps′ = rotr90(peps)
    function rotr90_pullback(Δpeps)
        return NoTangent(), rotl90(Δpeps)
    end
    return peps′, rotr90_pullback
end

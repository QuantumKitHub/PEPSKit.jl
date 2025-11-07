"""
    struct InfinitePEPO{T<:PEPOTensor}

Represents an infinite projected entangled-pair operator (PEPO) on a 3D cubic lattice.

## Fields

$(TYPEDFIELDS)
"""
struct InfinitePEPO{T <: PEPOTensor}
    A::Array{T, 3}
    InfinitePEPO{T}(A::Array{T, 3}) where {T} = new{T}(A)
    function InfinitePEPO(A::Array{T, 3}) where {T <: PEPOTensor}
        # space checks
        bosonic_braiding = BraidingStyle(sectortype(T)) === Bosonic()
        for (d, w, h) in Tuple.(CartesianIndices(A))
            (bosonic_braiding || (!isdual(domain_physicalspace(A[d, w, h])) && !isdual(codomain_physicalspace(A[d, w, h])))) ||
                throw(ArgumentError("Dual physical spaces for symmetry sectors with non-trivial twists are not allowed (for now)."))
            codomain_physicalspace(A[d, w, h]) == domain_physicalspace(A[d, w, _next(h, end)]) ||
                throw(SpaceMismatch("Physical space at site $((d, w, h)) does not match."))
            north_virtualspace(A[d, w, h]) == south_virtualspace(A[_prev(d, end), w, h])' ||
                throw(SpaceMismatch("North virtual space at site $((d, w, h)) does not match."))
            east_virtualspace(A[d, w, h]) == west_virtualspace(A[d, _next(w, end), h])' ||
                throw(SpaceMismatch("East virtual space at site $((d, w, h)) does not match."))
        end
        return new{T}(A)
    end
end

## Constructors

"""
    InfinitePEPO(A::AbstractArray{T, 3})

Allow users to pass in an array of tensors.
"""
function InfinitePEPO(A::AbstractArray{T, 3}) where {T <: PEPOTensor}
    return InfinitePEPO(Array(deepcopy(A)))
end

"""
    InfinitePEPO([f=randn, T=ComplexF64,] Pspaces, Nspaces, Espaces)

Allow users to pass in arrays of spaces.
"""
function InfinitePEPO(
        Pspaces::A, Nspaces::A, Espaces::A = Nspaces
    ) where {A <: AbstractArray{<:ElementarySpace, 3}}
    return InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPO(
        f, T, Pspaces::A, Nspaces::A, Espaces::A = Nspaces
    ) where {A <: AbstractArray{<:ElementarySpace, 3}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    Sspaces = adjoint.(circshift(Nspaces, (1, 0, 0)))
    Wspaces = adjoint.(circshift(Espaces, (0, -1, 0)))
    Ppspaces = adjoint.(circshift(Pspaces, (0, 0, -1)))

    P = map(Pspaces, Ppspaces, Nspaces, Espaces, Sspaces, Wspaces) do P, Pp, N, E, S, W
        return f(T, P * Pp ← N * E * S * W)
    end

    return InfinitePEPO(P)
end

function InfinitePEPO(
        Pspaces::A, Nspaces::A, Espaces::A = Nspaces
    ) where {A <: AbstractArray{<:ElementarySpace, 2}}
    return InfinitePEPO(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfinitePEPO(
        f, T, Pspaces::A, Nspaces::A, Espaces::A = Nspaces
    ) where {A <: AbstractArray{<:ElementarySpace, 2}}
    size(Pspaces) == size(Nspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))
    Pspaces = reshape(Pspaces, (size(Pspaces)..., 1))
    Nspaces = reshape(Pspaces, (size(Nspaces)..., 1))
    Espaces = reshape(Pspaces, (size(Espaces)..., 1))
    return InfinitePEPO(f, T, Pspaces, Nspaces, Espaces)
end

"""
    InfinitePEPO(A::PEPOTensor; unitcell=(1, 1, 1))

Create an InfinitePEPO by specifying a tensor which is repeated across the unit cell.
"""
function InfinitePEPO(A::T; unitcell::Tuple{Int, Int, Int} = (1, 1, 1)) where {T <: PEPOTensor}
    return InfinitePEPO(fill(A, unitcell))
end

"""
    InfinitePEPO([f=randn, T=ComplexF64,] Pspace::S, Nspace::S, [Espace::S]; unitcell=(1,1,1)) where {S<:ElementarySpace}

Create an InfinitePEPO by specifying its spaces and unit cell.
"""
function InfinitePEPO(
        Pspace::S, Nspace::S, Espace::S = Nspace; unitcell::Tuple{Int, Int, Int} = (1, 1, 1)
    ) where {S <: ElementarySpace}
    return InfinitePEPO(
        randn, ComplexF64,
        fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell),
    )
end
function InfinitePEPO(
        f, T, Pspace::S, Nspace::S, Espace::S = Nspace; unitcell::Tuple{Int, Int, Int} = (1, 1, 1)
    ) where {S <: ElementarySpace}
    return InfinitePEPO(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

function initializePEPS(
        T::InfinitePEPO{<:PEPOTensor{S}}, vspace::S
    ) where {S <: ElementarySpace}
    Pspaces = map(Iterators.product(axes(T, 1), axes(T, 2))) do (r, c)
        return domain_physicalspace(T, r, c)
    end
    Nspaces = repeat([vspace], size(T, 1), size(T, 2))
    Espaces = repeat([vspace], size(T, 1), size(T, 2))
    return InfinitePEPS(Pspaces, Nspaces, Espaces)
end

## Unit cell interface

unitcell(t::InfinitePEPO) = t.A
Base.size(A::InfinitePEPO, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePEPO) = length(unitcell(A))
Base.eltype(::Type{InfinitePEPO{T}}) where {T} = T
Base.eltype(A::InfinitePEPO) = eltype(typeof(A))

Base.copy(A::InfinitePEPO) = InfinitePEPO(copy(unitcell(A)))
function Base.similar(A::InfinitePEPO, T::Type{TorA} = scalartype(A)) where {TorA}
    return InfinitePEPO(map(t -> similar(t, T), unitcell(A)))
end
Base.repeat(A::InfinitePEPO, counts...) = InfinitePEPO(repeat(unitcell(A), counts...))

Base.getindex(A::InfinitePEPO, args...) = Base.getindex(unitcell(A), args...)
Base.setindex!(A::InfinitePEPO, args...) = (Base.setindex!(unitcell(A), args...); A)
Base.axes(A::InfinitePEPO, args...) = axes(unitcell(A), args...)
eachcoordinate(A::InfinitePEPO) = collect(Iterators.product(axes(A)...))
function eachcoordinate(A::InfinitePEPO, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces

TensorKit.spacetype(::Type{P}) where {P <: InfinitePEPO} = spacetype(eltype(P))
function virtualspace(T::InfinitePEPO, r::Int, c::Int, h::Int, dir)
    Nr, Nc, Nh = size(T)
    return virtualspace(T[mod1(r, Nr), mod1(c, Nc), mod1(h, Nh)], dir)
end
function domain_physicalspace(T::InfinitePEPO, r::Int, c::Int)
    Nr, Nc, = size(T)
    return domain_physicalspace(T[mod1(r, Nr), mod1(c, Nc), 1])
end
function codomain_physicalspace(T::InfinitePEPO, r::Int, c::Int)
    Nr, Nc, = size(T)
    return codomain_physicalspace(T[mod1(r, Nr), mod1(c, Nc), end])
end
physicalspace(T::InfinitePEPO) = [physicalspace(T, row, col) for row in axes(T, 1), col in axes(T, 2)]
function physicalspace(T::InfinitePEPO, r::Int, c::Int)
    codomain_physicalspace(T, r, c) == domain_physicalspace(T, r, c) || throw(
        SpaceMismatch(
            "Domain and codomain physical spaces at site $((r, c)) do not match."
        ),
    )
    return codomain_physicalspace(T, r, c)
end

## InfiniteSquareNetwork interface

function InfiniteSquareNetwork(top::InfinitePEPS, mid::InfinitePEPO, bot::InfinitePEPS = top)
    size(top) == size(bot) == size(mid)[1:2] || throw(
        ArgumentError("Top PEPS, bottom PEPS and PEPO layers should have equal sizes")
    )
    return InfiniteSquareNetwork(
        map(tuple, unitcell(top), unitcell(bot), eachslice(unitcell(mid); dims = 3)...)
    )
end

## Conversion to states

function InfinitePartitionFunction(ρ::InfinitePEPO)
    size(ρ, 3) == 1 || throw(DimensionMismatch("Only single-layer density matrices can be converted to partition functions"))
    return InfinitePartitionFunction(
        trace_physicalspaces.(reshape(unitcell(ρ), size(ρ, 1), size(ρ, 2)))
    )
end

function InfinitePEPS(ρ::InfinitePEPO)
    size(ρ, 3) == 1 || throw(DimensionMismatch("Only single-layer density matrices can be converted to partition functions"))
    return InfinitePEPS(
        first.(fuse_physicalspaces.(reshape(unitcell(ρ), size(ρ, 1), size(ρ, 2))))
    )
end


## Vector interface

VI.scalartype(::Type{NT}) where {NT <: InfinitePEPO} = scalartype(eltype(NT))
VI.zerovector(A::InfinitePEPO) = InfinitePEPO(zerovector(unitcell(A)))

## (Approximate) equality
function Base.:(==)(A₁::InfinitePEPO, A₂::InfinitePEPO)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfinitePEPO, A₂::InfinitePEPO; kwargs...)
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end

## Rotations

Base.rotl90(A::InfinitePEPO) =
    InfinitePEPO(stack(rotl90, eachslice(rotl90.(unitcell(A)); dims = 3)))
Base.rotr90(A::InfinitePEPO) =
    InfinitePEPO(stack(rotr90, eachslice(rotr90.(unitcell(A)); dims = 3)))
Base.rot180(A::InfinitePEPO) =
    InfinitePEPO(stack(rot180, eachslice(rot180.(unitcell(A)); dims = 3)))

## Chainrules

function ChainRulesCore.rrule(
        ::Type{InfiniteSquareNetwork},
        top::InfinitePEPS, mid::InfinitePEPO{P}, bot::InfinitePEPS,
    ) where {P <: PEPOTensor}
    network = InfiniteSquareNetwork(top, mid, bot)

    function InfiniteSquareNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δtop = InfinitePEPS(map(ket, unitcell(Δnetwork)))
        Δbot = InfinitePEPS(map(bra, unitcell(Δnetwork)))
        Δmid = InfinitePEPO(_stack_tuples(map(pepo, unitcell(Δnetwork))))
        return NoTangent(), Δtop, Δmid, Δbot
    end
    return network, InfiniteSquareNetwork_pullback
end

function _stack_tuples(A::Matrix{NTuple{N, T}}) where {N, T}
    out = Array{T}(undef, size(A)..., N)
    for (r, c) in Iterators.product(axes(A)...)
        out[r, c, :] .= A[r, c]
    end
    return out
end

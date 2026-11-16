"""
    struct InfinitePEPSTriangular{T<:PEPSTensorTriangular}

Represents an infinite projected entangled-pair state on a 2D triangular lattice.

The tensors are stored in an `Nrows × Ncols` unit cell that is periodic under the
translations `(0, Ncols)` and `(Nrows, shift)`, see [`TriangularUnitCell`](@ref). The
three-sublattice (√3 × √3) state `[A B C; C A B; B C A]` is stored with only three tensors
as `unitcell = (1, 3)` and `shift = 1`.

## Fields

$(TYPEDFIELDS)
"""
struct InfinitePEPSTriangular{T <: PEPSTensorTriangular}
    A::Matrix{T}
    shift::Int
    function InfinitePEPSTriangular{T}(A::Matrix{T}, shift::Integer = 0) where {T <: PEPSTensorTriangular}
        return new{T}(A, TriangularUnitCell(size(A), shift).shift)
    end
    function InfinitePEPSTriangular(A::Array{T, 2}; shift::Integer = 0) where {T <: PEPSTensorTriangular}
        cell = TriangularUnitCell(size(A), shift)
        site(r, c) = A[_site(cell, r, c)...]
        bosonic_braiding = BraidingStyle(sectortype(T)) === Bosonic()
        for (r, c) in Tuple.(CartesianIndices(A))
            (bosonic_braiding || !isdual(physicalspace(A[r, c]))) ||
                throw(ArgumentError("Dual physical spaces for symmetry sectors with non-trivial twists are not allowed (for now)."))
            northwest_virtualspace(A[r, c]) == southeast_virtualspace(site(r - 1, c))' ||
                throw(SpaceMismatch("Northwest virtual space at site $((r, c)) does not match."))
            northeast_virtualspace(A[r, c]) == southwest_virtualspace(site(r - 1, c + 1))' ||
                throw(SpaceMismatch("Northeast virtual space at site $((r, c)) does not match."))
            east_virtualspace(A[r, c]) == west_virtualspace(site(r, c + 1))' ||
                throw(SpaceMismatch("East virtual space at site $((r, c)) does not match."))
            dim(space(A[r, c])) > 0 || @warn "no fusion channels at site ($r, $c)"
        end
        return new{T}(A, cell.shift)
    end
end

## Constructors

"""
    InfinitePEPSTriangular(A::AbstractMatrix{T}; shift=0)

Create an `InfinitePEPSTriangular` by specifying a matrix containing the PEPS tensors at each site in
the unit cell, see [`TriangularUnitCell`](@ref) for the meaning of `shift`.
"""
function InfinitePEPSTriangular(A::AbstractMatrix{<:PEPSTensorTriangular}; kwargs...)
    return InfinitePEPSTriangular(Array(deepcopy(A)); kwargs...) # TODO: find better way to copy
end

"""
    InfinitePEPSTriangular([f=randn, T=ComplexF64,] Pspaces::A, NWspaces::A, [NEspaces::A], [Espaces::A]; shift=0) where {A<:AbstractMatrix{ElementarySpace}}

Create an `InfinitePEPSTriangular` by specifying matrices of physical, northwest, northeast
and east virtual spaces. Omitted virtual spaces default to `NWspaces`; the opposite
spaces are the duals of the corresponding neighboring spaces, with periodic boundaries
as specified by `shift` (see [`TriangularUnitCell`](@ref)).
"""
function InfinitePEPSTriangular(
        f, T::Type{<:Number}, Pspaces::M, NWspaces::M, NEspaces::M = NWspaces, Espaces::M = NWspaces;
        shift::Integer = 0,
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
    size(Pspaces) == size(NWspaces) == size(NEspaces) == size(Espaces) ||
        throw(ArgumentError("Input spaces should have equal sizes."))

    cell = TriangularUnitCell(size(Pspaces), shift)
    Nr, Nc = size(cell)
    SEspaces = [NWspaces[_site(cell, r + 1, c)...]' for r in 1:Nr, c in 1:Nc]
    SWspaces = [NEspaces[_site(cell, r + 1, c - 1)...]' for r in 1:Nr, c in 1:Nc]
    Wspaces = [Espaces[_site(cell, r, c - 1)...]' for r in 1:Nr, c in 1:Nc]

    A = map(Pspaces, NWspaces, NEspaces, Espaces, SEspaces, SWspaces, Wspaces) do P, NW, NE, E, SE, SW, W
        return PEPSTensorTriangular(f, T, P, NW, NE, E, SE, SW, W)
    end

    return InfinitePEPSTriangular(A; shift)
end
function InfinitePEPSTriangular(
        Pspaces::A, virtual_spaces...; kwargs...
    ) where {A <: Union{AbstractMatrix{<:ElementarySpace}, ElementarySpace}}
    return InfinitePEPSTriangular(randn, ComplexF64, Pspaces, virtual_spaces...; kwargs...)
end

"""
    InfinitePEPSTriangular(A::PEPSTensorTriangular; unitcell=(1, 1), shift=0)

Create an `InfinitePEPSTriangular` by specifying a tensor and unit cell.

The unit cell uses one-based matrix coordinates `[r, c]`. The virtual legs are ordered
`NW, NE, E, SE, SW, W`, connecting to neighbors at offsets
`(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)`, respectively.
Contractions wrap these coordinates periodically; direct `peps[r, c]` indexing follows
ordinary matrix bounds. See [`TriangularUnitCell`](@ref) for the meaning of `shift`.
"""
function InfinitePEPSTriangular(A::T; unitcell::Tuple{Int, Int} = (1, 1), shift::Integer = 0) where {T <: PEPSTensorTriangular}
    return InfinitePEPSTriangular(fill(A, unitcell); shift)
end

# expand PEPS spaces to unit cell size
function _fill_state_virtual_spaces_triangular(
        NWspace::S, NEspace::S = NWspace, Espace::S = NWspace; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return fill(NWspace, unitcell), fill(NEspace, unitcell), fill(Espace, unitcell)
end

"""
    InfinitePEPSTriangular([f=randn, T=ComplexF64,] Pspace, NWspace, [NEspace], [Espace]; unitcell=(1,1), shift=0)

Create an `InfinitePEPSTriangular` with uniform physical, northwest, northeast and east
virtual spaces and the given unit-cell size. Omitted virtual spaces default to `NWspace`.
See [`TriangularUnitCell`](@ref) for the meaning of `shift`; for instance,
`unitcell = (1, 3), shift = 1` gives a three-sublattice state with three independent tensors.
"""
function InfinitePEPSTriangular(
        f, T::Type{<:Number}, Pspace::S, vspaces...; unitcell::Tuple{Int, Int} = (1, 1), shift::Integer = 0
    ) where {S <: ElementarySpace}
    return InfinitePEPSTriangular(
        f, T,
        _fill_state_physical_spaces(Pspace; unitcell),
        _fill_state_virtual_spaces_triangular(vspaces...; unitcell)...;
        shift,
    )
end

## Unit cell interface

unitcell(t::InfinitePEPSTriangular) = t.A
_cell(t::InfinitePEPSTriangular) = TriangularUnitCell(size(t.A), t.shift)
@non_differentiable _cell(t::InfinitePEPSTriangular)

# wrap new unit-cell tensors with the same geometry as `A`
_rewrap(A::InfinitePEPSTriangular, tensors::AbstractMatrix) = InfinitePEPSTriangular(tensors; shift = A.shift)
function _check_cells(A₁, A₂)
    _cell(A₁) == _cell(A₂) || throw(ArgumentError("unit cells do not match"))
    return nothing
end
@non_differentiable _check_cells(args...)
Base.size(A::InfinitePEPSTriangular, args...) = size(unitcell(A), args...)
Base.length(A::InfinitePEPSTriangular) = length(unitcell(A))
Base.eltype(::Type{InfinitePEPSTriangular{T}}) where {T} = T
Base.eltype(A::InfinitePEPSTriangular) = eltype(typeof(A))

Base.copy(A::InfinitePEPSTriangular) = _rewrap(A, copy(unitcell(A)))
function Base.similar(A::InfinitePEPSTriangular, T::Type{TorA} = scalartype(A)) where {TorA}
    return _rewrap(A, map(t -> similar(t, T), unitcell(A)))
end
function Base.repeat(A::InfinitePEPSTriangular, counts...)
    iszero(A.shift) || throw(ArgumentError("repeating a sheared unit cell is not supported"))
    return InfinitePEPSTriangular(repeat(unitcell(A), counts...))
end

Base.getindex(A::InfinitePEPSTriangular, args...) = Base.getindex(unitcell(A), args...)
Base.setindex!(A::InfinitePEPSTriangular, args...) = (Base.setindex!(unitcell(A), args...); A)
Base.axes(A::InfinitePEPSTriangular, args...) = axes(unitcell(A), args...)
eachcoordinate(A::InfinitePEPSTriangular) = collect(Iterators.product(axes(A)...))
function eachcoordinate(A::InfinitePEPSTriangular, dirs)
    return collect(Iterators.product(dirs, axes(A, 1), axes(A, 2)))
end

## Spaces

TensorKit.spacetype(::Type{T}) where {T <: InfinitePEPSTriangular} = spacetype(eltype(T))
virtualspace(n::InfinitePEPSTriangular, dir) = virtualspace.(unitcell(n), dir)
function virtualspace(n::InfinitePEPSTriangular, r::Int, c::Int, dir)
    return virtualspace(n[_site(_cell(n), r, c)...], dir)
end
physicalspace(n::InfinitePEPSTriangular) = physicalspace.(unitcell(n))
function physicalspace(n::InfinitePEPSTriangular, r::Int, c::Int)
    return physicalspace(n[_site(_cell(n), r, c)...])
end

## InfiniteTriangularNetwork interface

function InfiniteTriangularNetwork(top::InfinitePEPSTriangular, bot::InfinitePEPSTriangular = top)
    _cell(top) == _cell(bot) || throw(
        ArgumentError("Top and bottom PEPS should have the same unit cell")
    )
    return InfiniteTriangularNetwork(map(tuple, unitcell(top), unitcell(bot)); shift = top.shift)
end

## Vector interface

VI.scalartype(::Type{NT}) where {NT <: InfinitePEPSTriangular} = scalartype(eltype(NT))
VI.zerovector(A::InfinitePEPSTriangular) = _rewrap(A, zerovector(unitcell(A)))

function VI.scale(ψ::InfinitePEPSTriangular, α::Number)
    _scale = Base.Fix2(scale, α)
    return _rewrap(ψ, map(_scale, unitcell(ψ)))
end
function VI.scale!(ψ::InfinitePEPSTriangular, α::Number)
    _scale! = Base.Fix2(scale!, α)
    foreach(_scale!, unitcell(ψ))
    return ψ
end
function VI.scale!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    foreach(_scale!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.scale!!(ψ::InfinitePEPSTriangular, α::Number) = scale!(ψ, α)
VI.scale!!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number) = scale!(ψ₁, ψ₂, α)

function VI.add(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number)
    _check_cells(ψ₁, ψ₂)
    _add(x, y) = add(x, y, α, β)
    return _rewrap(ψ₁, map(_add, unitcell(ψ₁), unitcell(ψ₂)))
end
function VI.add!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, unitcell(ψ₁), unitcell(ψ₂))
    return ψ₁
end
VI.add!!(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular, α::Number, β::Number) = add!(ψ₁, ψ₂, α, β)
VI.inner(ψ₁::InfinitePEPSTriangular, ψ₂::InfinitePEPSTriangular) = inner(unitcell(ψ₁), unitcell(ψ₂))

## Math

function Base.:+(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    _check_cells(A₁, A₂)
    return _rewrap(A₁, unitcell(A₁) + unitcell(A₂))
end
function Base.:-(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    _check_cells(A₁, A₂)
    return _rewrap(A₁, unitcell(A₁) - unitcell(A₂))
end
Base.:*(α::Number, A::InfinitePEPSTriangular) = _rewrap(A, α * unitcell(A))
Base.:*(A::InfinitePEPSTriangular, α::Number) = α * A
Base.:/(A::InfinitePEPSTriangular, α::Number) = _rewrap(A, unitcell(A) / α)
LinearAlgebra.dot(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular) = inner(A₁, A₂)
LinearAlgebra.norm(A::InfinitePEPSTriangular) = norm(unitcell(A))

## (Approximate) equality
function Base.:(==)(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular)
    _cell(A₁) == _cell(A₂) || return false
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return p₁ == p₂
    end
end
function Base.isapprox(A₁::InfinitePEPSTriangular, A₂::InfinitePEPSTriangular; kwargs...)
    _cell(A₁) == _cell(A₂) || return false
    return all(zip(unitcell(A₁), unitcell(A₂))) do (p₁, p₂)
        return isapprox(p₁, p₂; kwargs...)
    end
end


## Chainrules

function ChainRulesCore.rrule(::typeof(Base.getindex), network::InfinitePEPSTriangular, args...)
    tensor = network[args...]

    function getindex_pullback(Δtensor_)
        Δtensor = unthunk(Δtensor_)
        Δnetwork = zerovector(network)
        Δnetwork[args...] = Δtensor
        return NoTangent(), Δnetwork, NoTangent(), NoTangent()
    end
    return tensor, getindex_pullback
end

function ChainRulesCore.rrule(
        ::Type{InfiniteTriangularNetwork}, top::InfinitePEPSTriangular, bot::InfinitePEPSTriangular
    )
    network = InfiniteTriangularNetwork(top, bot)

    function InfiniteTriangularNetwork_pullback(Δnetwork_)
        Δnetwork = unthunk(Δnetwork_)
        Δtop = _rewrap(top, map(ket, unitcell(Δnetwork)))
        Δbot = _rewrap(bot, map(bra, unitcell(Δnetwork)))
        return NoTangent(), Δtop, Δbot
    end
    return network, InfiniteTriangularNetwork_pullback
end

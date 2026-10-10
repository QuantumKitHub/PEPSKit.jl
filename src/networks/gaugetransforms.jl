"""
    VirtualGaugeTransform(matrices[, inverses])
    VirtualGaugeTransform(pairs)

Gauge transformations on the virtual bonds of an [`InfinitePEPS`](@ref) or [`InfinitePEPO`](@ref).
`matrices` contains one invertible `MPSBondTensor` per north/east bond, with shape `(2, rows, cols)` for a PEPS or `(2, rows, cols, layers)` for a PEPO.
`inverses` contains the corresponding inverse maps and is computed when omitted; alternatively, supply an array of `(X, X⁻¹)` pairs.
Explicit inverse maps must have compatible spaces and are assumed to be the inverses of `matrices`.

At `(r, c)`, the north and east legs receive `matrices[NORTH, r, c]` and `matrices[EAST, r, c]`.
The south and west legs receive the inverses belonging to the north bond at `(r + 1, c)` and the east bond at `(r, c - 1)`, respectively, with periodic indexing.
For a PEPO, the same convention applies independently to each layer, leaving both physical legs unchanged.
Use [`CTMRGEnvGaugeTransform`](@ref) for transformations of CTMRG boundary legs.
"""
struct VirtualGaugeTransform{N, A <: AbstractArray{<:MPSBondTensor, N}, B <: AbstractArray{<:MPSBondTensor, N}}
    matrices::A
    inverses::B
    function VirtualGaugeTransform(
            matrices::AbstractArray{<:MPSBondTensor, N},
            inverses::AbstractArray{<:MPSBondTensor, N},
        ) where {N}
        N in (3, 4) || throw(ArgumentError("Virtual gauges require direction, row, column, and optionally layer indices"))
        _check_gauge_matrices(matrices, inverses, 2)
        return new{N, typeof(matrices), typeof(inverses)}(matrices, inverses)
    end
end

"""
    CTMRGEnvGaugeTransform(matrices[, inverses])
    CTMRGEnvGaugeTransform(pairs)

Gauge transformations on the boundary legs of a [`CTMRGEnv`](@ref), with shape `(4, rows, cols)` ordered north, east, south, west.
Each matrix is an `MPSBondTensor` acting on the first (outgoing) boundary leg of the edge at the same coordinate.
The inverse at the neighboring bond acts on its last (incoming) boundary leg, and the corner tensors are transformed consistently.
The neighbor is east, south, west, or north for a north, east, south, or west edge, respectively.
The network-facing virtual legs are unchanged.

The matrices need not be unitary.
Inverse maps are computed when omitted, or may be supplied explicitly with compatible spaces; alternatively, supply `(X, X⁻¹)` pairs.
Explicit inverse maps are assumed to be the inverses of `matrices`.
This follows the bond indexing used by CTMRG environment gauge fixing.
"""
struct CTMRGEnvGaugeTransform{A <: AbstractArray{<:MPSBondTensor, 3}, B <: AbstractArray{<:MPSBondTensor, 3}}
    matrices::A
    inverses::B
    function CTMRGEnvGaugeTransform(
            matrices::AbstractArray{<:MPSBondTensor, 3},
            inverses::AbstractArray{<:MPSBondTensor, 3},
        )
        _check_gauge_matrices(matrices, inverses, 4)
        return new{typeof(matrices), typeof(inverses)}(matrices, inverses)
    end
end

"""Validate the layout and tensor-map spaces of a collection of gauge pairs."""
function _check_gauge_matrices(
        matrices::AbstractArray{<:MPSBondTensor}, inverses::AbstractArray{<:MPSBondTensor}, ndirs::Int
    )
    Base.require_one_based_indexing(matrices, inverses)
    axes(matrices) == axes(inverses) || throw(DimensionMismatch("Gauge matrices and inverses must have matching axes"))
    size(matrices, 1) == ndirs || throw(DimensionMismatch("Expected $ndirs gauge directions"))
    all(>(0), size(matrices)) || throw(ArgumentError("Gauge arrays must be nonempty"))
    for (X, Xinv) in zip(matrices, inverses)
        domain(X) == codomain(Xinv) && codomain(X) == domain(Xinv) ||
            throw(SpaceMismatch("Gauge matrices and inverses must have opposite domain and codomain spaces"))
    end
    return nothing
end

VirtualGaugeTransform(matrices::AbstractArray{<:MPSBondTensor}) =
    VirtualGaugeTransform(matrices, inv.(matrices))
CTMRGEnvGaugeTransform(matrices::AbstractArray{<:MPSBondTensor, 3}) =
    CTMRGEnvGaugeTransform(matrices, inv.(matrices))
VirtualGaugeTransform(pairs::AbstractArray{<:Tuple{MPSBondTensor, MPSBondTensor}}) =
    VirtualGaugeTransform(first.(pairs), last.(pairs))
CTMRGEnvGaugeTransform(pairs::AbstractArray{<:Tuple{MPSBondTensor, MPSBondTensor}, 3}) =
    CTMRGEnvGaugeTransform(first.(pairs), last.(pairs))

Base.inv(gauge::VirtualGaugeTransform) = VirtualGaugeTransform(gauge.inverses, gauge.matrices)
Base.inv(gauge::CTMRGEnvGaugeTransform) = CTMRGEnvGaugeTransform(gauge.inverses, gauge.matrices)

"""Rotate virtual bonds counterclockwise, reversing the old north bonds into new east bonds."""
function _rotl90_virtual_gauge(
        matrices::AbstractArray{<:MPSBondTensor}, inverses::AbstractArray{<:MPSBondTensor}
    )
    nr, nc = size(matrices)[2:3]
    return map(CartesianIndices((2, nc, nr, size(matrices)[4:end]...))) do I
        d, r, c = Tuple(I)[1:3]
        layer = Tuple(I)[4:end]
        # Match the leg permutation in state rotations, including fermionic signs.
        return d == NORTH ? matrices[EAST, c, nc + 1 - r, layer...] :
            permute(inverses[NORTH, _next(c, nr), nc + 1 - r, layer...], ((2,), (1,)))
    end
end

"""Rotate virtual bonds clockwise, reversing the old east bonds into new north bonds."""
function _rotr90_virtual_gauge(
        matrices::AbstractArray{<:MPSBondTensor}, inverses::AbstractArray{<:MPSBondTensor}
    )
    nr, nc = size(matrices)[2:3]
    return map(CartesianIndices((2, nc, nr, size(matrices)[4:end]...))) do I
        d, r, c = Tuple(I)[1:3]
        layer = Tuple(I)[4:end]
        return d == NORTH ?
            permute(inverses[EAST, nr + 1 - c, _prev(r, nc), layer...], ((2,), (1,))) :
            matrices[NORTH, nr + 1 - c, r, layer...]
    end
end

"""Rotate virtual bonds by a half turn, reversing both bond directions using inverse maps."""
function _rot180_virtual_gauge(inverses::AbstractArray{<:MPSBondTensor})
    nr, nc = size(inverses)[2:3]
    return map(CartesianIndices(inverses)) do I
        d, r, c = Tuple(I)[1:3]
        layer = Tuple(I)[4:end]
        r, c = nr + 1 - r, nc + 1 - c
        Xinv = d == NORTH ? inverses[NORTH, _next(r, nr), c, layer...] :
            inverses[EAST, r, _prev(c, nc), layer...]
        return permute(Xinv, ((2,), (1,)))
    end
end

"""Rotate boundary gauge directions and unit-cell coordinates counterclockwise."""
function _rotl90_boundary_gauge(matrices::AbstractArray{<:MPSBondTensor, 3})
    return map(CartesianIndices((4, size(matrices, 3), size(matrices, 2)))) do I
        d, r, c = Tuple(I)
        return matrices[_next(d, 4), c, size(matrices, 3) + 1 - r]
    end
end

"""Rotate boundary gauge directions and unit-cell coordinates clockwise."""
function _rotr90_boundary_gauge(matrices::AbstractArray{<:MPSBondTensor, 3})
    return map(CartesianIndices((4, size(matrices, 3), size(matrices, 2)))) do I
        d, r, c = Tuple(I)
        return matrices[_prev(d, 4), size(matrices, 2) + 1 - c, r]
    end
end

"""Rotate boundary gauge directions and unit-cell coordinates by a half turn."""
function _rot180_boundary_gauge(matrices::AbstractArray{<:MPSBondTensor, 3})
    return map(CartesianIndices(matrices)) do I
        d, r, c = Tuple(I)
        return matrices[mod1(d + 2, 4), size(matrices, 2) + 1 - r, size(matrices, 3) + 1 - c]
    end
end

"""
    rotl90(gauge::Union{VirtualGaugeTransform, CTMRGEnvGaugeTransform})
    rotr90(gauge::Union{VirtualGaugeTransform, CTMRGEnvGaugeTransform})
    rot180(gauge::Union{VirtualGaugeTransform, CTMRGEnvGaugeTransform})

Rotate a gauge consistently with its state or CTMRG environment, leaving PEPO layers in place.
For each rotation `R`, `R(gauge_transform(state, gauge)) ≈ gauge_transform(R(state), R(gauge))`.
"""
Base.rotl90(gauge::VirtualGaugeTransform) = VirtualGaugeTransform(
    _rotl90_virtual_gauge(gauge.matrices, gauge.inverses),
    _rotl90_virtual_gauge(gauge.inverses, gauge.matrices)
)
Base.rotl90(gauge::CTMRGEnvGaugeTransform) = CTMRGEnvGaugeTransform(
    _rotl90_boundary_gauge(gauge.matrices), _rotl90_boundary_gauge(gauge.inverses)
)
Base.rotr90(gauge::VirtualGaugeTransform) = VirtualGaugeTransform(
    _rotr90_virtual_gauge(gauge.matrices, gauge.inverses),
    _rotr90_virtual_gauge(gauge.inverses, gauge.matrices)
)
Base.rotr90(gauge::CTMRGEnvGaugeTransform) = CTMRGEnvGaugeTransform(
    _rotr90_boundary_gauge(gauge.matrices), _rotr90_boundary_gauge(gauge.inverses)
)
Base.rot180(gauge::VirtualGaugeTransform) = VirtualGaugeTransform(
    _rot180_virtual_gauge(gauge.inverses), _rot180_virtual_gauge(gauge.matrices)
)
Base.rot180(gauge::CTMRGEnvGaugeTransform) = CTMRGEnvGaugeTransform(
    _rot180_boundary_gauge(gauge.matrices), _rot180_boundary_gauge(gauge.inverses)
)

"""Check that a gauge array has the expected unit cell and layer count."""
function _check_gauge_size(gauge::Union{VirtualGaugeTransform, CTMRGEnvGaugeTransform}, dims::Tuple)
    size(gauge.matrices)[2:end] == dims || throw(DimensionMismatch("Gauge unit cell does not match the object being transformed"))
    return nothing
end

"""Retrieve the north, east, south, and west maps acting on a site's virtual legs."""
function _virtual_gauge_factors(gauge::VirtualGaugeTransform, r::Int, c::Int, layer::Int...)
    X, Xinv = gauge.matrices, gauge.inverses
    return X[NORTH, r, c, layer...], X[EAST, r, c, layer...],
        Xinv[NORTH, _next(r, size(X, 2)), c, layer...],
        Xinv[EAST, r, _prev(c, size(X, 3)), layer...]
end

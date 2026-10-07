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

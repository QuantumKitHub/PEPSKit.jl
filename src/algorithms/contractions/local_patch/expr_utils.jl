# Contraction expression utilities for local patches
# --------------------------------------------------

"""
$(SIGNATURES)

Returns which slot of `open` the patch position `(r, c)` occupies, or `nothing` if that site
carries no open physical leg. `open` itself may be `nothing`, meaning no site does.
"""
_open_slot(open, r, c) = isnothing(open) ? nothing : findfirst(==(CartesianIndex(r, c)), open)

"""
$(SIGNATURES)

Returns the virtual leg labels of the bulk factor at patch position `(i, j)` of `layer`, in
domain order `(N, E, S, W)`.

This is the bulk half of the label protocol, and is the same for every state type: legs facing
the perimeter of a `gridsize` patch take the `virtuallabel(SIDE, layer, ...)` labels that
[`boundary_contraction_expr`](@ref) consumes, while legs facing another site take an interior
`:horizontal` or `:vertical` label shared with that neighbour.
"""
function _bulk_virtuallabels(i, j, layer, gridsize)
    return (
        i == 1 ? virtuallabel(NORTH, layer, j) : virtuallabel(:vertical, layer, i - 1, j),
        j == gridsize[2] ? virtuallabel(EAST, layer, i) :
            virtuallabel(:horizontal, layer, i, j),
        i == gridsize[1] ? virtuallabel(SOUTH, layer, j) : virtuallabel(:vertical, layer, i, j),
        j == 1 ? virtuallabel(WEST, layer, i) : virtuallabel(:horizontal, layer, i, j - 1),
    )
end


# Patch geometry
# --------------

"""
$(SIGNATURES)

Recover the patch coordinates from `Val`-encoded indices, checking that they do not overlap.
Uses an implementation in the type domain, so that the patch geometry is available to the
generated contraction expressions.
"""
function _patch_inds(inds::Type)
    sites = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters))
    allunique(sites) || throw(ArgumentError("Indices should not overlap: $sites."))
    return sites
end
_patch_inds(inds::Tuple{Vararg{Val}}) = _patch_inds(typeof(inds))

"""
$(SIGNATURES)

Row and column ranges of the rectangular patch spanned by `sites`.
"""
function _patch_ranges(sites)
    rows, cols = getindex.(sites, 1), getindex.(sites, 2)
    return UnitRange(extrema(rows)...), UnitRange(extrema(cols)...)
end

"""
$(SIGNATURES)

Number of rows and columns of the rectangular patch spanned by `sites`.
"""
function _patch_gridsize(sites)
    rowrange, colrange = _patch_ranges(sites)
    return length(rowrange), length(colrange)
end

"""
$(SIGNATURES)

Shape of the patch spanned by `sites` as a string, e.g. `"2x2"`. For error messages and shape
guards of environments which only support a restricted patch geometry.
"""
function _patch_shape_string(sites)
    nrows, ncols = _patch_gridsize(sites)
    return "$(nrows)x$(ncols)"
end

# Contraction of local operators on arbitrary lattice locations
# -------------------------------------------------------------


# Contraction label helpers
# -------------------------

function tensorlabel(args...)
    return Symbol(ntuple(i -> iseven(i) ? :_ : args[(i + 1) >> 1], 2 * length(args) - 1)...)
end
envlabel(args...) = tensorlabel(:χ, args...)
virtuallabel(args...) = tensorlabel(:D, args...)
physicallabel(args...) = tensorlabel(:d, args...)

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


# Assembly helpers
# ----------------

"""
$(SIGNATURES)

Check whether the `@tensor` contraction for a given state type should be emitted with
`contractcheck = true`. Enabled for PEPO sandwiches, disabled for other state types,
in order to preserve the legacy behavior for different state types.
"""
contractcheck(state::Type) = false
contractcheck(::Type{<:InfinitePEPO}) = true
contractcheck(::Type{<:Tuple{Vararg{InfinitePEPO}}}) = true

"""
$(SIGNATURES)

Wrap an assembled product in `@autoopt @tensor`, with `lhs` on the left if given and as a
scalar otherwise, emitting `contractcheck = true` only when `check` is set.
"""
function _tensor_expr(prod, lhs, check::Bool)
    return if isnothing(lhs)
        check ? :(@autoopt @tensor contractcheck = true $prod) : :(@autoopt @tensor $prod)
    else
        check ? :(@autoopt @tensor contractcheck = true $lhs := $prod) :
            :(@autoopt @tensor $lhs := $prod)
    end
end


# Contraction expression generators
# ---------------------------------

## Boundary: environment

"""
    boundary_contraction_expr(::Type{Env}, rowrange, colrange)

Build the contraction expressions for the environment factors surrounding the patch spanned
by `rowrange` and `colrange`, dispatching on the type of the environment.

The returned factors must consume exactly the perimeter labels the bulk exposes -
`virtuallabel(NORTH, layer, j)`, `virtuallabel(EAST, layer, i)`,
`virtuallabel(SOUTH, layer, j)` and `virtuallabel(WEST, layer, i)`, one per layer - and must
close every label they introduce themselves among their own factors.

!!! note
    The expression generator assumes the environment variable name is `env`.
"""
boundary_contraction_expr(env::Type, rowrange, colrange) = throw(
    ArgumentError("No patch boundary contraction defined for environments of type $env.")
)

function boundary_contraction_expr(
        ::Type{<:CTMRGEnv{C, T}}, rowrange, colrange
    ) where {C, T}
    # the edges carry one virtual leg per layer of the sandwich, on top of the two environment
    # indices threading the ring, so the height follows from the edge tensor type
    height = numout(T) - 1
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    C_NW = :(corner(env, NORTHWEST, $(rmin - 1), $(cmin - 1)))
    corner_NW = tensorexpr(C_NW, envlabel(WEST, 0), envlabel(NORTH, 0))

    C_NE = :(corner(env, NORTHEAST, $(rmin - 1), $(cmax + 1)))
    corner_NE = tensorexpr(C_NE, envlabel(NORTH, gridsize[2]), envlabel(EAST, 0))

    C_SE = :(corner(env, SOUTHEAST, $(rmax + 1), $(cmax + 1)))
    corner_SE = tensorexpr(C_SE, envlabel(EAST, gridsize[1]), envlabel(SOUTH, gridsize[2]))

    C_SW = :(corner(env, SOUTHWEST, $(rmax + 1), $(cmin - 1)))
    corner_SW = tensorexpr(C_SW, envlabel(SOUTH, 0), envlabel(WEST, gridsize[1]))

    edges_N = map(1:gridsize[2]) do i
        E_N = :(edge(env, NORTH, $(rmin - 1), $(cmin + i - 1)))
        return tensorexpr(
            E_N,
            (envlabel(NORTH, i - 1), virtuallabel.(NORTH, ntuple(identity, height), i)...),
            envlabel(NORTH, i),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        E_E = :(edge(env, EAST, $(rmin + i - 1), $(cmax + 1)))
        return tensorexpr(
            E_E,
            (envlabel(EAST, i - 1), virtuallabel.(EAST, ntuple(identity, height), i)...),
            envlabel(EAST, i),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        E_S = :(edge(env, SOUTH, $(rmax + 1), $(cmin + i - 1)))
        return tensorexpr(
            E_S,
            (envlabel(SOUTH, i), virtuallabel.(SOUTH, ntuple(identity, height), i)...),
            envlabel(SOUTH, i - 1),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        E_W = :(edge(env, WEST, $(rmin + i - 1), $(cmin - 1)))
        return tensorexpr(
            E_W,
            (envlabel(WEST, i), virtuallabel.(WEST, ntuple(identity, height), i)...),
            envlabel(WEST, i - 1),
        )
    end

    return [
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
    ]
end


## Bulk: state

"""
    bulk_contraction_expr(::Type{State}, rowrange, colrange, open)

Build the contraction expressions for the factors inside the patch spanned by `rowrange`
and `colrange`, dispatching on the type of the state.

`open` is the vector of sites whose physical legs are left open, or `nothing` when every
physical leg is contracted within the bulk. Open sites are labelled
`physicallabel(:O, layer, slot)`, where `slot` is the position of the site in `open`; closed
sites share a label between the layers. Physical labels are this generator's business alone.

Interior bonds must be closed among the returned factors, which must expose their
perimeter-facing virtual legs under the labels [`boundary_contraction_expr`](@ref) consumes.

!!! note
    The expression generator assumes the state variable name is `state`.
"""
bulk_contraction_expr(state::Type, rowrange, colrange, open) = throw(
    ArgumentError("No patch bulk contraction defined for states of type $state.")
)

function bulk_contraction_expr(
        ::Type{<:Tuple{InfinitePEPS, InfinitePEPS}}, rowrange, colrange, open
    )
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    layers = map(1:2) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
            physical_label = if isnothing(inds_id)
                physicallabel(i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            return tensorexpr(
                :(state[$(side)][$(rmin + i - 1), $(cmin + j - 1)]),
                (physical_label,),
                _bulk_virtuallabels(i, j, side, gridsize),
            )
        end
    end

    ket, bra = layers
    return [ket..., map(x -> Expr(:call, :conj, x), bra)...]
end

function bulk_contraction_expr(::Type{<:InfinitePEPO}, rowrange, colrange, open)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    # a single layer is not wrapped in a tuple, so it is indexed directly
    layer = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
        physical_label_out = if isnothing(inds_id)
            physicallabel(i, j)     # traced over the layer
        else
            physicallabel(:O, 1, inds_id)
        end
        physical_label_in = if isnothing(inds_id)
            physicallabel(i, j)
        else
            physicallabel(:O, 2, inds_id)
        end
        return tensorexpr(
            :(twistdual(state[$(rmin + i - 1), $(cmin + j - 1)], 2)),
            (physical_label_out, physical_label_in),
            _bulk_virtuallabels(i, j, 1, gridsize),
        )
    end

    return vec(layer)
end

function bulk_contraction_expr(
        ::Type{<:Tuple{InfinitePEPO, InfinitePEPO}}, rowrange, colrange, open
    )
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    layers = map(1:2) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = _open_slot(open, rmin + i - 1, cmin + j - 1)
            physical_label_out = if isnothing(inds_id)
                physicallabel(:out, i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            # the two layers are linked through a shared label on the open sites
            physical_label_in = if isnothing(inds_id)
                physicallabel(:in, i, j)
            else
                physicallabel(:Oopen, inds_id)
            end
            tensor_name = if side == 2
                :(state[2][$(rmin + i - 1), $(cmin + j - 1)])
            else
                :(twistdual(state[1][$(rmin + i - 1), $(cmin + j - 1)], (1, 2)))
            end
            return tensorexpr(
                tensor_name, (physical_label_out, physical_label_in),
                _bulk_virtuallabels(i, j, side, gridsize),
            )
        end
    end

    ket, bra = layers
    return [ket..., map(x -> Expr(:call, :conj, x), bra)...]
end

function bulk_contraction_expr(
        state::Type{<:Tuple{Vararg{InfinitePEPO}}}, rowrange, colrange, open
    )
    return throw(
        ArgumentError(
            "Cannot contract a patch of $(length(state.parameters)) PEPO layers; only a \
            single layer or a two-layer sandwich are supported."
        )
    )
end


## Operator

"""
    operator_contraction_expr(::Type{Operator}, nsites)

Build the contraction expressions for the operator factors inserted into a patch acting on
`nsites` sites, dispatching on the type of the operator.

The factors refer to the enclosing generated function's `operator` argument by name, and
attach to the open physical legs of the bulk: `physicallabel(:O, 2, k)` is the bra index of
site `k` and `physicallabel(:O, 1, k)` its ket index. Every state type labels its open legs
with that same pair, so this generator does not depend on the state. Any label it introduces
itself must be closed among its own factors.

!!! note
    The expression generator assumes the operator variable name is `operator`.
"""
operator_contraction_expr(operator::Type, nsites) = throw(
    ArgumentError("No patch operator contraction defined for operators of type $operator.")
)

function operator_contraction_expr(::Type{<:AbstractTensorMap}, nsites)
    return [
        tensorexpr(
            :operator,
            ntuple(i -> physicallabel(:O, 2, i), nsites),
            ntuple(i -> physicallabel(:O, 1, i), nsites),
        ),
    ]
end

# The MPO bond is small - the operator's Schmidt rank, e.g. 3 for Heisenberg XYZ - so it is
# labelled as a physical rather than a virtual dimension, which gives `@autoopt` the right
# order of magnitude when it searches for a contraction order.
mpolabel(args...) = physicallabel(:mpo, args...)

function operator_contraction_expr(::Type{<:MPOTerm}, nsites)
    # one factor per site, linked by a chain of bonds:
    #   W_1 : bra_1 <- ket_1 (x) b_1,  W_i : bra_i <- b_{i-1} (x) ket_i (x) b_i,
    #   W_N : bra_N <- b_{N-1} (x) ket_N
    return map(1:nsites) do i
        out = (physicallabel(:O, 2, i),)
        ins = if nsites == 1
            (physicallabel(:O, 1, i),)
        elseif i == 1
            (physicallabel(:O, 1, i), mpolabel(1))
        elseif i == nsites
            (mpolabel(nsites - 1), physicallabel(:O, 1, i))
        else
            (mpolabel(i - 1), physicallabel(:O, 1, i), mpolabel(i))
        end
        return tensorexpr(:(operator[$i]), out, ins)
    end
end


# Low-level patch contractions using generated contraction expressions
# --------------------------------------------------------------------

"""
$(SIGNATURES)

Contract the rectangular patch spanned by `inds` with `operator` inserted on those sites,
leaving no physical leg open, and return the resulting scalar.

The sites are carried as `Val` parameters so that the patch geometry is available while the
contraction is generated, and `state` is the tuple of layers making up the sandwich. The
expression is assembled from [`boundary_contraction_expr`](@ref),
[`bulk_contraction_expr`](@ref) and [`operator_contraction_expr`](@ref), which dispatch on the
types of `env`, `state` and `operator` respectively, so this single method covers every
combination those three have methods for.
"""
@generated function _contract_local_operator(
        inds::NTuple{N, Val}, operator, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, sites)...,
        operator_contraction_expr(operator, N)...,
    )

    returnex = _tensor_expr(multiplication_ex, nothing, contractcheck(state))
    return macroexpand(@__MODULE__, returnex)
end

"""
$(SIGNATURES)

Contract the rectangular patch spanned by `inds` with no operator inserted, pairing the
physical legs of the layers on every site, and return the resulting scalar.

Assembled as [`_contract_local_operator`](@ref), except that
[`bulk_contraction_expr`](@ref) is passed `nothing` in place of the open sites, so no physical
leg is left open and no operator factor is needed. Note this is the norm of the patch within
the given environment, not the physical norm of the state.
"""
@generated function _contract_local_norm(
        inds::NTuple{N, Val}, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, nothing)...,   # legs paired, not open
    )

    returnex = _tensor_expr(multiplication_ex, nothing, contractcheck(state))
    return macroexpand(@__MODULE__, returnex)
end

"""
$(SIGNATURES)

Contract the rectangular patch spanned by `inds` leaving the physical legs of those sites
open, and return the resulting reduced density matrix, normalized by its supertrace.

Assembled as [`_contract_local_operator`](@ref) but without an operator factor, so the open
legs become the indices of `ρ`: `physicallabel(:O, 1, k)` in its codomain and
`physicallabel(:O, 2, k)` in its domain, for each site `k`. Normalization uses `str` rather
than `tr`, since the supertrace carries the fermionic signs.
"""
@generated function _contract_densitymatrix(
        inds::NTuple{N, Val}, state, env
    ) where {N}
    sites = _patch_inds(inds)
    rowrange, colrange = _patch_ranges(sites)

    multiplication_ex = Expr(
        :call, :*,
        boundary_contraction_expr(env, rowrange, colrange)...,
        bulk_contraction_expr(state, rowrange, colrange, sites)...,
    )
    result = tensorexpr(
        :ρ,
        ntuple(i -> physicallabel(:O, 1, i), N),
        ntuple(i -> physicallabel(:O, 2, i), N),
    )

    multex = _tensor_expr(multiplication_ex, result, contractcheck(state))
    return quote
        $(macroexpand(@__MODULE__, multex))
        return ρ / str(ρ)
    end
end

# Fast path specializations
# -------------------------

# Special case 1x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
# EE_SWA is largest object so keep largest legs to the front there
function reduced_densitymatrix1x1(
        inds::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(inds)

    # Unpack variables and absorb corners
    A = ket[row, col]
    Ā = bra[row, col]

    E_north = absorb_right(
        edge(env, NORTH, row - 1, col), corner(env, NORTHEAST, row - 1, col + 1)
    )
    E_east = absorb_right(
        edge(env, EAST, row, col + 1), corner(env, SOUTHEAST, row + 1, col + 1)
    )
    E_south = absorb_right(
        edge(env, SOUTH, row + 1, col), corner(env, SOUTHWEST, row + 1, col - 1)
    )
    E_west = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_SW[χSE χNW DSb DWb; DSt DWt] :=
        E_south[χSE DSt DSb; χSW] * E_west[χSW DWt DWb; χNW]

    @tensor EE_SWA[χSE χNW DNt DEt; dt DSb DWb] :=
        EE_SW[χSE χNW DSb DWb; DSt DWt] * A[dt; DNt DEt DSt DWt]

    @tensor EE_NE[DNb DEb; χSE χNW DNt DEt] :=
        E_north[χNW DNt DNb; χNE] * E_east[χNE DEt DEb; χSE]

    @tensor EEAEE[dt; DNb DEb DSb DWb] :=
        EE_NE[DNb DEb; χSE χNW DNt DEt] * EE_SWA[χSE χNW DNt DEt; dt DSb DWb]

    @tensor ρ[dt; db] := EEAEE[dt; DNb DEb DSb DWb] * conj(Ā[db; DNb DEb DSb DWb])

    return ρ / str(ρ)
end

# Special case 2x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
function reduced_densitymatrix2x1(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_north = ket[row, col]
    Ā_north = bra[row, col]
    A_south = ket[row + 1, col]
    Ā_south = bra[row + 1, col]

    E_north = absorb_right(
        edge(env, NORTH, row - 1, col), corner(env, NORTHEAST, row - 1, col + 1)
    )
    E_northeast = edge(env, EAST, row, col + 1)
    E_southeast = absorb_right(
        edge(env, EAST, row + 1, col + 1), corner(env, SOUTHEAST, row + 2, col + 1)
    )
    E_south = absorb_right(
        edge(env, SOUTH, row + 2, col), corner(env, SOUTHWEST, row + 2, col - 1)
    )
    E_southwest = edge(env, WEST, row + 1, col - 1)
    E_northwest = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_NW[χW χNE DNWt DNt; DNWb DNb] :=
        E_northwest[χW DNWt DNWb; χNW] * E_north[χNW DNt DNb; χNE]
    @tensor EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] :=
        EE_NW[χW χNE DNWt DNt; DNWb DNb] * conj(Ā_north[dNb; DNb DNEb DMb DNWb])
    @tensor EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] :=
        EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] * A_north[dNt; DNt DNEt DMt DNWt]
    @tensor EEEAA_N[dNt dNb; χW DMt DMb χE] :=
        EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] * E_northeast[χNE DNEt DNEb; χE]

    @tensor EE_SE[χE χSW DSEt DSt; DSEb DSb] :=
        E_southeast[χE DSEt DSEb; χSE] * E_south[χSE DSt DSb; χSW]
    @tensor EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] :=
        EE_SE[χE χSW DSEt DSt; DSEb DSb] * conj(Ā_south[dSb; DMb DSEb DSb DSWb])
    @tensor EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] :=
        EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] * A_south[dSt; DMt DSEt DSt DSWt]
    @tensor EEEAA_S[χW DMt DMb χE; dSt dSb] :=
        EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] * E_southwest[χSW DSWt DSWb; χW]

    @tensor ρ[dNt dSt; dNb dSb] :=
        EEEAA_N[dNt dNb; χW DMt DMb χE] * EEEAA_S[χW DMt DMb χE; dSt dSb]

    return ρ / str(ρ)
end

function reduced_densitymatrix1x2(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_west = ket[row, col]
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    E_northwest = edge(env, NORTH, row - 1, col)
    E_northeast = absorb_right(
        edge(env, NORTH, row - 1, col + 1), corner(env, NORTHEAST, row - 1, col + 2)
    )
    E_east = absorb_right(
        edge(env, EAST, row, col + 2), corner(env, SOUTHEAST, row + 1, col + 2)
    )
    E_southeast = edge(env, SOUTH, row + 1, col + 1)
    E_southwest = absorb_right(
        edge(env, SOUTH, row + 1, col), corner(env, SOUTHWEST, row + 1, col - 1)
    )
    E_west = absorb_right(
        edge(env, WEST, row, col - 1), corner(env, NORTHWEST, row - 1, col - 1)
    )

    @tensor EE_SW[χS χNW DSWt DWt; DSWb DWb] :=
        E_southwest[χS DSWt DSWb; χSW] * E_west[χSW DWt DWb; χNW]
    @tensor EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] :=
        EE_SW[χS χNW DSWt DWt; DSWb DWb] * conj(Ā_west[dWb; DNWb DMb DSWb DWb])
    @tensor EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] :=
        EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] * A_west[dWt; DNWt DMt DSWt DWt]
    @tensor EEEAA_W[dWt dWb; χS DMt DMb χN] :=
        EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] * E_northwest[χNW DNWt DNWb; χN]

    @tensor EE_NE[χN χSE DNEt DEt; DNEb DEb] :=
        E_northeast[χN DNEt DNEb; χNE] * E_east[χNE DEt DEb; χSE]
    @tensor EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] :=
        EE_NE[χN χSE DNEt DEt; DNEb DEb] * conj(Ā_east[dEb; DNEb DEb DSEb DMb])
    @tensor EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] :=
        EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] * A_east[dEt; DNEt DEt DSEt DMt]
    @tensor EEEAA_E[χS DMt DMb χN; dEt dEb] :=
        EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] * E_southeast[χSE DSEt DSEb; χS]

    @tensor ρ[dWt dEt; dWb dEb] :=
        EEEAA_W[dWt dWb; χS DMt DMb χN] * EEEAA_E[χS DMt DMb χN; dEt dEb]

    return ρ / str(ρ)
end

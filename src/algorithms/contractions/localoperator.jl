# Contraction of local operators on arbitrary lattice locations
# -------------------------------------------------------------

# currently need this because MPSKit restricts tensor names to symbols
_totuple(t) = t isa Tuple ? t : tuple(t)
MPSKit.tensorexpr(ex::Expr, inds::Tuple) = Expr(:ref, ex, _totuple(inds)...)
function MPSKit.tensorexpr(ex::Expr, indout, indin)
    return Expr(
        :typed_vcat, ex, Expr(:row, _totuple(indout)...), Expr(:row, _totuple(indin)...)
    )
end

function tensorlabel(args...)
    return Symbol(ntuple(i -> iseven(i) ? :_ : args[(i + 1) >> 1], 2 * length(args) - 1)...)
end
envlabel(args...) = tensorlabel(:χ, args...)
virtuallabel(args...) = tensorlabel(:D, args...)
physicallabel(args...) = tensorlabel(:d, args...)

"""
$(SIGNATURES)

Contract a local operator `O` on the PEPS `peps` at the indices `inds` using the environment `env`.

This works by generating the appropriate contraction on a rectangular patch with its corners
specified by `inds`. The `peps` is contracted with `O` from above and below, and the PEPS-operator
sandwich is surrounded with the appropriate environment tensors.
"""
function contract_local_operator(
        inds::NTuple{N, CartesianIndex{2}},
        O::AbstractTensorMap{T, S, N, N},
        ket::InfinitePEPS, bra::InfinitePEPS,
        env::CTMRGEnv,
    ) where {T, S, N}
    static_inds = Val.(inds)
    return _contract_local_operator(static_inds, O, ket, bra, env)
end
function contract_local_operator(
        inds::NTuple{N, Tuple{Int, Int}},
        O::AbstractTensorMap{T, S, N, N},
        ket::InfinitePEPS, bra::InfinitePEPS,
        env::CTMRGEnv,
    ) where {T, S, N}
    return contract_local_operator(CartesianIndex.(inds), O, ket, bra, env)
end

# This implements the contraction of an operator acting on sites `inds`.
# The generated function ensures that we can use @tensor to write dynamic contractions (and maximize performance).

function _contract_corner_expr(rowrange, colrange)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    C_NW = :(env.corners[NORTHWEST, mod1($(rmin - 1), end), mod1($(cmin - 1), end)])
    corner_NW = tensorexpr(C_NW, envlabel(WEST, 0), envlabel(NORTH, 0))

    C_NE = :(env.corners[NORTHEAST, mod1($(rmin - 1), end), mod1($(cmax + 1), end)])
    corner_NE = tensorexpr(C_NE, envlabel(NORTH, gridsize[2]), envlabel(EAST, 0))

    C_SE = :(env.corners[SOUTHEAST, mod1($(rmax + 1), end), mod1($(cmax + 1), end)])
    corner_SE = tensorexpr(C_SE, envlabel(EAST, gridsize[1]), envlabel(SOUTH, gridsize[2]))

    C_SW = :(env.corners[SOUTHWEST, mod1($(rmax + 1), end), mod1($(cmin - 1), end)])
    corner_SW = tensorexpr(C_SW, envlabel(SOUTH, 0), envlabel(WEST, gridsize[1]))

    return corner_NW, corner_NE, corner_SE, corner_SW
end

function _contract_edge_expr(rowrange, colrange)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    edges_N = map(1:gridsize[2]) do i
        E_N = :(env.edges[NORTH, mod1($(rmin - 1), end), mod1($(cmin + i - 1), end)])
        return tensorexpr(
            E_N,
            (
                envlabel(NORTH, i - 1),
                virtuallabel(NORTH, :ket, i),
                virtuallabel(NORTH, :bra, i),
            ),
            envlabel(NORTH, i),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        E_E = :(env.edges[EAST, mod1($(rmin + i - 1), end), mod1($(cmax + 1), end)])
        return tensorexpr(
            E_E,
            (
                envlabel(EAST, i - 1),
                virtuallabel(EAST, :ket, i),
                virtuallabel(EAST, :bra, i),
            ),
            envlabel(EAST, i),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        E_S = :(env.edges[SOUTH, mod1($(rmax + 1), end), mod1($(cmin + i - 1), end)])
        return tensorexpr(
            E_S,
            (
                envlabel(SOUTH, i),
                virtuallabel(SOUTH, :ket, i),
                virtuallabel(SOUTH, :bra, i),
            ),
            envlabel(SOUTH, i - 1),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        E_W = :(env.edges[WEST, mod1($(rmin + i - 1), end), mod1($(cmin - 1), end)])
        return tensorexpr(
            E_W,
            (envlabel(WEST, i), virtuallabel(WEST, :ket, i), virtuallabel(WEST, :bra, i)),
            envlabel(WEST, i - 1),
        )
    end

    return edges_N, edges_E, edges_S, edges_W
end

function _contract_state_expr(rowrange, colrange, cartesian_inds = nothing)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    return map((:bra, :ket)) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = if isnothing(cartesian_inds)
                nothing
            else
                findfirst(==(CartesianIndex(rmin + i - 1, cmin + j - 1)), cartesian_inds)
            end
            physical_label = if isnothing(inds_id)
                physicallabel(i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            return tensorexpr(
                :($(side)[mod1($(rmin + i - 1), end), mod1($(cmin + j - 1), end)]),
                (physical_label,),
                (
                    if i == 1
                        virtuallabel(NORTH, side, j)
                    else
                        virtuallabel(:vertical, side, i - 1, j)
                    end,
                    if j == gridsize[2]
                        virtuallabel(EAST, side, i)
                    else
                        virtuallabel(:horizontal, side, i, j)
                    end,
                    if i == gridsize[1]
                        virtuallabel(SOUTH, side, j)
                    else
                        virtuallabel(:vertical, side, i, j)
                    end,
                    if j == 1
                        virtuallabel(WEST, side, i)
                    else
                        virtuallabel(:horizontal, side, i, j - 1)
                    end,
                ),
            )
        end
    end
end

function _contract_pepo_state_expr(rowrange, colrange, cartesian_inds = nothing)
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    return map((:bra, :ket)) do side
        return map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
            inds_id = if isnothing(cartesian_inds)
                nothing
            else
                findfirst(==(CartesianIndex(rmin + i - 1, cmin + j - 1)), cartesian_inds)
            end
            physical_label_in = if isnothing(inds_id)
                physicallabel(:in, i, j)
            else
                physicallabel(:Oopen, inds_id)
            end
            physical_label_out = if isnothing(inds_id)
                physicallabel(:out, i, j)
            else
                physicallabel(:O, side, inds_id)
            end
            return tensorexpr(
                if side == :ket
                    :(twistdual(ket[mod1($(rmin + i - 1), end), mod1($(cmin + j - 1), end)], (1, 2)))
                else
                    :(bra[mod1($(rmin + i - 1), end), mod1($(cmin + j - 1), end)])
                end,
                (physical_label_out, physical_label_in),
                (
                    if i == 1
                        virtuallabel(NORTH, side, j)
                    else
                        virtuallabel(:vertical, side, i - 1, j)
                    end,
                    if j == gridsize[2]
                        virtuallabel(EAST, side, i)
                    else
                        virtuallabel(:horizontal, side, i, j)
                    end,
                    if i == gridsize[1]
                        virtuallabel(SOUTH, side, j)
                    else
                        virtuallabel(:vertical, side, i, j)
                    end,
                    if j == 1
                        virtuallabel(WEST, side, i)
                    else
                        virtuallabel(:horizontal, side, i, j - 1)
                    end,
                ),
            )
        end
    end
end

@generated function _contract_local_operator(
        inds::NTuple{N, Val},
        O::AbstractTensorMap{T, S, N, N},
        ket::InfinitePEPS, bra::InfinitePEPS,
        env::CTMRGEnv,
    ) where {T, S, N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange)
    operator = tensorexpr(
        :O,
        ntuple(i -> physicallabel(:O, :bra, i), N),
        ntuple(i -> physicallabel(:O, :ket, i), N),
    )
    bra, ket = _contract_state_expr(rowrange, colrange, cartesian_inds)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
        operator,
    )

    returnex = quote
        @autoopt @tensor $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end

"""
$(SIGNATURES)

Contract a local norm of the PEPS `peps` around indices `inds`.

This works analogously to [`contract_local_operator`](@ref) by generating the contraction
on a rectangular patch based on `inds` but replacing the operator with an identity such
that the PEPS norm is computed. (Note that this is not the physical norm of the state.)
"""
function contract_local_norm(
        inds::NTuple{N, CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    static_inds = Val.(inds)
    return _contract_local_norm(static_inds, ket, bra, env)
end
function contract_local_norm(
        inds::NTuple{N, Tuple{Int, Int}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    return contract_local_norm(CartesianIndex.(inds), ket, bra, env)
end
@generated function _contract_local_norm(
        inds::NTuple{N, Val}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) || throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange)
    bra, ket = _contract_state_expr(rowrange, colrange)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
    )

    returnex = quote
        @autoopt @tensor $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end

@doc """
$(SIGNATURES)

Construct the reduced density matrix `ρ` of the PEPS `peps` with open indices `inds` using the environment `env`.
Alternatively, construct the reduced density matrix `ρ` of a full density matrix PEPO with open indices `inds` using the environment `env`.

This works by generating the appropriate contraction on a rectangular patch with its corners
specified by `inds`. The result is normalized such that `tr(ρ) = 1`. 
""" reduced_densitymatrix

function reduced_densitymatrix(
        inds::NTuple{N, CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    static_inds = Val.(inds)
    return _contract_densitymatrix(static_inds, ket, bra, env)
end
function reduced_densitymatrix(
        inds::NTuple{N, CartesianIndex{2}}, ket::InfinitePEPO, bra::InfinitePEPO, env::CTMRGEnv
    ) where {N}
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Val.(inds)
    return _contract_densitymatrix(static_inds, ket, bra, env)
end
function reduced_densitymatrix(
        inds::NTuple{N, Tuple{Int, Int}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    return reduced_densitymatrix(CartesianIndex.(inds), ket, bra, env)
end
function reduced_densitymatrix(
        inds::NTuple{N, Tuple{Int, Int}}, ket::InfinitePEPO, bra::InfinitePEPO, env::CTMRGEnv
    ) where {N}
    return reduced_densitymatrix(CartesianIndex.(inds), ket, bra, env)
end
function reduced_densitymatrix(inds, ket::InfinitePEPS, env::CTMRGEnv)
    return reduced_densitymatrix(inds, ket, ket, env)
end

# Special case 1x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
# EE_SWA is largest object so keep largest legs to the front there
function reduced_densitymatrix(
        inds::Tuple{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(inds[1])

    # Unpack variables and absorb corners
    A = ket[mod1(row, end), mod1(col, end)]
    Ā = bra[mod1(row, end), mod1(col, end)]

    E_north =
        env.edges[NORTH, mod1(row - 1, end), mod1(col, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 1, end)], 1)
    E_east =
        env.edges[EAST, mod1(row, end), mod1(col + 1, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 1, end), mod1(col + 1, end)], 1)
    E_south =
        env.edges[SOUTH, mod1(row + 1, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 1, end), mod1(col - 1, end)], 1)
    E_west =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

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

function reduced_densitymatrix(
        inds::NTuple{2, CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    if inds[2] - inds[1] == CartesianIndex(1, 0)
        return reduced_densitymatrix2x1(inds[1], ket, bra, env)
    elseif inds[2] - inds[1] == CartesianIndex(0, 1)
        return reduced_densitymatrix1x2(inds[1], ket, bra, env)
    else
        static_inds = Val.(inds)
        return _contract_densitymatrix(static_inds, ket, bra, env)
    end
end

# Special case 2x1 density matrix:
# Keep contraction order but try to optimize intermediate permutations:
function reduced_densitymatrix2x1(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    )
    row, col = Tuple(ind)

    # Unpack variables and absorb corners
    A_north = ket[mod1(row, end), mod1(col, end)]
    Ā_north = bra[mod1(row, end), mod1(col, end)]
    A_south = ket[mod1(row + 1, end), mod1(col, end)]
    Ā_south = bra[mod1(row + 1, end), mod1(col, end)]

    E_north =
        env.edges[NORTH, mod1(row - 1, end), mod1(col, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 1, end)], 1)
    E_northeast = env.edges[EAST, mod1(row, end), mod1(col + 1, end)]
    E_southeast =
        env.edges[EAST, mod1(row + 1, end), mod1(col + 1, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 2, end), mod1(col + 1, end)], 1)
    E_south =
        env.edges[SOUTH, mod1(row + 2, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 2, end), mod1(col - 1, end)], 1)
    E_southwest = env.edges[WEST, mod1(row + 1, end), mod1(col - 1, end)]
    E_northwest =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

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
    A_west = ket[mod1(row, end), mod1(col, end)]
    Ā_west = bra[mod1(row, end), mod1(col, end)]
    A_east = ket[mod1(row, end), mod1(col + 1, end)]
    Ā_east = bra[mod1(row, end), mod1(col + 1, end)]

    E_northwest = env.edges[NORTH, mod1(row - 1, end), mod1(col, end)]
    E_northeast =
        env.edges[NORTH, mod1(row - 1, end), mod1(col + 1, end)] *
        twistdual(env.corners[NORTHEAST, mod1(row - 1, end), mod1(col + 2, end)], 1)
    E_east =
        env.edges[EAST, mod1(row, end), mod1(col + 2, end)] *
        twistdual(env.corners[SOUTHEAST, mod1(row + 1, end), mod1(col + 2, end)], 1)
    E_southeast = env.edges[SOUTH, mod1(row + 1, end), mod1(col + 1, end)]
    E_southwest =
        env.edges[SOUTH, mod1(row + 1, end), mod1(col, end)] *
        twistdual(env.corners[SOUTHWEST, mod1(row + 1, end), mod1(col - 1, end)], 1)
    E_west =
        env.edges[WEST, mod1(row, end), mod1(col - 1, end)] *
        twistdual(env.corners[NORTHWEST, mod1(row - 1, end), mod1(col - 1, end)], 1)

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

@generated function _contract_densitymatrix(
        inds::NTuple{N, Val}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
    ) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange)
    result = tensorexpr(
        :ρ,
        ntuple(i -> physicallabel(:O, :ket, i), N),
        ntuple(i -> physicallabel(:O, :bra, i), N),
    )
    bra, ket = _contract_state_expr(rowrange, colrange, cartesian_inds)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
    )
    multex = :(@autoopt @tensor $result := $multiplication_ex)
    return quote
        $(macroexpand(@__MODULE__, multex))
        return ρ / str(ρ)
    end
end

@generated function _contract_densitymatrix(
        inds::NTuple{N, Val}, ket::InfinitePEPO, bra::InfinitePEPO, env::CTMRGEnv
    ) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange)
    result = tensorexpr(
        :ρ,
        ntuple(i -> physicallabel(:O, :ket, i), N),
        ntuple(i -> physicallabel(:O, :bra, i), N),
    )
    bra, ket = _contract_pepo_state_expr(rowrange, colrange, cartesian_inds)

    multiplication_ex = Expr(
        :call, :*,
        corner_NW, corner_NE, corner_SE, corner_SW,
        edges_N..., edges_E..., edges_S..., edges_W...,
        ket..., map(x -> Expr(:call, :conj, x), bra)...,
    )
    multex = :(@autoopt @tensor contractcheck = true $result := $multiplication_ex)
    return quote
        $(macroexpand(@__MODULE__, multex))
        return ρ / str(ρ)
    end
end

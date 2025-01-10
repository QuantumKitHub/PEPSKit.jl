# Contraction of local operators on arbitrary lattice locations
# -------------------------------------------------------------
import MPSKit: tensorexpr

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
    contract_localoperator(inds, O, peps, env)

Contract a local operator `O` on the PEPS `peps` at the indices `inds` using the environment `env`.
"""
function contract_localoperator(
    inds::NTuple{N,CartesianIndex{2}},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    static_inds = Val.(inds)
    return _contract_localoperator(static_inds, O, ket, bra, env)
end
function contract_localoperator(
    inds::NTuple{N,Tuple{Int,Int}},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    return contract_localoperator(CartesianIndex.(inds), O, ket, bra, env)
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

function _contract_state_expr(rowrange, colrange, cartesian_inds=nothing)
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
                :(bra[mod1($(rmin + i - 1), end), mod1($(cmin + j - 1), end)]),
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

@generated function _contract_localoperator(
    inds::NTuple{N,Val},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
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
        :call,
        :*,
        corner_NW,
        corner_NE,
        corner_SE,
        corner_SW,
        edges_N...,
        edges_E...,
        edges_S...,
        edges_W...,
        ket...,
        map(x -> Expr(:call, :conj, x), bra)...,
        operator,
    )

    returnex = quote
        @autoopt @tensor opt = $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end

"""
    contract_localnorm(inds, peps, env)

Contract a local norm of the PEPS `peps` around indices `inds`.
"""
function contract_localnorm(
    inds::NTuple{N,CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    static_inds = Val.(inds)
    return _contract_localnorm(static_inds, ket, bra, env)
end
function contract_localnorm(
    inds::NTuple{N,Tuple{Int,Int}}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    return contract_localnorm(CartesianIndex.(inds), ket, bra, env)
end
@generated function _contract_localnorm(
    inds::NTuple{N,Val}, ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv
) where {N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    allunique(cartesian_inds) ||
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    rowrange = getindex.(cartesian_inds, 1)
    colrange = getindex.(cartesian_inds, 2)

    corner_NW, corner_NE, corner_SE, corner_SW = _contract_corner_expr(rowrange, colrange)
    edges_N, edges_E, edges_S, edges_W = _contract_edge_expr(rowrange, colrange)
    bra, ket = _contract_state_expr(rowrange, colrange)

    multiplication_ex = Expr(
        :call,
        :*,
        corner_NW,
        corner_NE,
        corner_SE,
        corner_SW,
        edges_N...,
        edges_E...,
        edges_S...,
        edges_W...,
        ket...,
        map(x -> Expr(:call, :conj, x), bra)...,
    )

    returnex = quote
        @autoopt @tensor opt = $multiplication_ex
    end
    return macroexpand(@__MODULE__, returnex)
end

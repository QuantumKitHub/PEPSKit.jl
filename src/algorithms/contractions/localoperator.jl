# Contraction of local operators on arbitrary lattice locations
# -------------------------------------------------------------
import MPSKit: tensorexpr

# currently need this because MPSKit restricts tensor names to symbols
MPSKit.tensorexpr(ex::Expr, inds) = Expr(:ref, ex, inds...)
function MPSKit.tensorexpr(ex::Expr, indout, indin)
    return Expr(:typed_vcat, ex, Expr(:row, indout...), Expr(:row, indin...))
end

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
@generated function _contract_localoperator(
    inds::NTuple{N,Val},
    O::AbstractTensorMap{S,N,N},
    ket::InfinitePEPS,
    bra::InfinitePEPS,
    env::CTMRGEnv,
) where {S,N}
    cartesian_inds = collect(CartesianIndex{2}, map(x -> x.parameters[1], inds.parameters)) # weird hack to extract information from Val
    if !allunique(cartesian_inds)
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    end

    rmin, rmax = extrema(getindex.(cartesian_inds, 1))
    cmin, cmax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    corner_NW = tensorexpr(
        :(env.corners[
            NORTHWEST, mod1($(rmin - 1), size(ket, 1)), mod1($(cmin - 1), size(ket, 2))
        ]),
        (:χ_C_NW_1,),
        (:χ_C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[
            NORTHEAST, mod1($(rmin - 1), size(ket, 1)), mod1($(cmax + 1), size(ket, 2))
        ]),
        (:χ_C_NE_1,),
        (:χ_C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[
            SOUTHEAST, mod1($(rmax + 1), size(ket, 1)), mod1($(cmax + 1), size(ket, 2))
        ]),
        (:χ_C_SE_1,),
        (:χ_C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[
            SOUTHWEST, mod1($(rmax + 1), size(ket, 1)), mod1($(cmin - 1), size(ket, 2))
        ]),
        (:χ_C_SW_1,),
        (:χ_C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH,
                mod1($(rmin - 1), size(ket, 1)),
                mod1($(cmin + i - 1), size(ket, 2)),
            ]),
            (
                (i == 1 ? :χ_C_NW_2 : Symbol(:χ_E_N, i - 1)),
                Symbol(:D_E_N_top, i),
                Symbol(:D_E_N_bot, i),
            ),
            ((i == gridsize[2] ? :χ_C_NE_1 : Symbol(:χ_E_N, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST,
                mod1($(rmin + i - 1), size(ket, 1)),
                mod1($(cmax + 1), size(ket, 2)),
            ]),
            (
                (i == 1 ? :χ_C_NE_2 : Symbol(:χ_E_E, i - 1)),
                Symbol(:D_E_E_top, i),
                Symbol(:D_E_E_bot, i),
            ),
            ((i == gridsize[1] ? :χ_C_SE_1 : Symbol(:χ_E_E, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH,
                mod1($(rmax + 1), size(ket, 1)),
                mod1($(cmin + i - 1), size(ket, 2)),
            ]),
            (
                (i == gridsize[2] ? :χ_C_SE_2 : Symbol(:χ_E_S, i)),
                Symbol(:D_E_S_top, i),
                Symbol(:D_E_S_bot, i),
            ),
            ((i == 1 ? :χ_C_SW_1 : Symbol(:χ_E_S, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST,
                mod1($(rmin + i - 1), size(ket, 1)),
                mod1($(cmin - 1), size(ket, 2)),
            ]),
            (
                (i == gridsize[1] ? :χ_C_SW_2 : Symbol(:χ_E_W, i)),
                Symbol(:D_E_W_top, i),
                Symbol(:D_E_W_bot, i),
            ),
            ((i == 1 ? :χ_C_NW_1 : Symbol(:χ_E_W, i - 1)),),
        )
    end

    operator = tensorexpr(
        :O, ntuple(i -> Symbol(:d_O_out_, i), N), ntuple(i -> Symbol(:d_O_in_, i), N)
    )

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(rmin + i - 1, cmin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:d, i, "_", j) : Symbol(:d_O_out_, inds_id)
        return tensorexpr(
            :(bra[
                mod1($(rmin + i - 1), size(bra, 1)), mod1($(cmin + j - 1), size(bra, 2))
            ]),
            (physical_label,),
            (
                (
                    if i == 1
                        Symbol(:D_E_N_bot, j)
                    else
                        Symbol(:D_bra_vertical, i - 1, "_", j)
                    end
                ),
                (
                    if j == gridsize[2]
                        Symbol(:D_E_E_bot, i)
                    else
                        Symbol(:D_bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:D_E_S_bot, j)
                    else
                        Symbol(:D_bra_vertical, i, "_", j)
                    end
                ),
                (
                    if j == 1
                        Symbol(:D_E_W_bot, i)
                    else
                        Symbol(:D_bra_horizontal, i, "_", j - 1)
                    end
                ),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(rmin + i - 1, cmin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:d, i, "_", j) : Symbol(:d_O_in_, inds_id)
        return tensorexpr(
            :(ket[
                mod1($(rmin + i - 1), size(ket, 1)), mod1($(cmin + j - 1), size(ket, 2))
            ]),
            (physical_label,),
            (
                (
                    if i == 1
                        Symbol(:D_E_N_top, j)
                    else
                        Symbol(:D_ket_vertical, i - 1, "_", j)
                    end
                ),
                (
                    if j == gridsize[2]
                        Symbol(:D_E_E_top, i)
                    else
                        Symbol(:D_ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:D_E_S_top, j)
                    else
                        Symbol(:D_ket_vertical, i, "_", j)
                    end
                ),
                (
                    if j == 1
                        Symbol(:D_E_W_top, i)
                    else
                        Symbol(:D_ket_horizontal, i, "_", j - 1)
                    end
                ),
            ),
        )
    end

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
    if !allunique(cartesian_inds)
        throw(ArgumentError("Indices should not overlap: $cartesian_inds."))
    end

    rmin, rmax = extrema(getindex.(cartesian_inds, 1))
    cmin, cmax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    corner_NW = tensorexpr(
        :(env.corners[
            NORTHWEST, mod1($(rmin - 1), size(ket, 1)), mod1($(cmin - 1), size(ket, 2))
        ]),
        (:χ_C_NW_1,),
        (:χ_C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[
            NORTHEAST, mod1($(rmin - 1), size(ket, 1)), mod1($(cmax + 1), size(ket, 2))
        ]),
        (:χ_C_NE_1,),
        (:χ_C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[
            SOUTHEAST, mod1($(rmax + 1), size(ket, 1)), mod1($(cmax + 1), size(ket, 2))
        ]),
        (:χ_C_SE_1,),
        (:χ_C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[
            SOUTHWEST, mod1($(rmax + 1), size(ket, 1)), mod1($(cmin - 1), size(ket, 2))
        ]),
        (:χ_C_SW_1,),
        (:χ_C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH,
                mod1($(rmin - 1), size(ket, 1)),
                mod1($(cmin + i - 1), size(ket, 2)),
            ]),
            (
                (i == 1 ? :χ_C_NW_2 : Symbol(:χ_E_N, i - 1)),
                Symbol(:D_E_N_top, i),
                Symbol(:D_E_N_bot, i),
            ),
            ((i == gridsize[2] ? :χ_C_NE_1 : Symbol(:χ_E_N, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST,
                mod1($(rmin + i - 1), size(ket, 1)),
                mod1($(cmax + 1), size(ket, 2)),
            ]),
            (
                (i == 1 ? :χ_C_NE_2 : Symbol(:χ_E_E, i - 1)),
                Symbol(:D_E_E_top, i),
                Symbol(:D_E_E_bot, i),
            ),
            ((i == gridsize[1] ? :χ_C_SE_1 : Symbol(:χ_E_E, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH,
                mod1($(rmax + 1), size(ket, 1)),
                mod1($(cmin + i - 1), size(ket, 2)),
            ]),
            (
                (i == gridsize[2] ? :χ_C_SE_2 : Symbol(:χ_E_S, i)),
                Symbol(:D_E_S_top, i),
                Symbol(:D_E_S_bot, i),
            ),
            ((i == 1 ? :χ_C_SW_1 : Symbol(:χ_E_S, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST,
                mod1($(rmin + i - 1), size(ket, 1)),
                mod1($(cmin - 1), size(ket, 2)),
            ]),
            (
                (i == gridsize[1] ? :χ_C_SW_2 : Symbol(:χ_E_W, i)),
                Symbol(:D_E_W_top, i),
                Symbol(:D_E_W_bot, i),
            ),
            ((i == 1 ? :χ_C_NW_1 : Symbol(:χ_E_W, i - 1)),),
        )
    end

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(bra[
                mod1($(rmin + i - 1), size(ket, 1)), mod1($(cmin + j - 1), size(ket, 2))
            ]),
            (Symbol(:d, i, "_", j),),
            (
                (
                    if i == 1
                        Symbol(:D_E_N_bot, j)
                    else
                        Symbol(:D_bra_vertical, i - 1, "_", j)
                    end
                ),
                (
                    if j == gridsize[2]
                        Symbol(:D_E_E_bot, i)
                    else
                        Symbol(:D_bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:D_E_S_bot, j)
                    else
                        Symbol(:D_bra_vertical, i, "_", j)
                    end
                ),
                (
                    if j == 1
                        Symbol(:D_E_W_bot, i)
                    else
                        Symbol(:D_bra_horizontal, i, "_", j - 1)
                    end
                ),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(ket[
                mod1($(rmin + i - 1), size(ket, 1)), mod1($(cmin + j - 1), size(ket, 2))
            ]),
            (Symbol(:d, i, "_", j),),
            (
                (
                    if i == 1
                        Symbol(:D_E_N_top, j)
                    else
                        Symbol(:D_ket_vertical, i - 1, "_", j)
                    end
                ),
                (
                    if j == gridsize[2]
                        Symbol(:D_E_E_top, i)
                    else
                        Symbol(:D_ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:D_E_S_top, j)
                    else
                        Symbol(:D_ket_vertical, i, "_", j)
                    end
                ),
                (
                    if j == 1
                        Symbol(:D_E_W_top, i)
                    else
                        Symbol(:D_ket_horizontal, i, "_", j - 1)
                    end
                ),
            ),
        )
    end

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

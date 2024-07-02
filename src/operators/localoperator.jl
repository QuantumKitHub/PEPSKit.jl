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
# TODO: add cost model
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

    xmin, xmax = extrema(getindex.(cartesian_inds, 1))
    ymin, ymax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (xmax - xmin + 1, ymax - ymin + 1)

    corner_NW = tensorexpr(
        :(env.corners[NORTHWEST, mod1($(xmin), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_NW_1,),
        (:C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[NORTHEAST, mod1($(xmin), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_NE_1,),
        (:C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[SOUTHEAST, mod1($(xmax), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_SE_1,),
        (:C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[SOUTHWEST, mod1($(xmax), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_SW_1,),
        (:C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH, mod1($(xmin), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NW_2 : Symbol(:E_N_virtual, i - 1)),
                Symbol(:E_N_top, i),
                Symbol(:E_N_bot, i),
            ),
            ((i == gridsize[2] ? :C_NE_1 : Symbol(:E_N_virtual, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymax), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NE_2 : Symbol(:E_E_virtual, i - 1)),
                Symbol(:E_E_top, i),
                Symbol(:E_E_bot, i),
            ),
            ((i == gridsize[1] ? :C_SE_1 : Symbol(:E_E_virtual, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH, mod1($(xmax), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == gridsize[2] ? :C_SE_2 : Symbol(:E_S_virtual, i)),
                Symbol(:E_S_top, i),
                Symbol(:E_S_bot, i),
            ),
            ((i == 1 ? :C_SW_1 : Symbol(:E_S_virtual, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin), size(ket, 2))
            ]),
            (
                (i == gridsize[1] ? :C_SW_2 : Symbol(:E_W_virtual, i)),
                Symbol(:E_W_top, i),
                Symbol(:E_W_bot, i),
            ),
            ((i == 1 ? :C_NW_1 : Symbol(:E_W_virtual, i - 1)),),
        )
    end

    operator = tensorexpr(:O, (:O_out_1, :O_out_2), (:O_in_1, :O_in_2))

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(xmin + i - 1, ymin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:physical, i, "_", j) : Symbol(:O_out_, inds_id)
        return tensorexpr(
            :(bra[
                mod1($(xmin + i - 1), size(bra, 1)), mod1($(ymin + j - 1), size(bra, 2))
            ]),
            (physical_label,),
            (
                (i == 1 ? Symbol(:E_N_bot, j) : Symbol(:bra_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_bot, i)
                    else
                        Symbol(:bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_bot, j)
                    else
                        Symbol(:bra_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_bot, i) : Symbol(:bra_horizontal, i, "_", j - 1)),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        inds_id = findfirst(==(CartesianIndex(xmin + i - 1, ymin + j - 1)), cartesian_inds)
        physical_label =
            isnothing(inds_id) ? Symbol(:physical, i, "_", j) : Symbol(:O_in_, inds_id)
        return tensorexpr(
            :(ket[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (physical_label,),
            (
                (i == 1 ? Symbol(:E_N_top, j) : Symbol(:ket_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_top, i)
                    else
                        Symbol(:ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_top, j)
                    else
                        Symbol(:ket_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_top, i) : Symbol(:ket_horizontal, i, "_", j - 1)),
            ),
        )
    end

    multiplication_ex = Expr(
        :call,
        :*,
        operator,
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
    return quote
        @tensor opt = true $multiplication_ex
    end
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

    xmin, xmax = extrema(getindex.(cartesian_inds, 1))
    ymin, ymax = extrema(getindex.(cartesian_inds, 2))

    gridsize = (xmax - xmin + 1, ymax - ymin + 1)

    corner_NW = tensorexpr(
        :(env.corners[NORTHWEST, mod1($(xmin), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_NW_1,),
        (:C_NW_2,),
    )
    corner_NE = tensorexpr(
        :(env.corners[NORTHEAST, mod1($(xmin), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_NE_1,),
        (:C_NE_2,),
    )
    corner_SE = tensorexpr(
        :(env.corners[SOUTHEAST, mod1($(xmax), size(ket, 1)), mod1($(ymax), size(ket, 2))]),
        (:C_SE_1,),
        (:C_SE_2,),
    )
    corner_SW = tensorexpr(
        :(env.corners[SOUTHWEST, mod1($(xmax), size(ket, 1)), mod1($(ymin), size(ket, 2))]),
        (:C_SW_1,),
        (:C_SW_2,),
    )

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                NORTH, mod1($(xmin), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NW_2 : Symbol(:E_N_virtual, i - 1)),
                Symbol(:E_N_top, i),
                Symbol(:E_N_bot, i),
            ),
            ((i == gridsize[2] ? :C_NE_1 : Symbol(:E_N_virtual, i)),),
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                EAST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymax), size(ket, 2))
            ]),
            (
                (i == 1 ? :C_NE_2 : Symbol(:E_E_virtual, i - 1)),
                Symbol(:E_E_top, i),
                Symbol(:E_E_bot, i),
            ),
            ((i == gridsize[1] ? :C_SE_1 : Symbol(:E_E_virtual, i)),),
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(env.edges[
                SOUTH, mod1($(xmax), size(ket, 1)), mod1($(ymin + i - 1), size(ket, 2))
            ]),
            (
                (i == gridsize[2] ? :C_SE_2 : Symbol(:E_S_virtual, i)),
                Symbol(:E_S_top, i),
                Symbol(:E_S_bot, i),
            ),
            ((i == 1 ? :C_SW_1 : Symbol(:E_S_virtual, i - 1)),),
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(env.edges[
                WEST, mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin), size(ket, 2))
            ]),
            (
                (i == gridsize[1] ? :C_SW_2 : Symbol(:E_W_virtual, i)),
                Symbol(:E_W_top, i),
                Symbol(:E_W_bot, i),
            ),
            ((i == 1 ? :C_NW_1 : Symbol(:E_W_virtual, i - 1)),),
        )
    end

    bra = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(bra[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (Symbol(:physical, i, "_", j),),
            (
                (i == 1 ? Symbol(:E_N_bot, j) : Symbol(:bra_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_bot, i)
                    else
                        Symbol(:bra_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_bot, j)
                    else
                        Symbol(:bra_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_bot, i) : Symbol(:bra_horizontal, i, "_", j - 1)),
            ),
        )
    end

    ket = map(Iterators.product(1:gridsize[1], 1:gridsize[2])) do (i, j)
        return tensorexpr(
            :(ket[
                mod1($(xmin + i - 1), size(ket, 1)), mod1($(ymin + j - 1), size(ket, 2))
            ]),
            (Symbol(:physical, i, "_", j),),
            (
                (i == 1 ? Symbol(:E_N_top, j) : Symbol(:ket_vertical, i - 1, "_", j)),
                (
                    if j == gridsize[2]
                        Symbol(:E_E_top, i)
                    else
                        Symbol(:ket_horizontal, i, "_", j)
                    end
                ),
                (
                    if i == gridsize[1]
                        Symbol(:E_S_top, j)
                    else
                        Symbol(:ket_vertical, i, "_", j)
                    end
                ),
                (j == 1 ? Symbol(:E_W_top, i) : Symbol(:ket_horizontal, i, "_", j - 1)),
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
    return quote
        @tensor opt = true $multiplication_ex
    end
end
# TODO: change this implementation to a type-stable one

abstract type AbstractInteraction end

"""
    struct OnSite <: AbstractInteraction

Trivial interaction representing terms that act on one isolated site.
"""
struct OnSite <: AbstractInteraction end

"""
    struct NearestNeighbor <: AbstractInteraction

Interaction representing nearest neighbor terms that act on two adjacent sites.
"""
struct NearestNeighbor <: AbstractInteraction end

"""
    struct NLocalOperator{I<:AbstractInteraction}
    
Operator in form of a `AbstractTensorMap` which is parametrized by an interaction type.
Mostly, this is used to define Hamiltonian terms and observables.
"""
struct NLocalOperator{I<:AbstractInteraction}
    op::AbstractTensorMap
end

"""
    struct AnisotropicNNOperator{I<:AbstractInteraction}
    
Operator which includes an on-site term and two nearest-neighbor terms, vertical and horizontal.
"""
struct AnisotropicNNOperator
    h0::NLocalOperator{OnSite}
    hx::NLocalOperator{NearestNeighbor}
    hy::NLocalOperator{NearestNeighbor}
end
function AnisotropicNNOperator(
    h0::AbstractTensorMap{S,1,1},
    hx::AbstractTensorMap{S,2,2},
    hy::AbstractTensorMap{S,2,2}=hx,
) where {S}
    return AnisotropicNNOperator(
        NLocalOperator{OnSite}(h0),
        NLocalOperator{NearestNeighbor}(hx),
        NLocalOperator{NearestNeighbor}(hy),
    )
end
# TODO: include the on-site term in the two-site terms, to reduce number of contractions.

@doc """
    operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::AbstractInteraction)

Contract a PEPS and a CTMRG environment to form an operator environment.
The open bonds correspond to the indices of an operator with the specified
`AbstractInteraction` type.
"""
operator_env

function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::OnSite)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        @tensor opt = true ρ[-1; -2] :=
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][-1; 3 7 11 15] *
            conj(peps[r, c][-2; 4 8 12 16])
    end
end

function operator_env(peps::InfinitePEPS, env::CTMRGEnv, ::NearestNeighbor)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        @tensor opt = true ρ[-12 -18; -11 -20] :=
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][-12; 5 15 6 7] *
            conj(peps[r, c][-11; 8 19 10 9]) *
            peps[r, cnext][-18; 16 25 17 15] *
            conj(peps[r, cnext][-20; 22 26 21 19])
    end
end

@doc """
    MPSKit.expectation_value(peps::InfinitePEPS, env, O::NLocalOperator)

Evaluate the expectation value of any `NLocalOperator` on each unit-cell entry
of `peps` and `env`.
""" MPSKit.expectation_value(::InfinitePEPS, ::Any, ::NLocalOperator)

# Optimal contraction order is obtained by manually trying out some space sizes and using costcheck = warn
# in principle, we would like to use opt = true, but this does not give optimal results without also supplying costs
# However, due to a bug in tensoroperations this is currently not possible with integer labels.
# Thus, this is a workaround until the bug is fixed. (otherwise we'd need to rewrite all the labels to symbols...)

# 1-site operator expectation values on unit cell
function MPSKit.expectation_value(
    peps::InfinitePEPS, env::CTMRGEnv, O::NLocalOperator{OnSite}
)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        o = @tensor order = (6, 2, 5, 10, 14, 13, 11, 15, 7, 9, 1, 3, 4, 8, 12, 16, 18, 17) begin
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][17; 3 7 11 15] *
            conj(peps[r, c][18; 4 8 12 16]) *
            O.op[18; 17]
        end
        n = @tensor order = (9, 13, 10, 5, 1, 2, 4, 16, 6, 8, 14, 12, 17, 3, 7, 11, 15) begin
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][17; 3 7 11 15] *
            conj(peps[r, c][17; 4 8 12 16])
        end
        o / n
    end
end

#! format: off
function MPSKit.expectation_value(
    peps::InfinitePEPS, env, O::NLocalOperator{NearestNeighbor}
)
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        o = @tensor order = (
            28, 24, 23, 16, 25, 22, 26, 27, 17, 21, 4, 1, 3, 5, 7, 8, 9, 2, 6, 10, 14, 19,
            15, 13, 31, 32, 29, 30,
        ) begin # physical spaces
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][29; 5 15 6 7] *
            conj(peps[r, c][31; 8 19 10 9]) *
            peps[r, cnext][30; 16 25 17 15] *
            conj(peps[r, cnext][32; 22 26 21 19]) *
            O.op[31 32; 29 30]
        end

        n = @tensor order = (
            2, 3, 1, 5, 7, 28, 24, 23, 16, 25, 30, 22, 26, 27, 17, 21, 14, 15, 6, 4, 13, 29,
            8, 19, 10, 9,
        ) begin
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][29; 5 15 6 7] *
            conj(peps[r, c][29; 8 19 10 9]) *
            peps[r, cnext][30; 16 25 17 15] *
            conj(peps[r, cnext][30; 22 26 21 19])
        end
        o / n
    end
end
#! format: on

"""
    costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})
    
Compute the expectation value of a nearest-neighbor operator.
This is used to evaluate and differentiate the energy in ground-state PEPS optimizations.
"""
function costfun(peps::InfinitePEPS, env, op::NLocalOperator{NearestNeighbor})
    oh = sum(expectation_value(peps, env, op))
    ov = sum(expectation_value(rotl90(peps), rotl90(env), op))
    return real(oh + ov)
end

"""
    costfun(peps::InfinitePEPS, env, op::AnisotropicNNOperator)
    
Compute the expectation value of an on-site and an anisotropic nearest-neighbor operator.
This is used to evaluate and differentiate the energy in ground-state PEPS optimizations.
"""
function costfun(peps::InfinitePEPS, env, op::AnisotropicNNOperator)
    oos = sum(expectation_value(peps, env, op.h0))
    oh = sum(expectation_value(peps, env, op.hx))
    ov = sum(expectation_value(rotr90(peps), rotate_north(env, WEST), op.hy))
    #ov = sum(expectation_value(rotl90(peps), rotl90(env), op.hy))
    return real(oos + oh + ov)
end

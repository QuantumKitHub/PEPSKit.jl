#
# Contractions used in boundary MPS contractions of infinite square networks
#

## Repartitions

# NOTE: duplicates of MPSKit._transpose_front and MPSKit._transpose_tail internals

"""
    repartition_right(A::EdgeTensor; copy = true)

Move the physical legs of an (N, 1) tensor map representing a left isometry from the
codomain into the domain, mapping the `(χ, D...) ← χ` partition used for left-gauged tensors
onto the `χ ← (χ, D...)` partition used for right-gauged tensors.
"""
repartition_right(A::EdgeTensor; copy = true) = repartition(A, 1; copy)

"""
    repartition_left(A::RightProjector; copy = true)

Move the physical legs of a (1, N) tensor map representing a right isometry from the
codomain into the domain, mapping the `χ ← (χ, D...)` partition used for right-gauged
tensors onto the `(χ, D...) ← χ` partition used for left-gauged tensors.
"""
repartition_left(A::RightProjector; copy = true) = repartition(A, numind(A) - 1; copy)

"""
    _repartition(t::AbstractTensorMap, N₁::Int, N₂::Int=numind(t) - N₁; copy=false)

Differentiable stand-in for [`repartition`](@extref `TensorKit.repartition-Tuple{AbstractTensorMap, Int64, Int64}`). Identical to it, except that
it avoids a bug with the `backend` and `allocator` keyword arguments in the TensorKit
`rrule` implementation.

NOTE: to be removed once https://github.com/QuantumKitHub/TensorKit.jl/pull/513 is merged
and released.
"""
function _repartition(
        t::AbstractTensorMap, N₁::Int, N₂::Int = numind(t) - N₁; copy::Bool = false
    )
    N₁ + N₂ == numind(t) ||
        throw(ArgumentError("Invalid repartition: $(numind(t)) to ($N₁, $N₂)"))
    p₁, p₂ = let all_inds = (codomainind(t)..., reverse(domainind(t))...)
        ntuple(i -> all_inds[i], N₁), reverse(ntuple(i -> all_inds[i + N₁], N₂))
    end
    return transpose(t, (p₁, p₂); copy)
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(repartition_right), A::EdgeTensor
    )
    A´, repartition_pullback = rrule_via_ad(config, _repartition, A, 1)
    repartition_right_pullback(ΔA´) = NoTangent(), repartition_pullback(ΔA´)[2]
    return A´, repartition_right_pullback
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(repartition_left), A::RightProjector
    )
    A´, repartition_pullback = rrule_via_ad(config, _repartition, A, numind(A) - 1)
    repartition_left_pullback(ΔA´) = NoTangent(), repartition_pullback(ΔA´)[2]
    return A´, repartition_left_pullback
end

## Effective operators

"""
    ∂C(C::CornerTensor, GL::EdgeTensor, GR::EdgeTensor)

Apply the effective bond operator defined by the left and right environments `GL` and `GR`
to a bond tensor `C`.
This is exactly equivalent to the action of a [`MPSKit.C_hamiltonian`](@extref), but avoids
issues with AD through an [`MPSKit.MPODerivativeOperator`](@extref) and planar contractions.
"""
function ∂C(C::CornerTensor{S}, GL::EdgeTensor{S}, GR::EdgeTensor{S}) where {S}
    GR = twistdual(GR, numind(GR))
    return _∂C(C, GL, GR)
end
@generated function _∂C(
        C::CornerTensor{S}, GL::EdgeTensor{S, N}, GR::EdgeTensor{S, N}
    ) where {S, N}
    C´_e = tensorexpr(:C´, -1, -2)
    C_e = tensorexpr(:C, 1, 2)
    GL_e = tensorexpr(:GL, (-1, (3:(N + 1))...), 1)
    GR_e = tensorexpr(:GR, (2:(N + 1)...,), -2)
    return macroexpand(@__MODULE__, :(return @tensor $C´_e := $GL_e * $C_e * $GR_e))
end

"""
    ∂AC(AC::EdgeTensor, GL::EdgeTensor, O, GR::EdgeTensor)

Apply the effective site operator defined by the left and right environments `GL` and `GR`
and the local network tensor `O` to a center-gauged MPS tensor `AC`.
This is exactly equivalent to the action of an [`MPSKit.AC_hamiltonian`](@extref), but
avoids issues with AD through an [`MPSKit.MPODerivativeOperator`](@extref) and planar
contractions.
"""
function ∂AC(AC::EdgeTensor, GL::EdgeTensor, O, GR::EdgeTensor)
    GR = twistdual(GR, numind(GR))
    return _∂AC(AC, GL, O, GR)
end
function _∂AC(
        AC::EdgeTensor{S, 3}, GL::EdgeTensor{S, 3}, O::PEPSSandwich, GR::EdgeTensor{S, 3}
    ) where {S}
    return @autoopt @tensor AC′[χ_SW D_S_above D_S_below; χ_SE] :=
        GL[χ_SW D_W_above D_W_below; χ_NW] *
        AC[χ_NW D_N_above D_N_below; χ_NE] *
        GR[χ_NE D_E_above D_E_below; χ_SE] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function _∂AC(
        AC::EdgeTensor{S, 2}, GL::EdgeTensor{S, 2}, O::PartitionFunctionTensor,
        GR::EdgeTensor{S, 2},
    ) where {S}
    return @autoopt @tensor AC′[χ_SW D_S; χ_SE] :=
        GL[χ_SW D_W; χ_NW] *
        AC[χ_NW D_N; χ_NE] *
        GR[χ_NE D_E; χ_SE] *
        O[D_W D_S; D_N D_E]
end

## Center-gauged MPS tensors

# The center-gauged tensor of a boundary MPS, constructed symmetrically from both gauges. For
# a top MPS in the positions of a `BoundaryMPSEnv`, `C[r, c]` lies to its right and
# `C[r, c - 1]` to its left; for a bottom MPS in its own leg order it is the other way around.
function _center_tensor(AL, AR, C, r, c)
    return (absorb_right(_at(AL, r, c), _at(C, r, c)) + absorb_left(_at(AR, r, c), _at(C, r, c - 1))) / 2
end
function _bottom_center_tensor(AL, AR, C, r, c)
    return (absorb_right(_at(AL, r, c), _at(C, r, c - 1)) + absorb_left(_at(AR, r, c), _at(C, r, c))) / 2
end
_center_tensor(env::C4vBoundaryMPSEnv) = (absorb_right(env.AL, env.C) + absorb_left(env.AR, env.C)) / 2

# The bottom boundary MPS of a C4v environment is the conjugate of the top one, read
# from west to east. In the leg order of a south edge, its tensors are the conjugated top
# tensors with their virtual legs exchanged.
_south(A::EdgeTensor{S, 3}) where {S} = @tensor A´[χ_E D_a D_b; χ_W] := conj(A[χ_W D_a D_b; χ_E])
_south(A::EdgeTensor{S, 2}) where {S} = @tensor A´[χ_E D; χ_W] := conj(A[χ_W D; χ_E])

## Tensors surrounding a patch

# The boundary MPS tensors surrounding a patch, in the leg order and at the positions of the
# CTMRG edges with the corners absorbed:
#
# * north of `(r, c)`: the top MPS above row `r`, center-gauged on the west end of the patch
#   and right-gauged elsewhere;
# * south of `(r, c)`: the bottom MPS below row `r`, center-gauged on the east end of the
#   patch and, in its own leg order, right-gauged elsewhere;
# * east and west of `(r, c)`: the environments of the top MPS for all but the last row of
#   the patch, and the mixed environments for the last row.
function _patch_north(env::BoundaryMPSEnv, r, c, west::Bool)
    return west ? _center_tensor(env.ALt, env.ARt, env.Ct, r - 1, c) : _at(env.ARt, r - 1, c)
end
function _patch_south(env::BoundaryMPSEnv, r, c, east::Bool)
    return east ? _bottom_center_tensor(env.ALb, env.ARb, env.Cb, r + 1, c) : _at(env.ARb, r + 1, c)
end
_patch_east(env::BoundaryMPSEnv, r, c, last::Bool) = _at(last ? env.GRm : env.GRt, r, c + 1)
_patch_west(env::BoundaryMPSEnv, r, c, last::Bool) = _at(last ? env.GLm : env.GLt, r, c - 1)

# for a C4v environment, the bottom MPS is the conjugate of the top one, and all
# environments coincide
_patch_north(env::C4vBoundaryMPSEnv, r, c, west::Bool) = west ? _center_tensor(env) : env.AR
_patch_south(env::C4vBoundaryMPSEnv, r, c, east::Bool) = _south(east ? _center_tensor(env) : env.AL)
_patch_east(env::C4vBoundaryMPSEnv, r, c, last::Bool) = env.GR
_patch_west(env::C4vBoundaryMPSEnv, r, c, last::Bool) = env.GL

# the north, east, south and west tensors surrounding a single site
function _site_edges(env::BoundaryMPSEnvs, r, c)
    return (
        _patch_north(env, r, c, true), _patch_east(env, r, c, true),
        _patch_south(env, r, c, true), _patch_west(env, r, c, true),
    )
end

_edgetype(::Type{<:C4vBoundaryMPSEnv{TC, TE}}) where {TC, TE} = TE
_edgetype(::Type{<:BoundaryMPSEnv{TC, TE}}) where {TC, TE} = TE

"""
    boundary_contraction_expr(::Type{<:BoundaryMPSEnvs}, rowrange, colrange)

Build the contraction expressions for the boundary MPS tensors surrounding the patch spanned
by `rowrange` and `colrange`.

There are no corner tensors, since the center-gauged tensors of the boundary MPS absorb
them, so the edges of neighboring sides share a single environment label.
"""
function boundary_contraction_expr(::Type{E}, rowrange, colrange) where {E <: BoundaryMPSEnvs}
    # the MPS tensors carry one virtual leg per layer of the sandwich, on top of the two
    # environment indices threading the ring, so the height follows from the tensor type
    height = numout(_edgetype(E)) - 1
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    # corners are absorbed into the center-gauged tensors, so neighboring sides share a label
    north_labels = [
        envlabel(:NW), (envlabel(NORTH, i) for i in 1:(gridsize[2] - 1))..., envlabel(:NE),
    ]
    east_labels = [
        envlabel(:NE), (envlabel(EAST, i) for i in 1:(gridsize[1] - 1))..., envlabel(:SE),
    ]
    south_labels = [
        envlabel(:SW), (envlabel(SOUTH, i) for i in 1:(gridsize[2] - 1))..., envlabel(:SE),
    ]
    west_labels = [
        envlabel(:NW), (envlabel(WEST, i) for i in 1:(gridsize[1] - 1))..., envlabel(:SW),
    ]

    edges_N = map(1:gridsize[2]) do i
        return tensorexpr(
            :(_patch_north(env, $rmin, $(cmin + i - 1), $(i == 1))),
            (north_labels[i], virtuallabel.(NORTH, ntuple(identity, height), i)...),
            north_labels[i + 1],
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(_patch_east(env, $(rmin + i - 1), $cmax, $(i == gridsize[1]))),
            (east_labels[i], virtuallabel.(EAST, ntuple(identity, height), i)...),
            east_labels[i + 1],
        )
    end

    edges_S = map(1:gridsize[2]) do i
        return tensorexpr(
            :(_patch_south(env, $rmax, $(cmin + i - 1), $(i == gridsize[2]))),
            (south_labels[i + 1], virtuallabel.(SOUTH, ntuple(identity, height), i)...),
            south_labels[i],
        )
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(_patch_west(env, $(rmin + i - 1), $cmin, $(i == gridsize[1]))),
            (west_labels[i + 1], virtuallabel.(WEST, ntuple(identity, height), i)...),
            west_labels[i],
        )
    end

    return [edges_N..., edges_E..., edges_S..., edges_W...]
end

## Density matrices

# The fast paths use the contractions of the CTMRG density matrices, with the boundary MPS
# tensors surrounding the patch in place of the edges with absorbed corners.

function reduced_densitymatrix1x1(
        ind::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BoundaryMPSEnvs
    )
    r, c = Tuple(ind)
    return _densitymatrix1x1(_site_edges(env, r, c)..., ket[r, c], bra[r, c])
end

function reduced_densitymatrix2x1(
        ind::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BoundaryMPSEnvs
    )
    r, c = Tuple(ind)
    return _densitymatrix2x1(
        _patch_north(env, r, c, true),
        _patch_east(env, r, c, false), _patch_east(env, r + 1, c, true),
        _patch_south(env, r + 1, c, true),
        _patch_west(env, r + 1, c, true), _patch_west(env, r, c, false),
        ket[r, c], bra[r, c], ket[r + 1, c], bra[r + 1, c],
    )
end

function reduced_densitymatrix1x2(
        ind::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS, env::BoundaryMPSEnvs
    )
    r, c = Tuple(ind)
    return _densitymatrix1x2(
        _patch_north(env, r, c, true), _patch_north(env, r, c + 1, false),
        _patch_east(env, r, c + 1, true),
        _patch_south(env, r, c + 1, true), _patch_south(env, r, c, false),
        _patch_west(env, r, c, true),
        ket[r, c], bra[r, c], ket[r, c + 1], bra[r, c + 1],
    )
end

## Local contractions

function _contract_site(ind::Tuple{Int, Int}, network::InfiniteSquareNetwork, env::BoundaryMPSEnvs)
    return _contract_site(_site_edges(env, ind...)..., network[ind...])
end
function _contract_site(
        E_north::EdgeTensor{S, 2}, E_east::EdgeTensor{S, 2},
        E_south::EdgeTensor{S, 2}, E_west::EdgeTensor{S, 2},
        O::PartitionFunctionTensor,
    ) where {S}
    return @autoopt @tensor E_north[χ_NW D_N; χ_NE] * E_east[χ_NE D_E; χ_SE] *
        E_south[χ_SE D_S; χ_SW] * E_west[χ_SW D_W; χ_NW] * O[D_W D_S; D_N D_E]
end
function _contract_site(
        E_north::EdgeTensor{S, 3}, E_east::EdgeTensor{S, 3},
        E_south::EdgeTensor{S, 3}, E_west::EdgeTensor{S, 3},
        O::PEPSSandwich,
    ) where {S}
    return @autoopt @tensor E_north[χ_NW D_N_above D_N_below; χ_NE] *
        E_east[χ_NE D_E_above D_E_below; χ_SE] *
        E_south[χ_SE D_S_above D_S_below; χ_SW] *
        E_west[χ_SW D_W_above D_W_below; χ_NW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end

function contract_local_tensor(inds::Tuple{Int, Int}, O::PartitionFunctionTensor, env::BoundaryMPSEnvs)
    return _contract_site(_site_edges(env, inds...)..., O)
end
function contract_local_tensor(inds::CartesianIndex{2}, O::AbstractTensorMap, env::BoundaryMPSEnvs)
    return contract_local_tensor(Tuple(inds), O, env)
end

function apply_local_preconditioner(g_rc::PEPSTensor, env::BoundaryMPSEnvs, δ, (r, c), norm_pref)
    E_north, E_east, E_south, E_west = _site_edges(env, r, c)
    @autoopt @tensor g_rc_prec[d; D_N_below D_E_below D_S_below D_W_below] :=
        g_rc[d; D_N_above D_E_above D_S_above D_W_above] *
        E_north[χ_NW D_N_above D_N_below; χ_NE] *
        E_east[χ_NE D_E_above D_E_below; χ_SE] *
        E_south[χ_SE D_S_above D_S_below; χ_SW] *
        E_west[χ_SW D_W_above D_W_below; χ_NW]
    g_rc_prec = twistdual(g_rc_prec, 2:5)
    return g_rc_prec / norm_pref + δ * g_rc
end

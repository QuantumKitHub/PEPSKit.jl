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

Differentiable stand-in for [`TensorKit.repartition`](@extref). Identical to it, except that
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
This is exactly equivalent to the action of an [`MPSKit.C_Hamiltonian`](@extref), but avoids
issues with AD through an [`MPSKit.DerivativeOperator`](@extref) and planar contractions.
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
This is exactly equivalent to the action of an [`MPSKit.AC_Hamiltonian`](@extref), but
avoids issues with AD through an [`MPSKit.DerivativeOperator`](@extref) and planar
contractions.
"""
function ∂AC(AC::E, GL::E, O, GR::E) where {E <: EdgeTensor}
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

## Local contractions

"""
    get_AC(env::SymmetricBoundaryMPSEnv)

Construct the center-gauged MPS tensor of a symmetric boundary MPS environment by
symmetrically combining its left- and right-gauged forms.
"""
function get_AC(env::SymmetricBoundaryMPSEnv)
    return (absorb_right(env.AL, env.C) + absorb_left(env.AR, env.C)) / 2
end

# Accessors emitted by the generated contractions below. The north and south rows carry the
# gauge center in their westmost slot and right-gauged tensors elsewhere, while a symmetric
# boundary MPS has a single left and right environment for every row.
_north_edge(env::SymmetricBoundaryMPSEnv, i::Int) = isone(i) ? get_AC(env) : env.AR
_south_edge(env::SymmetricBoundaryMPSEnv, i::Int) = isone(i) ? get_AC(env) : env.AR
_east_edge(env::SymmetricBoundaryMPSEnv, ::Int) = env.GR
_west_edge(env::SymmetricBoundaryMPSEnv, ::Int) = env.GL

"""
    boundary_contraction_expr(::Type{<:SymmetricBoundaryMPSEnv}, rowrange, colrange)

Build the contraction expressions for the boundary MPS tensors surrounding the patch spanned
by `rowrange` and `colrange`.

Two things distinguish it from the CTMRG version. There are no corner tensors, since the
gauge center of the boundary MPS already absorbs them, so edges of neighboring sides share a
single environment label. And the boundary MPS acts as its own bra, so the south row is the
conjugate of the north row traversed in the opposite direction, which flips its codomain and
domain relative to a CTMRG south edge.
"""
function boundary_contraction_expr(
        ::Type{<:SymmetricBoundaryMPSEnv{TC, TE}}, rowrange, colrange
    ) where {TC, TE}
    # the MPS tensors carry one virtual leg per layer of the sandwich, on top of the two
    # environment indices threading the ring, so the height follows from the tensor type
    height = numout(TE) - 1
    rmin, rmax = extrema(rowrange)
    cmin, cmax = extrema(colrange)
    gridsize = (rmax - rmin + 1, cmax - cmin + 1)

    # corners are absorbed into the gauge center, so neighboring sides share a label
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
            :(_north_edge(env, $i)),
            (north_labels[i], virtuallabel.(NORTH, ntuple(identity, height), i)...),
            north_labels[i + 1],
        )
    end

    edges_E = map(1:gridsize[1]) do i
        return tensorexpr(
            :(_east_edge(env, $i)),
            (east_labels[i], virtuallabel.(EAST, ntuple(identity, height), i)...),
            east_labels[i + 1],
        )
    end

    edges_S = map(1:gridsize[2]) do i
        edge = tensorexpr(
            :(_south_edge(env, $i)),
            (south_labels[i], virtuallabel.(SOUTH, ntuple(identity, height), i)...),
            south_labels[i + 1],
        )
        return Expr(:call, :conj, edge)
    end

    edges_W = map(1:gridsize[1]) do i
        return tensorexpr(
            :(_west_edge(env, $i)),
            (west_labels[i + 1], virtuallabel.(WEST, ntuple(identity, height), i)...),
            west_labels[i],
        )
    end

    return [edges_N..., edges_E..., edges_S..., edges_W...]
end

## Density matrices and local contractions

# `reduced_densitymatrix` itself is generic in the environment; only the patch-size special
# cases it dispatches to need a symmetric boundary MPS implementation.

# Special cases mirroring the corresponding CTMRG density matrices, which keep the same
# contraction order but avoid unnecessary intermediate permutations. Since the gauge center
# of the boundary MPS already absorbs the corners, no corner absorption step is needed here;
# the only structural difference from the CTMRG versions is that the south edges are the
# conjugates of the north ones, and hence carry swapped codomain and domain.

function reduced_densitymatrix1x1(
        inds::CartesianIndex{2}, ket::InfinitePEPS, bra::InfinitePEPS,
        env::SymmetricBoundaryMPSEnv,
    )
    row, col = Tuple(inds)

    A = ket[row, col]
    Ā = bra[row, col]

    E_north = _north_edge(env, 1)
    E_east = _east_edge(env, 1)
    E_south = _south_edge(env, 1)
    E_west = _west_edge(env, 1)

    @tensor EE_SW[χSE χNW DSb DWb; DSt DWt] :=
        conj(E_south[χSW DSt DSb; χSE]) * E_west[χSW DWt DWb; χNW]

    @tensor EE_SWA[χSE χNW DNt DEt; dt DSb DWb] :=
        EE_SW[χSE χNW DSb DWb; DSt DWt] * A[dt; DNt DEt DSt DWt]

    @tensor EE_NE[DNb DEb; χSE χNW DNt DEt] :=
        E_north[χNW DNt DNb; χNE] * E_east[χNE DEt DEb; χSE]

    @tensor EEAEE[dt; DNb DEb DSb DWb] :=
        EE_NE[DNb DEb; χSE χNW DNt DEt] * EE_SWA[χSE χNW DNt DEt; dt DSb DWb]

    @tensor ρ[dt; db] := EEAEE[dt; DNb DEb DSb DWb] * conj(Ā[db; DNb DEb DSb DWb])

    return ρ / str(ρ)
end

function reduced_densitymatrix2x1(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS,
        env::SymmetricBoundaryMPSEnv,
    )
    row, col = Tuple(ind)

    A_north = ket[row, col]
    Ā_north = bra[row, col]
    A_south = ket[row + 1, col]
    Ā_south = bra[row + 1, col]

    E_north = _north_edge(env, 1)
    E_northeast = _east_edge(env, 1)
    E_southeast = _east_edge(env, 2)
    E_south = _south_edge(env, 1)
    E_southwest = _west_edge(env, 2)
    E_northwest = _west_edge(env, 1)

    @tensor EE_NW[χW χNE DNWt DNt; DNWb DNb] :=
        E_northwest[χW DNWt DNWb; χNW] * E_north[χNW DNt DNb; χNE]
    @tensor EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] :=
        EE_NW[χW χNE DNWt DNt; DNWb DNb] * conj(Ā_north[dNb; DNb DNEb DMb DNWb])
    @tensor EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] :=
        EEA_NW[χW DMb dNb χNE DNEb; DNWt DNt] * A_north[dNt; DNt DNEt DMt DNWt]
    @tensor EEEAA_N[dNt dNb; χW DMt DMb χE] :=
        EEAA_NW[χW DMb dNb dNt DMt; χNE DNEt DNEb] * E_northeast[χNE DNEt DNEb; χE]

    @tensor EE_SE[χE χSW DSEt DSt; DSEb DSb] :=
        E_southeast[χE DSEt DSEb; χSE] * conj(E_south[χSW DSt DSb; χSE])
    @tensor EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] :=
        EE_SE[χE χSW DSEt DSt; DSEb DSb] * conj(Ā_south[dSb; DMb DSEb DSb DSWb])
    @tensor EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] :=
        EEA_SE[χE DMb dSb χSW DSWb; DSEt DSt] * A_south[dSt; DMt DSEt DSt DSWt]
    @tensor EEEAA_S[χW DMt DMb χE; dSt dSb] :=
        EEAA_SE[χE DMb dSb dSt DMt; χSW DSWt DSWb] * E_southwest[χSW DSWt DSWb; χW]

    @tensor ρ[dNt dSt; dNb dSb] :=
        EEEAA_N[dNt dNb; χW DMt DMb χE] * EEEAA_S[χW DMt DMb χE; dSt dSb]

    return ρ / str(ρ)
end

function reduced_densitymatrix1x2(
        ind::CartesianIndex, ket::InfinitePEPS, bra::InfinitePEPS,
        env::SymmetricBoundaryMPSEnv,
    )
    row, col = Tuple(ind)

    A_west = ket[row, col]
    Ā_west = bra[row, col]
    A_east = ket[row, col + 1]
    Ā_east = bra[row, col + 1]

    E_northwest = _north_edge(env, 1)
    E_northeast = _north_edge(env, 2)
    E_east = _east_edge(env, 1)
    E_southeast = _south_edge(env, 2)
    E_southwest = _south_edge(env, 1)
    E_west = _west_edge(env, 1)

    @tensor EE_SW[χS χNW DSWt DWt; DSWb DWb] :=
        conj(E_southwest[χSW DSWt DSWb; χS]) * E_west[χSW DWt DWb; χNW]
    @tensor EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] :=
        EE_SW[χS χNW DSWt DWt; DSWb DWb] * conj(Ā_west[dWb; DNWb DMb DSWb DWb])
    @tensor EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] :=
        EEA_SW[χS DMb dWb χNW DNWb; DSWt DWt] * A_west[dWt; DNWt DMt DSWt DWt]
    @tensor EEEAA_W[dWt dWb; χS DMt DMb χN] :=
        EEAA_SW[χS DMb dWb dWt DMt; χNW DNWt DNWb] * E_northwest[χNW DNWt DNWb; χN]

    @tensor EE_NE[χN χSE DNEt DEt; DNEb DEb] :=
        E_northeast[χN DNEt DNEb; χNE] * E_east[χNE DEt DEb; χSE]
    @tensor EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] :=
        EE_NE[χN χSE DNEt DEt; DNEb DEb] * conj(Ā_east[dEb; DNEb DEb DSEb DMb])
    @tensor EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] :=
        EEA_NE[χN DMb dEb χSE DSEb; DNEt DEt] * A_east[dEt; DNEt DEt DSEt DMt]
    @tensor EEEAA_E[χS DMt DMb χN; dEt dEb] :=
        EEAA_NE[χN DMb dEb dEt DMt; χSE DSEt DSEb] * conj(E_southeast[χS DSEt DSEb; χSE])

    @tensor ρ[dWt dEt; dWb dEb] :=
        EEEAA_W[dWt dWb; χS DMt DMb χN] * EEEAA_E[χS DMt DMb χN; dEt dEb]

    return ρ / str(ρ)
end

## Partition function tensor insertions

function _contract_site(
        AC::EdgeTensor{S, 2}, GL::EdgeTensor{S, 2}, GR::EdgeTensor{S, 2},
        O::PartitionFunctionTensor,
    ) where {S}
    @autoopt @tensor o =
        AC[χ_NW D_N; χ_NE] *
        GR[χ_NE D_E; χ_SE] *
        conj(AC[χ_SW D_S; χ_SE]) *
        GL[χ_SW D_W; χ_NW] *
        O[D_W D_S; D_N D_E]

    return o
end

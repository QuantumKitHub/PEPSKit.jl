using MPSKit: GenericMPSTensor, MPSBondTensor

#
# Environment transfer functions
#

## PEPS

function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @tensor GL′[-1 -2 -3; -4] :=
        GL[1 2 4; 7] *
        conj(Ā[1 3 6; -1]) *
        ket(O)[5; 8 -2 3 2] *
        conj(bra(O)[5; 9 -3 6 4]) *
        A[7 8 9; -4]
end

function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @tensor GR′[-1 -2 -3; -4] :=
        GR[7 6 2; 1] *
        conj(Ā[-4 4 3; 1]) *
        ket(O)[5; 9 6 4 -2] *
        conj(bra(O)[5; 8 2 3 -3]) *
        A[-1 9 8 7]
end

## PEPO

# specialize simple case
function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S}
    @tensor GL′[-1 -2 -3 -4; -5] :=
        GL[10 7 4 2; 1] *
        conj(Ā[10 11 12 13; -1]) *
        ket(O)[8; 9 -2 11 7] *
        only(pepo(O))[5 8; 6 -3 12 4] *
        conj(bra(O)[5; 3 -4 13 2]) *
        A[1 9 6 3; -5]
end

# general case
function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: env, above, below, top, mid, bot
    tensors = [GL, A, Ā, ket(O), pepo(O)..., bra(O)]

    # contraction order: GL, A, top, mid..., bot, Ā

    # number of contracted legs for full top-mid-bot stack
    nlegs_tmb = 5 + 3 * H

    # assign and collect all contraction indices
    indicesGL = [2 + nlegs_tmb, 2, ((1:3:((H + 1) * 3)) .+ 3)..., 1]
    indicesA = [1, 3, ((1:3:((H + 1) * 3)) .+ 4)..., -(N + 1)]
    indicesĀ = [((1:N) .+ (1 + nlegs_tmb))..., -1]
    indicesTop = [6, 3, -2, 3 + nlegs_tmb, 2]
    indicesBot = [1 + nlegs_tmb, nlegs_tmb, -N, 4 + H + nlegs_tmb, nlegs_tmb - 1]
    indicesMid = Vector{Vector{Int}}(undef, H)
    for h in 1:H
        indicesMid[h] = [
            3 + 3 * (h + 1), 3 + 3 * h, 2 + 3 * h, -(2 + h), 3 + h + nlegs_tmb, 1 + 3 * h
        ]
    end
    indices = [indicesGL, indicesA, indicesĀ, indicesTop, indicesMid..., indicesBot]

    # record conjflags
    conjlist = [false, false, true, false, repeat([false], H)..., true]

    # perform contraction, permute to restore partition
    GL′ = permute(ncon(tensors, indices, conjlist), (Tuple(1:N), (N + 1,)))

    return GL′
end

# specialize simple case
function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S}
    return @tensor GR′[-1 -2 -3 -4; -5] :=
        GR[10 7 4 2; 1] *
        conj(Ā[-5 9 6 3; 1]) *
        ket(O)[8; 11 7 9 -2] *
        only(pepo(O))[5 8; 12 4 6 -3] *
        conj(bra(O)[5; 13 2 3 -4]) *
        A[-1 11 12 13; 10]
end

# general case
function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: env, above, below, top, mid, bot
    tensors = [GR, A, Ā, ket(O), pepo(O)..., bra(O)]

    # contraction order: GR, A, top, mid..., bot, Ā

    # number of contracted legs for full top-mid-bot stack
    nlegs_tmb = 5 + 3 * H

    # assign and collect all contraction indices
    indicesGR = [1, 2, ((1:3:((H + 1) * 3)) .+ 3)..., 2 + nlegs_tmb]
    indicesA = [-1, 3, ((1:3:((H + 1) * 3)) .+ 4)..., 1]
    indicesĀ = [-(N + 1), ((2:N) .+ (1 + nlegs_tmb))..., 2 + nlegs_tmb]
    indicesTop = [6, 3, 2, 3 + nlegs_tmb, -2]
    indicesBot = [1 + nlegs_tmb, nlegs_tmb, nlegs_tmb - 1, 4 + H + nlegs_tmb, -N]
    indicesMid = Vector{Vector{Int}}(undef, H)
    for h in 1:H
        indicesMid[h] = [
            3 + 3 * (h + 1), 3 + 3 * h, 2 + 3 * h, 1 + 3 * h, 3 + h + nlegs_tmb, -(2 + h)
        ]
    end
    indices = [indicesGR, indicesA, indicesĀ, indicesTop, indicesMid..., indicesBot]

    # record conjflags
    conjlist = [false, false, true, false, repeat([false], H)..., true]

    # perform contraction, permute to restore partition
    GR′ = permute(ncon(tensors, indices, conjlist), (Tuple(1:N), (N + 1,)))

    return GR′
end

@generated function environment_overlap(
    GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    GL_e = tensorexpr(:GL, (N + 1, (2:N)...), 1)
    GR_e = tensorexpr(:GR, 1:N, N + 1)
    return macroexpand(@__MODULE__, :(return o = @tensor $GL_e * $GR_e))
end

# TODO: properly implement in the most efficient way
function MPSKit.contract_mpo_expval(
    AC::GenericMPSTensor{S,N},
    GL::GenericMPSTensor{S,N},
    O::Union{PEPSSandwich,PEPOSandwich},
    GR::GenericMPSTensor{S,N},
    ACbar::GenericMPSTensor{S,N}=AC,
) where {S,N}
    GL´ = MPSKit.transfer_left(GL, O, AC, ACbar)
    return environment_overlap(GL´, GR)
end

#
# Derivative contractions
#

## PEPS

function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,3}, GR::GenericMPSTensor{S,3}
) where {S}
    return @tensor C′[-1; -2] := GL[-1 3 4; 1] * C[1; 2] * GR[2 3 4; -2]
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S}
    return @tensor AC′[-1 -2 -3; -4] :=
        GL[-1 8 9; 7] *
        AC[7 4 2; 1] *
        GR[1 6 3; -4] *
        ket(O)[5; 4 6 -2 8] *
        conj(bra(O)[5; 2 3 -3 9])
end

# PEPS derivative
function ∂peps(
    AC::GenericMPSTensor{S,3},
    ĀC::GenericMPSTensor{S,3},
    O::PEPSTensor{S},
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S}
    return @tensor ∂p[-1; -2 -3 -4 -5] :=
        GL[8 5 -5; 1] *
        AC[1 6 -2; 7] *
        O[-1; 6 3 4 5] *
        GR[7 3 -3; 2] *
        conj(ĀC[8 4 -4; 2])
end

## PEPO

# specialize simple case
function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,4}, GR::GenericMPSTensor{S,4}
) where {S}
    return @tensor C′[-1; -2] := GL[-1 3 4 5; 1] * C[1; 2] * GR[2 3 4 5; -2]
end

function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    C′ = ncon([GL, C, GR], [[-1, ((2:N) .+ 1)..., 1], [1, 2], [2, ((2:N) .+ 1)..., -2]])
    return permute(C′, ((1,), (2,)))
end

# specialize simple case
function MPSKit.∂AC(
    AC::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S}
    return @tensor AC′[-1 -2 -3 -4; -5] :=
        GL[-1 2 4 7; 1] *
        AC[1 3 5 8; 10] *
        GR[10 11 12 13; -5] *
        ket(O)[6; 3 11 -2 2] *
        only(pepo(O))[9 6; 5 12 -3 4] *
        conj(bra(O)[9; 8 13 -4 7])
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: AC, GL, GR, top, mid, bot
    tensors = [AC, GL, GR, ket(O), pepo(O)..., bra(O)]

    # contraction order: AC, GL, top, mid..., bot, GR

    # number of contracted legs for full top-mid-bot stack
    nlegs_tmb = 5 + 3 * H

    # assign and collect all contraction indices
    indicesAC = [1, 3, ((1:3:((H + 1) * 3)) .+ 4)..., 2 + nlegs_tmb]
    indicesGL = [-1, 2, ((1:3:((H + 1) * 3)) .+ 3)..., 1]
    indicesGR = [((1:N) .+ (1 + nlegs_tmb))..., -(N + 1)]
    indicesTop = [6, 3, 3 + nlegs_tmb, -2, 2]
    indicesBot = [1 + nlegs_tmb, nlegs_tmb, 4 + H + nlegs_tmb, -N, nlegs_tmb - 1]
    indicesMid = Vector{Vector{Int}}(undef, H)
    for h in 1:H
        indicesMid[h] = [
            3 + 3 * (h + 1), 3 + 3 * h, 2 + 3 * h, 3 + h + nlegs_tmb, -(2 + h), 1 + 3 * h
        ]
    end
    indices = [indicesAC, indicesGL, indicesGR, indicesTop, indicesMid..., indicesBot]

    # record conjflags
    conjlist = [false, false, false, false, repeat([false], H)..., true]

    # perform contraction, permute to restore partition
    AC′ = permute(ncon(tensors, indices, conjlist), (Tuple(1:N), (N + 1,)))

    return AC′
end

# PEPS derivative

# sandwich with the bottom dropped out...
const ∂PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,Tuple{Vararg{P,N}}}
ket(p::∂PEPOSandwich) = p[1]
pepo(p::∂PEPOSandwich) = p[2]
pepo(p::∂PEPOSandwich, i::Int) = p[2][i]

# specialize simple case
function ∂peps(
    AC::GenericMPSTensor{S,4},
    ĀC::GenericMPSTensor{S,4},
    O::∂PEPOSandwich{1},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S}
    return @tensor ∂p[-1; -2 -3 -4 -5] :=
        GL[13 8 10 -5; 1] *
        AC[1 9 11 -2; 12] *
        ket(O)[5; 9 3 4 8] *
        only(pepo(O))[-1 5; 11 6 7 10] *
        GR[12 3 6 -3; 2] *
        conj(ĀC[13 4 7 -4; 2])
end

function ∂peps(
    AC::GenericMPSTensor{S,N},
    ĀC::GenericMPSTensor{S,N},
    O::∂PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: AC, GL, top, mid, GR, ĀC
    tensors = [AC, ĀC, GL, GR, ket(O), pepo(O)...]

    # contraction order: AC, GL, top, mid..., bot, GR

    # number of contracted legs for full top-mid stack with AC and GL
    nlegs_tm = 2 + 3 * H

    # assign and collect all contraction indices
    indicesAC = [1, 3, ((1:3:((H) * 3)) .+ 4)..., -2, 2 + nlegs_tm]
    indicesGL = [2 + nlegs_tm + (N - 1), 2, ((1:3:((H) * 3)) .+ 3)..., -5, 1]
    indicesTop = [6, 3, 3 + nlegs_tm, 3 + nlegs_tm + (N - 1), 2]
    indicesMid = Vector{Vector{Int}}(undef, H)
    for h in 1:H
        indicesMid[h] = [
            3 + 3 * (h + 1),
            3 + 3 * h,
            2 + 3 * h,
            3 + h + nlegs_tm,
            3 + h + nlegs_tm + (N - 1),
            1 + 3 * h,
        ]
    end
    indicesMid[end][1] = -1 # bottom physical leg is open
    indicesGR = [((1:(N - 1)) .+ (1 + nlegs_tm))..., -3, nlegs_tm + 2 * N]
    indicesĀC = [((1:(N - 1)) .+ (nlegs_tm + N))..., -4, nlegs_tm + 2 * N]
    indices = [indicesAC, indicesĀC, indicesGL, indicesGR, indicesTop, indicesMid...]

    # record conjflags
    conjlist = [false, true, false, false, false, repeat([false], H)...]

    # perform contraction, permute to restore partition
    ∂p = permute(ncon(tensors, indices, conjlist), ((1,), Tuple(2:5)))

    return ∂p
end

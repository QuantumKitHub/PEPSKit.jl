import MPSKit.GenericMPSTensor, MPSKit.MPSBondTensor

# PEPS
# ----

"""
    struct PEPS_∂∂C{T<:GenericMPSTensor{S,N}}

Represents the effective Hamiltonian for the zero-site derivative of an MPS.
"""
struct PEPS_∂∂C{T}
    GL::T
    GR::T
end

"""
    struct PEPS_∂∂AC{T,O1,O2}

Represents the effective Hamiltonian for the one-site derivative of an MPS.
"""
struct PEPS_∂∂AC{T,O}
    top::O
    bot::O
    GL::T
    GR::T
end

function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,3}, GR::GenericMPSTensor{S,3}
) where {S}
    return @tensor C′[-1; -2] := GL[-1 3 4; 1] * C[1; 2] * GR[2 3 4; -2]
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,3},
    O::NTuple{2,T},
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S,T<:PEPSTensor}
    return @tensor AC′[-1 -2 -3; -4] :=
        GL[-1 8 9; 7] *
        AC[7 4 2; 1] *
        GR[1 6 3; -4] *
        O[1][5; 4 6 -2 8] *
        conj(O[2][5; 2 3 -3 9])
end

(H::PEPS_∂∂C)(x) = MPSKit.∂C(x, H.GL, H.GR)
(H::PEPS_∂∂AC)(x) = MPSKit.∂AC(x, (H.top, H.bot), H.GL, H.GR)

function MPSKit.∂AC(x::RecursiveVec, O::Tuple, GL, GR)
    return RecursiveVec(
        circshift(
            map((v, O1, O2, l, r) -> ∂AC(v, (O1, O2), l, r), x.vecs, O[1], O[2], GL, GR), 1
        ),
    )
end

Base.:*(H::Union{<:PEPS_∂∂AC,<:PEPS_∂∂C}, v) = H(v)

# operator constructors
function MPSKit.∂∂C(pos::Int, mps, mpo::InfiniteTransferPEPS, cache)
    return PEPS_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂C(col::Int, mps, mpo::TransferPEPSMultiline, cache)
    return PEPS_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
end
function MPSKit.∂∂C(row::Int, col::Int, mps, mpo::TransferPEPSMultiline, cache)
    return PEPS_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))
end

function MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferPEPS, cache)
    return PEPS_∂∂AC(
        mpo.top[pos], mpo.bot[pos], lefenv(cache, pos, mps), rightenv(cache, pos, mps)
    )
end
function MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::TransferPEPSMultiline, cache)
    return PEPS_∂∂AC(
        mpo[row, col]..., leftenv(cache, row, col, mps), rightenv(cache, row, col, mps)
    )
end
function MPSKit.∂∂AC(col::Int, mps, mpo::TransferPEPSMultiline, cache)
    return PEPS_∂∂AC(
        first.(mpo[:, col]),
        last.(mpo[:, col]),
        leftenv(cache, col, mps),
        rightenv(cache, col, mps),
    )
end

# PEPS derivative
function ∂peps(
    AC::GenericMPSTensor{S,3},
    ĀC::GenericMPSTensor{S,3},
    O::T,
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S,T<:PEPSTensor}
    return @tensor ∂p[-1; -2 -3 -4 -5] :=
        GL[8 5 -5; 1] *
        AC[1 6 -2; 7] *
        O[-1; 6 3 4 5] *
        GR[7 3 -3; 2] *
        conj(ĀC[8 4 -4; 2])
end

# PEPO
# ----

"""
    struct PEPO_∂∂C{T<:GenericMPSTensor{S,N}}

Represents the effective Hamiltonian for the zero-site derivative of an MPS.
"""
struct PEPO_∂∂C{T}
    GL::T
    GR::T
end

"""
    struct PEPO_∂∂AC{T,O,P}

Represents the effective Hamiltonian for the one-site derivative of an MPS.
"""
struct PEPO_∂∂AC{T,O,P}
    top::O
    bot::O
    mid::P
    GL::T
    GR::T
end

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
    O::Tuple{T,T,Tuple{P}},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S,T<:PEPSTensor,P<:PEPOTensor}
    return @tensor AC′[-1 -2 -3 -4; -5] :=
        GL[-1 2 4 7; 1] *
        AC[1 3 5 8; 10] *
        GR[10 11 12 13; -5] *
        O[1][6; 3 11 -2 2] *
        O[3][1][9 6; 5 12 -3 4] *
        conj(O[2][9; 8 13 -4 7])
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,N},
    O::Tuple{T,T,Tuple{Vararg{P,H}}},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,T<:PEPSTensor,P<:PEPOTensor,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: AC, GL, GR, top, mid, bot
    tensors = [AC, GL, GR, O[1], O[3]..., O[2]]

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

(H::PEPO_∂∂C)(x) = MPSKit.∂C(x, H.GL, H.GR)
(H::PEPO_∂∂AC)(x) = MPSKit.∂AC(x, (H.top, H.bot, H.mid), H.GL, H.GR)

function MPSKit.∂AC(x::RecursiveVec, O::Tuple{T,T,P}, GL, GR) where {T,P}
    return RecursiveVec(
        circshift(
            map(
                (v, O1, O2, O3, l, r) -> ∂AC(v, (O1, O2, O3), l, r),
                x.vecs,
                O[1],
                O[2],
                O[3],
                GL,
                GR,
            ),
            1,
        ),
    )
end

Base.:*(H::Union{<:PEPO_∂∂AC,<:PEPO_∂∂C}, v) = H(v)

# operator constructors
function MPSKit.∂∂C(pos::Int, mps, ::InfiniteTransferPEPO, cache)
    return PEPO_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂C(col::Int, mps, ::TransferPEPOMultiline, cache)
    return PEPO_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
end
function MPSKit.∂∂C(row::Int, col::Int, mps, ::TransferPEPOMultiline, cache)
    return PEPO_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))
end

function MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferPEPO, cache)
    return PEPO_∂∂AC(mpo[pos]..., lefenv(cache, pos, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::TransferPEPOMultiline, cache)
    return PEPO_∂∂AC(
        mpo[row, col]..., leftenv(cache, row, col, mps), rightenv(cache, row, col, mps)
    )
end
function MPSKit.∂∂AC(col::Int, mps, mpo::TransferPEPOMultiline, cache)
    return PEPO_∂∂AC(
        map(x -> x[1], mpo[:, col]),
        map(x -> x[2], mpo[:, col]),
        map(x -> x[3], mpo[:, col]),
        leftenv(cache, col, mps),
        rightenv(cache, col, mps),
    )
end

# PEPS derivative

# specialize simple case
function ∂peps(
    AC::GenericMPSTensor{S,4},
    ĀC::GenericMPSTensor{S,4},
    O::Tuple{T,Tuple{P}},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S,T<:PEPSTensor,P<:PEPOTensor}
    return @tensor ∂p[-1; -2 -3 -4 -5] :=
        GL[13 8 10 -5; 1] *
        AC[1 9 11 -2; 12] *
        O[1][5; 9 3 4 8] *
        O[2][1][-1 5; 11 6 7 10] *
        GR[12 3 6 -3; 2] *
        conj(ĀC[13 4 7 -4; 2])
end

function ∂peps(
    AC::GenericMPSTensor{S,N},
    ĀC::GenericMPSTensor{S,N},
    O::Tuple{T,Tuple{Vararg{P,H}}},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,T,P,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: AC, GL, top, mid, GR, ĀC
    tensors = [AC, ĀC, GL, GR, O[1], O[2]...]

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

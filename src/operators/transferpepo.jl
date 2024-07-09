"""
    InfiniteTransferPEPO{T,O}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the expectation value of an `InfinitePEPO` between 'ket' and
'bra' `InfinitePEPS` states.
"""
struct InfiniteTransferPEPO{T,O}
    top::PeriodicArray{T,1}
    mid::PeriodicArray{O,2}
    bot::PeriodicArray{T,1}
end

InfiniteTransferPEPO(top, mid) = InfiniteTransferPEPO(top, mid, top)

"""
    InfiniteTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir, row)

Constructs a transfer operator corresponding to a single row of a partition function
representing the expectation value of `O` for the state `T`. The partition function is first
rotated such that the direction `dir` faces north, after which its `row`th row from the
north is selected.
"""
function InfiniteTransferPEPO(T::InfinitePEPS, O::InfinitePEPO, dir, row)
    T = rotate_north(T, dir)
    O = rotate_north(O, dir)
    return InfiniteTransferPEPO(PeriodicArray(T[row, :]), PeriodicArray(O[row, :, :]))
end

Base.size(transfer::InfiniteTransferPEPO) = size(transfer.top)
Base.size(transfer::InfiniteTransferPEPO, args...) = size(transfer.top, args...)
Base.length(transfer::InfiniteTransferPEPO) = size(transfer, 1)
height(transfer::InfiniteTransferPEPO) = size(transfer.mid, 2)
Base.getindex(O::InfiniteTransferPEPO, i) = (O.top[i], O.bot[i], Tuple(O.mid[i, :])) # TODO: not too sure about this

Base.iterate(O::InfiniteTransferPEPO, i=1) = i > length(O) ? nothing : (O[i], i + 1)

function virtual_spaces(O::InfiniteTransferPEPO, i, dir)
    return [
        space(O.top[i], dir + 1),
        space.(O.mid[i, :], Ref(dir + 2))...,
        space(O.bot[i], dir + 1)',
    ]
end
north_spaces(O::InfiniteTransferPEPO, i) = virtual_spaces(O, i, NORTH)
east_spaces(O::InfiniteTransferPEPO, i) = virtual_spaces(O, i, EAST)
south_spaces(O::InfiniteTransferPEPO, i) = virtual_spaces(O, i, SOUTH)
west_spaces(O::InfiniteTransferPEPO, i) = virtual_spaces(O, i, WEST)

function initializeMPS(O::InfiniteTransferPEPO, virtualspaces::AbstractArray{S,1}) where {S}
    return InfiniteMPS([
        TensorMap(
            rand,
            MPSKit.Defaults.eltype, # should be scalartype of transfer PEPO?
            virtualspaces[_prev(i, end)] * prod(adjoint.(north_spaces(O, i))),
            virtualspaces[mod1(i, end)],
        ) for i in 1:length(O)
    ])
end

function initializeMPS(O::InfiniteTransferPEPO, χ::Int)
    return InfiniteMPS([
        TensorMap(
            rand, MPSKit.Defaults.eltype, ℂ^χ * prod(adjoint.(north_spaces(O, i))), ℂ^χ
        ) for i in 1:length(O)
    ])
end

"""
    const TransferPEPOMultiline = MPSKit.Multiline{<:InfiniteTransferPEPO}

Type that represents a multi-line transfer operator, where each line each corresponds to a
row of a partition function encoding the overlap of an `InfinitePEPO` between 'ket' and
'bra' `InfinitePEPS` states.
"""
const TransferPEPOMultiline = MPSKit.Multiline{<:InfiniteTransferPEPO}
Base.convert(::Type{TransferPEPOMultiline}, O::InfiniteTransferPEPO) = MPSKit.Multiline([O])
Base.getindex(t::TransferPEPOMultiline, i::Colon, j::Int) = Base.getindex.(t.data[i], j)
Base.getindex(t::TransferPEPOMultiline, i::Int, j) = Base.getindex(t.data[i], j)

"""
    TransferPEPOMultiline(T::InfinitePEPS, O::InfinitePEPO, dir)

Construct a multi-row transfer operator corresponding to the partition function representing
the expectation value of `O` for the state `T`. The partition function is first rotated such
that the direction `dir` faces north.
"""
function TransferPEPOMultiline(T::InfinitePEPS, O::InfinitePEPO, dir)
    return MPSKit.Multiline(map(cr -> InfiniteTransferPEPO(T, O, dir, cr), 1:size(T, 1)))
end

# specialize simple case
function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,4},
    O::Tuple{T,T,Tuple{P}},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S,T<:PEPSTensor,P<:PEPOTensor}
    @tensor GL′[-1 -2 -3 -4; -5] :=
        GL[10 7 4 2; 1] *
        conj(Ā[10 11 12 13; -1]) *
        O[1][8; 9 -2 11 7] *
        O[3][1][5 8; 6 -3 12 4] *
        conj(O[2][5; 3 -4 13 2]) *
        A[1 9 6 3; -5]
end

# general case
function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,N},
    O::Tuple{T,T,Tuple{Vararg{P,H}}},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,T<:PEPSTensor,P<:PEPOTensor,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: env, above, below, top, mid, bot
    tensors = [GL, A, Ā, O[1], O[3]..., O[2]]

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
    O::Tuple{T,T,Tuple{P}},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S,T<:PEPSTensor,P<:PEPOTensor}
    return @tensor GR′[-1 -2 -3 -4; -5] :=
        GR[10 7 4 2; 1] *
        conj(Ā[-5 9 6 3; 1]) *
        O[1][8; 11 7 9 -2] *
        O[3][1][5 8; 12 4 6 -3] *
        conj(O[2][5; 13 2 3 -4]) *
        A[-1 11 12 13; 10]
end

# general case
function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,N},
    O::Tuple{T,T,Tuple{Vararg{P,H}}},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,T<:PEPSTensor,P<:PEPOTensor,N,H}
    # sanity check
    @assert H == N - 3

    # collect tensors in convenient order: env, above, below, top, mid, bot
    tensors = [GR, A, Ā, O[1], O[3]..., O[2]]

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

function MPSKit.expectation_value(st::InfiniteMPS, transfer::InfiniteTransferPEPO)
    return expectation_value(
        convert(MPSMultiline, st), convert(TransferPEPOMultiline, transfer)
    )
end
function MPSKit.expectation_value(st::MPSMultiline, mpo::TransferPEPOMultiline)
    return expectation_value(st, environments(st, mpo))
end
function MPSKit.expectation_value(
    st::MPSMultiline, ca::MPSKit.PerMPOInfEnv{H,V,S,A}
) where {H<:TransferPEPOMultiline,V,S,A}
    return expectation_value(st, ca.opp, ca)
end
function MPSKit.expectation_value(
    st::MPSMultiline, opp::TransferPEPOMultiline, ca::MPSKit.PerMPOInfEnv
)
    retval = prod(product(1:size(st, 1), 1:size(st, 2))) do (i, j)
        O_ij = opp[i, j]
        N = height(opp[1]) + 4
        # just reuse left environment contraction
        GL´ = transfer_left(leftenv(ca, i, j, st), O_ij, st.AC[i, j], st.AC[i + 1, j])
        retval[i, j] = TensorOperations.tensorscalar(
            ncon([GL´, rightenv(ca, i, j, st)], [[N, (2:(N - 1))..., 1], [(1:N)...]])
        )
    end
    return retval
end

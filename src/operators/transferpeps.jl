"""
    InfiniteTransferPEPS{T}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the overlap between 'ket' and 'bra' `InfinitePEPS` states.
"""
struct InfiniteTransferPEPS{TT, BT}
    top::Matrix{TT}
    bot::Matrix{BT}
end

function InfiniteTransferPEPS(ipeps::InfinitePEPS)
    top = ipeps.A
    bot = [A' for A in ipeps.A]
    return InfiniteTransferPEPS(top, bot)
end

function ChainRulesCore.rrule(::Type{InfiniteTransferPEPS}, top::Matrix, bot::Matrix)
    function pullback(Δ)
        return NoTangent(), Δ.top, Δ.bot
    end
    return InfiniteTransferPEPS(top, bot), pullback
end

Base.eltype(transfer::InfiniteTransferPEPS) = eltype(transfer.top[1])
Base.size(transfer::InfiniteTransferPEPS) = size(transfer.top)
# Base.size(transfer::InfiniteTransferPEPS, args...) = size(transfer.top, args...)
# Base.length(transfer::InfiniteTransferPEPS) = size(transfer, 1)
# Base.getindex(O::InfiniteTransferPEPS, i) = (O.top[i], O.bot[i])

# Base.iterate(O::InfiniteTransferPEPS, i=1) = i > length(O) ? nothing : (O[i], i + 1)

# function MPSKit.transfer_left(
#     GL::GenericMPSTensor{S,3},
#     O::NTuple{2,PEPSTensor},
#     A::GenericMPSTensor{S,3},
#     Ā::GenericMPSTensor{S,3},
# ) where {S}
#     return @tensor GL′[-1 -2 -3; -4] :=
#         GL[1 2 4; 7] *
#         conj(Ā[1 3 6; -1]) *
#         O[1][5; 8 -2 3 2] *
#         conj(O[2][5; 9 -3 6 4]) *
#         A[7 8 9; -4]
# end

# function MPSKit.transfer_right(
#     GR::GenericMPSTensor{S,3},
#     O::NTuple{2,PEPSTensor},
#     A::GenericMPSTensor{S,3},
#     Ā::GenericMPSTensor{S,3},
# ) where {S}
#     return @tensor GR′[-1 -2 -3; -4] :=
#         GR[7 6 2; 1] *
#         conj(Ā[-4 4 3; 1]) *
#         O[1][5; 9 6 4 -2] *
#         conj(O[2][5; 8 2 3 -3]) *
#         A[-1 9 8 7]
# end

# @doc """
#     MPSKit.expectation_value(st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO})
#     MPSKit.expectation_value(st::MPSMultiline, op::Union{TransferPEPSMultiline,TransferPEPOMultiline})

# Compute expectation value of the transfer operator `op` for the state `st` for each site in
# the unit cell.
# """ MPSKit.expectation_value(st, op)

# function MPSKit.expectation_value(st::InfiniteMPS, transfer::InfiniteTransferPEPS)
#     return expectation_value(
#         convert(MPSMultiline, st), convert(TransferPEPSMultiline, transfer)
#     )
# end
# function MPSKit.expectation_value(st::MPSMultiline, mpo::TransferPEPSMultiline)
#     return expectation_value(st, environments(st, mpo))
# end
# function MPSKit.expectation_value(
#     st::MPSMultiline, ca::MPSKit.PerMPOInfEnv{H,V,S,A}
# ) where {H<:TransferPEPSMultiline,V,S,A}
#     return expectation_value(st, ca.opp, ca)
# end
# function MPSKit.expectation_value(
#     st::MPSMultiline, opp::TransferPEPSMultiline, ca::MPSKit.PerMPOInfEnv
# )
#     return prod(product(1:size(st, 1), 1:size(st, 2))) do (i, j)
#         O_ij = opp[i, j]
#         return @tensor leftenv(ca, i, j, st)[1 2 4; 7] *
#             conj(st[i + 1].AC[j][1 3 6; 13]) *
#             O_ij[1][5; 8 11 3 2] *
#             conj(O_ij[2][5; 9 12 6 4]) *
#             st[i].AC[j][7 8 9; 10] *
#             rightenv(ca, i, j, st)[10 11 12; 13]
#     end
# end

# @doc """
#     MPSKit.leading_boundary(
#         st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO}, alg, [envs]
#     )
#     MPSKit.leading_boundary(
#         st::MPSMulitline, op::Union{TransferPEPSMultiline,TransferPEPOMultiline}, alg, [envs]
#     )

# Approximate the leading boundary MPS eigenvector for the transfer operator `op` using `st`
# as initial guess.
# """ MPSKit.leading_boundary(st, op, alg)

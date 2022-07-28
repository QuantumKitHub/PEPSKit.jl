import MPSKit.GenericMPSTensor, MPSKit.MPSBondTensor

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

function MPSKit.∂C(C::MPSBondTensor{S}, GL::GenericMPSTensor{S,3}, GR::GenericMPSTensor{S,3}) where {S}
    return @tensor C′[-1; -2] := GL[-1 3 4; 1] * C[1; 2] * GR[2 3 4; -2]
end

function MPSKit.∂AC(AC::GenericMPSTensor{S,3}, O::NTuple{2,T}, GL::GenericMPSTensor{S,3}, GR::GenericMPSTensor{S,3}) where {S, T<:PEPSTensor}
    return @tensor AC′[-1 -2 -3; -4] := GL[-1 8 9; 7] * AC[7 4 2; 1] * GR[1 6 3; -4] * O[1][5; 4 6 -2 8] * conj(O[2][5; 2 3 -3 9])
end

(H::PEPS_∂∂C)(x) = MPSKit.∂C(x, H.GL, H.GR)
(H::PEPS_∂∂AC)(x) = MPSKit.∂AC(x, (H.top, H.bot), H.GL, H.GR)

function MPSKit.∂AC(x::RecursiveVec, O::Tuple, GL, GR)
   return RecursiveVec(
        circshift(
            map((v, O1, O2, l, r) -> ∂AC(v, (O1, O2), l, r), x.vecs, O[1], O[2], GL, GR),
            1
        )
    )
end


Base.:*(H::Union{<:PEPS_∂∂AC,<:PEPS_∂∂C}, v) = H(v)


# operator constructors
MPSKit.∂∂C(pos::Int, mps, mpo::InfiniteTransferPEPS, cache) = PEPS_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
MPSKit.∂∂C(col::Int, mps, mpo::TransferPEPSMultiline, cache) = PEPS_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
MPSKit.∂∂C(row::Int, col::Int, mps, mpo::TransferPEPSMultiline, cache) = PEPS_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))

MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferPEPS, cache) = PEPS_∂∂AC(mpo.top[pos], mpo.bot[pos], lefenv(cache, pos, mps), rightenv(cache, pos, mps))
MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::TransferPEPSMultiline, cache) = PEPS_∂∂AC(mpo[row,col]..., leftenv(cache, row, col, mps), rightenv(cache, row, col, mps))
function MPSKit.∂∂AC(col::Int, mps, mpo::TransferPEPSMultiline, cache)
    return PEPS_∂∂AC(first.(mpo[:, col]), last.(mpo[:, col]), leftenv(cache, col, mps), rightenv(cache, col, mps))
end
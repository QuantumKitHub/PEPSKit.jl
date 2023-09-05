

struct InfiniteTransferPEPS{T}
    top::PeriodicArray{T,1}
    bot::PeriodicArray{T,1}
end

InfiniteTransferPEPS(top) = InfiniteTransferPEPS(top, top)

function InfiniteTransferPEPS(T::InfinitePEPS, dir, row)
    T = rotate_north(T, dir)
    return InfiniteTransferPEPS(PeriodicArray(T[row, :]))
end

Base.size(transfer::InfiniteTransferPEPS) = size(transfer.top)
Base.size(transfer::InfiniteTransferPEPS, args...) = size(transfer.top, args...)
Base.length(transfer::InfiniteTransferPEPS) = size(transfer, 1)
Base.getindex(O::InfiniteTransferPEPS, i) = (O.top[i], O.bot[i])

Base.iterate(O::InfiniteTransferPEPS, i=1) = i > length(O) ? nothing : (O[i], i + 1)

function initializeMPS(O::InfiniteTransferPEPS, virtualspaces::AbstractArray{S,1}) where {S}
    return InfiniteMPS(
        [
            TensorMap(
                rand,
                MPSKit.Defaults.eltype,
                virtualspaces[mod1(i-1, end)] * space(O.top[i], 2)' * space(O.bot[i], 2),
                virtualspaces[i]
            ) for i in 1:length(O)
        ]
    )
end

function initializeMPS(O::InfiniteTransferPEPS, χ::Int)
    return InfiniteMPS(
        [
            TensorMap(
                rand,
                MPSKit.Defaults.eltype,
                ℂ^χ * space(O.top[i], 2)' * space(O.bot[i], 2),
                ℂ^χ
            ) for i in 1:length(O)
        ]
    )
end

import MPSKit.GenericMPSTensor

const TransferPEPSMultiline = MPSKit.Multiline{<:InfiniteTransferPEPS}
Base.convert(::Type{TransferPEPSMultiline}, O::InfiniteTransferPEPS) = MPSKit.Multiline([O])
Base.getindex(t::TransferPEPSMultiline, i::Colon, j::Int) = Base.getindex.(t.data[i], j)
Base.getindex(t::TransferPEPSMultiline, i::Int, j) = Base.getindex(t.data[i], j)


function MPSKit.transfer_left(GL::GenericMPSTensor{S,3}, O::NTuple{2, PEPSTensor}, A::GenericMPSTensor{S,3}, Ā::GenericMPSTensor{S,3}) where S
    return @tensor GL′[-1 -2 -3; -4] := GL[1 4 2; 7] * conj(Ā[1 3 6; -1]) * 
        O[1][5; 8 -2 3 4] * conj(O[2][5; 9 -3 6 2]) * A[7 8 9; -4]
end

function MPSKit.transfer_right(GR::GenericMPSTensor{S,3}, O::NTuple{2, PEPSTensor}, A::GenericMPSTensor{S,3}, Ā::GenericMPSTensor{S,3}) where S
    return @tensor GR′[-1 -2 -3; -4] := GR[7 4 2; 1] * conj(Ā[-4 6 3; 1]) *
        O[1][5; 9 4 6 -2] * conj(O[2][5; 8 2 3 -3]) * A[-1 9 8 7]
end

function MPSKit.expectation_value(st::MPSMultiline, O::TransferPEPSMultiline)

end

function MPSKit.expectation_value(st::MPSMultiline, ca::MPSKit.PerMPOInfEnv{H,V,S,A}) where {H<:TransferPEPSMultiline,V,S,A}
    opp = ca.opp
    retval = PeriodicArray{eltype(st.AC[1, 1]),2}(undef, size(st, 1), size(st, 2))
    for (i, j) in product(1:size(st, 1), 1:size(st, 2))
        retval[i, j] = @plansor leftenv(ca, i, j, st)[1 2; 3] *
                                opp[i, j][2 4; 6 5] *
                                st.AC[i, j][3 6; 7] *
                                rightenv(ca, i, j, st)[7 5; 8] *
                                conj(st.AC[i+1, j][1 4; 8])
    end
    return retval
end

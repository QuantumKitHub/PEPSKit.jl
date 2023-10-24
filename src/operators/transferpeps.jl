
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
    return InfiniteMPS([
        TensorMap(
            rand,
            MPSKit.Defaults.eltype, # should be scalartype of transfer PEPS?
            virtualspaces[_prev(i, end)] * space(O.top[i], 2)' * space(O.bot[i], 2),
            virtualspaces[mod1(i, end)],
        ) for i in 1:length(O)
    ])
end

function initializeMPS(O::InfiniteTransferPEPS, χ::Int)
    return InfiniteMPS([
        TensorMap(
            rand,
            MPSKit.Defaults.eltype,
            ℂ^χ * space(O.top[i], 2)' * space(O.bot[i], 2),
            ℂ^χ,
        ) for i in 1:length(O)
    ])
end

import MPSKit.GenericMPSTensor

const TransferPEPSMultiline = MPSKit.Multiline{<:InfiniteTransferPEPS}
Base.convert(::Type{TransferPEPSMultiline}, O::InfiniteTransferPEPS) = MPSKit.Multiline([O])
Base.getindex(t::TransferPEPSMultiline, i::Colon, j::Int) = Base.getindex.(t.data[i], j)
Base.getindex(t::TransferPEPSMultiline, i::Int, j) = Base.getindex(t.data[i], j)

# multiline patch
function TransferPEPSMultiline(T::InfinitePEPS, dir)
    return MPSKit.Multiline(map(cr -> InfiniteTransferPEPS(T, dir, cr), 1:size(T, 1)))
end
function initializeMPS(O::MPSKit.Multiline, virtualspaces::AbstractArray{S,2}) where {S}
    mpss = map(cr -> initializeMPS(O[cr], virtualspaces[cr, :]), 1:size(O, 1))
    return MPSKit.Multiline(mpss)
end
function initializeMPS(O::MPSKit.Multiline, virtualspaces::AbstractArray{S,1}) where {S}
    return initializeMPS(O, repeat(virtualspaces', length(O), 1))
end
function initializeMPS(O::MPSKit.Multiline, χ::Int)
    return initializeMPS(O, repeat([ℂ^χ], length(O), length(O[1])))
end

function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,3},
    O::NTuple{2,PEPSTensor},
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @tensor GL′[-1 -2 -3; -4] :=
        GL[1 2 4; 7] *
        conj(Ā[1 3 6; -1]) *
        O[1][5; 8 -2 3 2] *
        conj(O[2][5; 9 -3 6 4]) *
        A[7 8 9; -4]
end

function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,3},
    O::NTuple{2,PEPSTensor},
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @tensor GR′[-1 -2 -3; -4] :=
        GR[7 6 2; 1] *
        conj(Ā[-4 4 3; 1]) *
        O[1][5; 9 6 4 -2] *
        conj(O[2][5; 8 2 3 -3]) *
        A[-1 9 8 7]
end

function MPSKit.expectation_value(st::InfiniteMPS, transfer::InfiniteTransferPEPS)
    return expectation_value(
        convert(MPSMultiline, st), convert(TransferPEPSMultiline, transfer)
    )
end
function MPSKit.expectation_value(st::MPSMultiline, mpo::TransferPEPSMultiline)
    return expectation_value(st, environments(st, mpo))
end
function MPSKit.expectation_value(
    st::MPSMultiline, ca::MPSKit.PerMPOInfEnv{H,V,S,A}
) where {H<:TransferPEPSMultiline,V,S,A}
    return expectation_value(st, ca.opp, ca)
end
function MPSKit.expectation_value(
    st::MPSMultiline, opp::TransferPEPSMultiline, ca::MPSKit.PerMPOInfEnv
)
    retval = PeriodicArray{scalartype(st.AC[1, 1]),2}(undef, size(st, 1), size(st, 2))
    for (i, j) in product(1:size(st, 1), 1:size(st, 2))
        O_ij = opp[i, j]
        retval[i, j] = @tensor leftenv(ca, i, j, st)[1 2 4; 7] *
            conj(st.AC[i + 1, j][1 3 6; 13]) *
            O_ij[1][5; 8 11 3 2] *
            conj(O_ij[2][5; 9 12 6 4]) *
            st.AC[i, j][7 8 9; 10] *
            rightenv(ca, i, j, st)[10 11 12; 13]
    end
    return retval
end

# PEPS derivatives
# ----------------

"""
    InfiniteTransferPEPS{T}

Represents an infinite transfer operator corresponding to a single row of a partition
function which corresponds to the overlap between 'ket' and 'bra' `InfinitePEPS` states.
"""
struct InfiniteTransferPEPS{T}
    top::PeriodicArray{T,1}
    bot::PeriodicArray{T,1}
end

InfiniteTransferPEPS(top) = InfiniteTransferPEPS(top, top)

function ChainRulesCore.rrule(::Type{InfiniteTransferPEPS}, top::PeriodicArray{T,1}, bot::PeriodicArray{T,1}) where {T}
    function pullback(Δ)
        return NoTangent(), Δ.top, Δ.bot
    end
    return InfiniteTransferPEPS(top, bot), pullback
end

"""
    InfiniteTransferPEPS(T::InfinitePEPS, dir, row)

Constructs a transfer operator corresponding to a single row of a partition function
representing the norm of the state `T`. The partition function is first rotated such that
the direction `dir` faces north, after which its `row`th row from the north is selected.
"""
function InfiniteTransferPEPS(T::InfinitePEPS, dir, row)
    T = rotate_north(T, dir)
    return InfiniteTransferPEPS(PeriodicArray(T[row, :]))
end

Base.size(transfer::InfiniteTransferPEPS) = size(transfer.top)
Base.size(transfer::InfiniteTransferPEPS, args...) = size(transfer.top, args...)
Base.length(transfer::InfiniteTransferPEPS) = size(transfer, 1)
Base.getindex(O::InfiniteTransferPEPS, i) = (O.top[i], O.bot[i])

Base.iterate(O::InfiniteTransferPEPS, i=1) = i > length(O) ? nothing : (O[i], i + 1)

import MPSKit.GenericMPSTensor

"""
    const TransferPEPSMultiline = MPSKit.Multiline{<:InfiniteTransferPEPS}

Type that represents a multi-line transfer operator, where each line each corresponds to a
row of a partition function encoding the overlap between 'ket' and 'bra' `InfinitePEPS`
states.
"""
const TransferPEPSMultiline = MPSKit.Multiline{<:InfiniteTransferPEPS}
Base.convert(::Type{TransferPEPSMultiline}, O::InfiniteTransferPEPS) = MPSKit.Multiline([O])
Base.getindex(t::TransferPEPSMultiline, i::Colon, j::Int) = Base.getindex.(t.data[i], j)
Base.getindex(t::TransferPEPSMultiline, i::Int, j) = Base.getindex(t.data[i], j)

"""
    TransferPEPSMultiline(T::InfinitePEPS, dir)

Construct a multi-row transfer operator corresponding to the partition function representing
the norm of the state `T`. The partition function is first rotated such
that the direction `dir` faces north.
"""
function TransferPEPSMultiline(T::InfinitePEPS, dir)
    rowsize = size(T, mod1(dir, 2))  # depends on dir
    return MPSKit.Multiline(map(cr -> InfiniteTransferPEPS(T, dir, cr), 1:rowsize))
end

"""
    initializeMPS(
        O::Union{InfiniteTransferPEPS,InfiniteTransferPEPO},
        virtualspaces::AbstractArray{<:ElementarySpace,1}
    )
    initializeMPS(
        O::Union{TransferPEPSMultiline,TransferPEPOMultiline},
        virtualspaces::AbstractArray{<:ElementarySpace,2}
    )

````

    l ←------- r
        / \
       /   \
      t     d
````
Initalize a boundary MPS for the transfer operator `O` by specifying an array of virtual
spaces consistent with the unit cell.
"""
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
function initializeMPS(O::MPSKit.Multiline, virtualspaces::AbstractArray{S,2}) where {S}
    mpss = map(cr -> initializeMPS(O[cr], virtualspaces[cr, :]), 1:size(O, 1))
    return MPSKit.Multiline(mpss)
end
function initializeMPS(O::MPSKit.Multiline, virtualspaces::AbstractArray{S,1}) where {S}
    return initializeMPS(O, repeat(virtualspaces, length(O), 1))
end
function initializeMPS(O::MPSKit.Multiline, V::ElementarySpace)
    return initializeMPS(O, repeat([V], length(O), length(O[1])))
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

@doc """
    MPSKit.expectation_value(st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO})
    MPSKit.expectation_value(st::MPSMultiline, op::Union{TransferPEPSMultiline,TransferPEPOMultiline})

Compute expectation value of the transfer operator `op` for the state `st` for each site in
the unit cell.
""" MPSKit.expectation_value(st, op)

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
    return prod(product(1:size(st, 1), 1:size(st, 2))) do (i, j)
        O_ij = opp[i, j]
        return @tensor leftenv(ca, i, j, st)[1 2 4; 7] *
            conj(st.AC[i + 1, j][1 3 6; 13]) *
            O_ij[1][5; 8 11 3 2] *
            conj(O_ij[2][5; 9 12 6 4]) *
            st.AC[i, j][7 8 9; 10] *
            rightenv(ca, i, j, st)[10 11 12; 13]
    end
end

@doc """
    MPSKit.leading_boundary(
        st::InfiniteMPS, op::Union{InfiniteTransferPEPS,InfiniteTransferPEPO}, alg, [envs]
    )
    MPSKit.leading_boundary(
        st::MPSMulitline, op::Union{TransferPEPSMultiline,TransferPEPOMultiline}, alg, [envs]
    )

Approximate the leading boundary MPS eigenvector for the transfer operator `op` using `st`
as initial guess.
""" MPSKit.leading_boundary(st, op, alg)

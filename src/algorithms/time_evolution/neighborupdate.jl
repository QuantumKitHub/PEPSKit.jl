include("neighborupdate/env_ntu.jl")
include("neighborupdate/eat.jl")

"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct NeighborhoodTensorUpdate
    dt::Float64
    maxiter::Int
    trscheme::TensorKit.TruncationScheme
end

function truncation_scheme(alg::NeighborhoodTensorUpdate, v::ElementarySpace)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(v)
    else
        return alg.trscheme
    end
end

"""
Neighborhood tensor update of the x-bond between sites `[r,c]` and `[r,c+1]`

"""
function _ntu_bondx!(
    row::Int,
    col::Int,
    gate::AbstractTensorMap{S,2,2},
    peps::InfinitePEPS,
    alg::NeighborhoodTensorUpdate,
) where {S<:ElementarySpace}
    Nr, Nc = size(peps)
    #= perform SVD of the gate

        ↑       ↑       ↑       ↑
        |--gate-| ====> Ta -←- Tb
        ↑       ↑       ↑       ↑
    =#
    Ta, s, Tb = tsvd(gate, ((1, 3), (2, 4)); trunc=truncerr(1e-15))
    Ta, Tb = absorb_s(Ta, s, Tb)
    #= apply gate to peps

                   ↗            ↗
                 Ta ----←---- Tb
                ↗           ↗
            ↓ ↗         ↓ ↗                 ↓ ↗     ↓ ↗
        -←- A ----←---- B -←-    ===>   -←- A ==<== B -←-
            ↓           ↓                   ↓       ↓
    =#
    A, B = deepcopy(peps.A[row, col]), deepcopy(peps.A[row, _next(col, Nc)])
    #= perform QR

            ↓ ↗     ↓ ↗              ↓ ↗            ↓ ↗
        -←- A ==<== B -←-   ==>  -←- Qa ⇐ Ra ← Rb ⇐ Qb -←-
            ↓       ↓                ↓              ↓
    =#

    #= perform SVD on `Ra Rb` 

        ⇐ Ra ← Rb ⇐  =  ⇐ Ma ← Mb ⇐ 
    =#

    # truncate A, B

    # update peps
    peps.A[row, col], peps.A[row, _next(col, Nc)] = A, B
end

"""
One round of neighborhood tensor update

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function ntu_iter(gate::LocalOperator, peps::InfinitePEPS, alg::NeighborhoodTensorUpdate) end

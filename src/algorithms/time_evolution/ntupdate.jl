include("ntupdate/env_ntu.jl")
include("ntupdate/eat.jl")

"""
Algorithm struct for neighborhood tensor update (NTU) of infinite PEPS.
Each NTU run stops when energy starts to increase.
"""
@kwdef struct NTUpdate
    dt::Float64
    maxiter::Int
    trscheme::TensorKit.TruncationScheme
end

function truncation_scheme(alg::NTUpdate, v::ElementarySpace)
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
    alg::NTUpdate,
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
    cp1 = _next(col, Nc)
    @tensor A[d n s w; e1 e2] := Ta[d da e1] * (peps.A[row, col])[da; n e2 s w]
    @tensor B[d n e s; w1 w2] := Tb[w1 d db] * (peps.A[row, cp1])[db; n e s w2]
    #= perform QR (The numbers show axis order of each tensor)

            ↓ ↗     ↓ ↗
        -←- A ==<== B -←-
            ↓       ↓

            2  1                                    2  1
            ↓ ↗                                     ↓ ↗
        5 ← Qa ← 3   1 ← Ra<= 2   2 <=Rb → 1   5 → Qb ← 3
            ↓                                       ↓
            4                                       4
    =#
    Qa, Ra = leftorth!(A)
    Qb, Rb = leftorth!(B)
    Qa = permute(Qa, ((1,), Tuple(2:5)))
    # construct environment of Ra, Rb
    env = bondenv_NN(peps, row, col, Qa, Qb)
    #= environment assisted truncation (EAT) between Ra, Rb

        ← Ra =<= Rb →  =  ← Ma ← Mb →
    =#

    # further optimize Ma, Mb using alternating least square algorithm
    # truncate A, B with Ma, Mb and update peps
    peps.A[row, col] = A
    return peps.A[row, cp1] = B
end

"""
One round of neighborhood tensor update

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
function ntu_iter(gate::LocalOperator, peps::InfinitePEPS, alg::NTUpdate) end

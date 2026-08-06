#
# Characteristic equations used in implicit differentiation of CMTRG contractions
#

## Convenience aliases

const CornerTensor{S, N} = AbstractTensorMap{T, S, 1, 1} where {T}
const EdgeTensor{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const LeftProjector{S, N} = AbstractTensorMap{T, S, N, 1} where {T}
const RightProjector{S, N} = AbstractTensorMap{T, S, 1, N} where {T}

## C4v symmetric case

# partial contractions for different networks

# west edge and its right projector, including the corner
function contract_UdEwC(
        C::CornerTensor{S}, Ew::EdgeTensor{S, 3}, Ud::RightProjector{S, 3}, O::PEPSSandwich
    ) where {S}
    @tensor UdEwC[χ_S; χ_NNW D_N_a D_N_b D_E_a D_E_b] :=
        Ud[χ_S; χ_WSW D_S_a D_S_b] *
        Ew[χ_WSW D_W_a D_W_b; χ_WNW] *
        C[χ_WNW; χ_NNW] * # already include the corner
        ket(O)[4; D_N_a D_E_a D_S_a D_W_a] *
        conj(bra(O)[4; D_N_b D_E_b D_S_b D_W_b])
    return UdEwC
end
function contract_UdEwC(
        C::CornerTensor{S}, Ew::EdgeTensor{S, 2}, Ud::RightProjector{S, 2}, O::PartitionFunctionTensor
    ) where {S}
    @tensor UdEwC[χ_S; χ_NNW D_N D_E] :=
        Ud[χ_S; χ_WSW D_S] *
        Ew[χ_WSW D_W; χ_WNW] *
        C[χ_WNW; χ_NNW] * # already include the corner
        O[D_W D_S; D_N D_E]
    return UdEwC
end

# north edge and its left projector
function contract_EnVd(
        En::EdgeTensor{S, 3}, Vd::LeftProjector{S, 3}, O::PEPSSandwich
    ) where {S}
    @autoopt @tensor EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E] :=
        Vd[χ_NNE D_E_a D_E_b; χ_E] *
        En[χ_NNW D_N_a D_N_b; χ_NNE] *
        ket(O)[d; D_N_a D_E_a D_S_a D_W_a] *
        conj(bra(O)[d; D_N_b D_E_b D_S_b D_W_b])
    return EnVd
end
function contract_EnVd(
        En::EdgeTensor{S, 2}, Vd::LeftProjector{S, 2}, O::PartitionFunctionTensor
    ) where {S}
    @autoopt @tensor EnVd[χ_NNW D_W D_S; χ_E] :=
        Vd[χ_NNE D_E; χ_E] *
        En[χ_NNW D_N; χ_NNE] *
        O[D_W D_S; D_N D_E]
    return EnVd
end

# northwest enlarged corner with its right projector
function contract_UdEwCEn(UdEwC::RightProjector{S, 5}, En::EdgeTensor{S, 3}) where {S}
    @tensor UdEwCEn[-1; -2 -3 -4] := UdEwC[-1; 1 2 3 -3 -4] * En[1 2 3; -2]
    return UdEwCEn
end
function contract_UdEwCEn(UdEwC::RightProjector{S, 3}, En::EdgeTensor{S, 2}) where {S}
    @tensor UdEwCEn[-1; -2 -3] := UdEwC[-1; 1 2 -3] * En[1 2; -2]
    return UdEwCEn
end

# northwest enlarged corner with its left projector
function contract_EwCEnVd(Ew::EdgeTensor{S, 3}, C::CornerTensor{S}, EnVd::LeftProjector{S, 5}) where {S}
    @autoopt @tensor EwCEnVd[χ_WSW D_S_a D_S_b; χ_E] :=
        Ew[χ_WSW D_W_a D_W_b; χ_WNW] *
        C[χ_WNW; χ_NNW] *
        EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E]
    return EwCEnVd
end
function contract_EwCEnVd(Ew::EdgeTensor{S, 2}, C::CornerTensor{S}, EnVd::LeftProjector{S, 3}) where {S}
    @autoopt @tensor EwCEnVd[χ_WSW D_S; χ_E] :=
        Ew[χ_WSW D_W; χ_WNW] *
        C[χ_WNW; χ_NNW] *
        EnVd[χ_NNW D_W D_S; χ_E]
    return EwCEnVd
end

# north edge renormalization
function contract_E´(Ud::RightProjector{S, 3}, EnVd::LeftProjector{S, 5}) where {S}
    @tensor E´[-1 -2 -3; -4] := Ud[-1; 1 2 3] * EnVd[1 2 3 -2 -3; -4]
    return E´
end
function contract_E´(Ud::RightProjector{S, 2}, EnVd::LeftProjector{S, 3}) where {S}
    @tensor E´[-1 -2; -3] := Ud[-1; 1 2] * EnVd[1 2 -2; -3]
    return E´
end

# north edge and its left projector, contracted with the rest of the northwest enlarged
# corner and the nullspace of the right projector
function contract_ULdEwCEnVd(
        Ew::EdgeTensor{S, 3},
        C::CornerTensor{S},
        EnVd::LeftProjector{S, 5},
        ULd::RightProjector{S, 3},
    ) where {S}
    @autoopt @tensor ULdEwCEnVd[χ_S; χ_E] :=
        EnVd[χ_NNW D_W_a D_W_b D_S_a D_S_b; χ_E] *
        C[χ_WNW; χ_NNW] *
        Ew[χ_WSW D_W_a D_W_b; χ_WNW] *
        ULd[χ_S; χ_WSW D_S_a D_S_b]
    return ULdEwCEnVd
end
function contract_ULdEwCEnVd(
        Ew::EdgeTensor{S, 2},
        C::CornerTensor{S},
        EnVd::LeftProjector{S, 3},
        ULd::RightProjector{S, 2},
    ) where {S}
    @autoopt @tensor ULdEwCEnVd[χ_S; χ_E] :=
        EnVd[χ_NNW D_W D_S; χ_E] *
        C[χ_WNW; χ_NNW] *
        Ew[χ_WSW D_W; χ_WNW] *
        ULd[χ_S; χ_WSW D_S]
    return ULdEwCEnVd
end

"""
# TODO: docstring with diagrams
"""
function generate_symmetric_characteristic_equation(
        Cfp::CornerTensor,
        Efp::EdgeTensor, # unused
        Ufp::LeftProjector,
        ULfp::LeftProjector,
    )

    iC = sdiag_pow(real(DiagonalTensorMap(Cfp)), -1)
    ULd = ULfp'

    function symmetric_characteristic_equation(state, C, E, u)
        network = InfiniteSquareNetwork(state)
        O = network[1, 1]

        # project input
        C = _project_hermitian(C)
        E = _project_hermitian(E)

        # assuming 'eigenvalue decomposition' of enlarged corner as ECE = U * S * V
        # where now sectretly V = U†
        U = Ufp + ULfp * u # left isometry
        Ep = physical_flip(E) # edge (west or south)

        # then the 'right projector' (which goes on the left side) is U†
        Ud = U'

        # evaluate partial contractions to reuse
        # north edge and its left projector
        EnVd = contract_EnVd(E, U, O)
        EwCEnVd = contract_EwCEnVd(Ep, C, EnVd)

        # F1: corner
        C´ = Ud * EwCEnVd
        C´ = _project_hermitian(C´) # project output
        λ_C = dot(C, C´)
        F1 = C´ / λ_C - C

        # F2: edge
        E´ = contract_E´(Ud, EnVd)
        E´ = _project_hermitian(E´) # project output
        λ_E = dot(E, E´)
        F2 = E´ / λ_E - E

        # F3: u
        ULdEwCEnVd = ULd * EwCEnVd
        F3 = (ULdEwCEnVd * iC) / λ_C - u

        return F1, F2, F3
    end

    return symmetric_characteristic_equation
end

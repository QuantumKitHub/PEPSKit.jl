# Projector contractions
# ----------------------

"""
$(SIGNATURES)

Contract the CTMRG left projector with the higher-dimensional subspace facing to the left.

```
     C  --  E_2    -- |~~|
     |       |        |V'| -- isqS --in
    E_1 --   A     -- |~~|
     |       |
    out
```
"""
function left_projector(E_1, C, E_2, V, isqS, A::PEPSSandwich)
    return @autoopt @tensor P_left[χ_out D_outabove D_outbelow; χ_in] :=
        E_1[χ_out D1 D2; χ1] * C[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A)[d; D3 D5 D_outabove D1] * conj(bra(A)[d; D4 D6 D_outbelow D2]) *
        conj(V[χ4; χ3 D5 D6]) * isqS[χ4; χ_in]
end
function left_projector(E_1, C, E_2, V, isqS, A::PFTensor)
    return @autoopt @tensor P_left[χ_out D_out; χ_in] :=
        E_1[χ_out D1; χ1] * C[χ1; χ2] * E_2[χ2 D2; χ3] *
        A[D1 D_out; D2 D3] * conj(V[χ4; χ3 D3]) * isqS[χ4; χ_in]
end

"""
$(SIGNATURES)

Contract the CTMRG right projector with the higher-dimensional subspace facing to the right.

```
                  |~~| --   E_2   --  C
    out-- isqS -- |U'|      |         |
                  |~~| --   A     -- E_1
                            |         |
                                      in
```
"""
function right_projector(E_1, C, E_2, U, isqS, A::PEPSSandwich)
    return @autoopt @tensor P_right[χ_out; χ_in D_inabove D_inbelow] :=
        isqS[χ_out; χ1] * conj(U[χ1; χ2 D1 D2]) *
        ket(A)[d; D3 D5 D_inabove D1] * conj(bra(A)[d; D4 D6 D_inbelow D2]) *
        E_2[χ2 D3 D4; χ3] * C[χ3; χ4] * E_1[χ4 D5 D6; χ_in]
end
function right_projector(E_1, C, E_2, U, isqS, A::PFTensor)
    return @autoopt @tensor P_right[χ_out; χ_in D_in] :=
        isqS[χ_out; χ1] * conj(U[χ1; χ2 D1]) *
        A[D1 D_in; D2 D3] *
        E_2[χ2 D2; χ3] * C[χ3; χ4] * E_1[χ4 D3; χ_in]
end

"""
$(SIGNATURES)

Compute projectors based on a SVD of `Q * Q_next`, where the inverse square root
`isqS` of the singular values is computed.

Left projector:
```
    -- |~~~~~~| -- |~~|
       |Q_next|    |V'| -- isqS --
    == |~~~~~~| == |~~|
```

Right projector:
```
               |~~| -- |~~~| --
    -- isqS -- |U'|    | Q |
               |~~| == |~~~| ==
```
"""
function contract_projectors(U, S, V, Q, Q_next)
    isqS = sdiag_pow(S, -0.5)
    P_left = Q_next * V' * isqS  # use * to respect fermionic case
    P_right = isqS * U' * Q
    return P_left, P_right
end

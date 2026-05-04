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
function left_projector(E_1, C, E_2, V, isqS, A)
    Vdt_isqS = twistdual(V', 1:(numind(V) - 1)) * isqS
    return _left_projector(E_1, C, E_2, Vdt_isqS, A)
end
function _left_projector(E_1, C, E_2, Vd, A::PEPSSandwich)
    return @autoopt @tensor P_left[χ_out D_outabove D_outbelow; χ_in] :=
        E_1[χ_out D1 D2; χ1] * C[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A)[d; D3 D5 D_outabove D1] * conj(bra(A)[d; D4 D6 D_outbelow D2]) *
        Vd[χ3 D5 D6; χ_in]
end
function _left_projector(E_1, C, E_2, Vd, A::PFTensor)
    return @autoopt @tensor P_left[χ_out D_out; χ_in] :=
        E_1[χ_out D1; χ1] * C[χ1; χ2] * E_2[χ2 D2; χ3] *
        A[D1 D_out; D2 D3] * Vd[χ3 D3; χ_in]
end
@generated function _left_projector(
        E_1, C, E_2, Vd, A::PEPOSandwich{H}
    ) where {H}

    E_west_e = _pepo_edge_expr(:E_1, :out, :WNW, :W, H)
    E_north_e = _pepo_edge_expr(:E_2, :NNW, :NE, :N, H)
    C_northwest_e = _corner_expr(:C, :WNW, :NNW)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)

    Vd_e = _pepo_domain_projector_expr(:Vd, :NE, :E, :in, H)

    P_left_e = _pepo_domain_projector_expr(:P_left, :out, :S, :in, H)

    rhs = Expr(
        :call, :*,
        E_west_e, C_northwest_e, E_north_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        Vd_e
    )

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $P_left_e := $rhs)
    )
end

"""
$(SIGNATURES)

Contract the CTMRG left projector with the higher-dimensional subspace facing to the left.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
    out            [~~~V'~]
                       |
                     isqS
                       |
                       in
```
"""
function left_projector(E_1, C_1, E_2, E_3, C_2, E_4, V, isqS, A_1, A_2)
    Vdt_isqS = twistdual(V', 1:(numind(V) - 1)) * isqS
    return _left_projector(E_1, C_1, E_2, E_3, C_2, E_4, Vdt_isqS, A_1, A_2)
end
function _left_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Vd, A_1::PEPSSandwich, A_2::PEPSSandwich
    )
    return @autoopt @tensor P_left[χ_out D_outa D_outb; χ_in] :=
        E_1[χ_out D_W1a D_W1b; χ_WNW] * C_1[χ_WNW; χ_NNW] * E_2[χ_NNW D_N1a D_N1b; χ_N] *
        ket(A_1)[d1; D_N1a D_Ca D_outa D_W1a] * conj(bra(A_1)[d1; D_N1b D_Cb D_outb D_W1b]) *
        E_3[χ_N D_N2a D_N2b; χ_NNE] * C_2[χ_NNE; χ_ENE] * E_4[χ_ENE D_E2a D_E2b; χ_SE] *
        ket(A_2)[d2; D_N2a D_E2a D_S2a D_Ca] * conj(bra(A_2)[d2; D_N2b D_E2b D_S2b D_Cb]) *
        Vd[χ_SE D_S2a D_S2b; χ_in]
end
function _left_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Vd, A_1::PFTensor, A_2::PFTensor
    )
    return @autoopt @tensor P_left[χ_out D_out; χ_in] :=
        E_1[χ_out D_W1; χ_WNW] * C_1[χ_WNW; χ_NNW] * E_2[χ_NNW D_N1; χ_N] *
        A_1[D_W1 D_out; D_N1 D_C] *
        E_3[χ_N D_N2; χ_NNE] * C_2[χ_NNE; χ_ENE] * E_4[χ_ENE D_E2; χ_SE] *
        A_2[D_C D_S2; D_N2 D_E2] *
        Vd[χ_SE D_S2; χ_in]
end
@generated function _left_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Vd, A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}
    ) where {H}

    Vd_e = _pepo_domain_projector_expr(:Vd, :SE, :S, :in, H, 2)
    proj_expr = _half_infinite_environment_expr(H)
    P_left_e = _pepo_domain_projector_expr(:P_left, :SW, :S, :in, H, 1)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $P_left_e := $proj_expr * $Vd_e)
    )
end

"""
$(SIGNATURES)

Contract the CTMRG right projector with the higher-dimensional subspace facing to the right.

```
                  |~~| --   E_1   --  C
    out-- isqS -- |U'|      |         |
                  |~~| --   A     -- E_2
                            |         |
                                      in
```
"""
function right_projector(E_1, C, E_2, U, isqS, A)
    isqS_Udt = isqS * twistnondual(U', 2:numind(U))
    return _right_projector(E_1, C, E_2, isqS_Udt, A)
end
function _right_projector(E_1, C, E_2, Ud, A::PEPSSandwich)
    return @autoopt @tensor P_right[χ_out; χ_in D_inabove D_inbelow] :=
        Ud[χ_out; χ2 D1 D2] *
        ket(A)[d; D3 D5 D_inabove D1] * conj(bra(A)[d; D4 D6 D_inbelow D2]) *
        E_1[χ2 D3 D4; χ3] * C[χ3; χ4] * E_2[χ4 D5 D6; χ_in]
end
function _right_projector(E_1, C, E_2, Ud, A::PFTensor)
    return @autoopt @tensor P_right[χ_out; χ_in D_in] :=
        Ud[χ_out; χ2 D1] *
        A[D1 D_in; D2 D3] *
        E_1[χ2 D2; χ3] * C[χ3; χ4] * E_2[χ4 D3; χ_in]
end
@generated function _right_projector(
        E_1, C, E_2, Ud, A::PEPOSandwich{H}
    ) where {H}

    E_north_e = _pepo_edge_expr(:E_1, :NW, :NNE, :N, H)
    E_east_e = _pepo_edge_expr(:E_2, :ENE, :in, :E, H)
    C_northeast_e = _corner_expr(:C, :NNE, :ENE)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:A, H)

    Ud_e = _pepo_codomain_projector_expr(:Ud, :out, :NW, :W, H)

    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :in, :S, H)

    rhs = Expr(
        :call, :*,
        E_north_e, C_northeast_e, E_east_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
        Ud_e
    )

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $P_right_e := $rhs)
    )
end

"""
$(SIGNATURES)

Contract the CTMRG left projector with the higher-dimensional subspace facing to the left.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
     [~~~U'~]             in            
         |
        isqS
         |
        out
```
"""
function right_projector(E_1, C_1, E_2, E_3, C_2, E_4, U, isqS, A_1, A_2)
    isqS_Udt = isqS * twistnondual(U', 2:numind(U))
    return _right_projector(E_1, C_1, E_2, E_3, C_2, E_4, isqS_Udt, A_1, A_2)
end
function _right_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Ud, A_1::PEPSSandwich, A_2::PEPSSandwich
    )
    return @autoopt @tensor P_right[χ_out; χ_in D_ina D_inb] :=
        E_1[χ_SW D_W1a D_W1b; χ_WNW] * C_1[χ_WNW; χ_NNW] * E_2[χ_NNW D_N1a D_N1b; χ_N] *
        ket(A_1)[d1; D_N1a D_Ca D_S1a D_W1a] * conj(bra(A_1)[d1; D_N1b D_Cb D_S1b D_W1b]) *
        E_3[χ_N D_N2a D_N2b; χ_NNE] * C_2[χ_NNE; χ_ENE] * E_4[χ_ENE D_E2a D_E2b; χ_in] *
        ket(A_2)[d2; D_N2a D_E2a D_ina D_Ca] * conj(bra(A_2)[d2; D_N2b D_E2b D_inb D_Cb]) *
        Ud[χ_out; χ_SW D_S1a D_S1b]
end
function _right_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Ud, A_1::PFTensor, A_2::PFTensor
    )
    return @autoopt @tensor P_right[χ_out; χ_in D_in] :=
        E_1[χ_SW D_W1; χ_WNW] * C_1[χ_WNW; χ_NNW] * E_2[χ_NNW D_N1; χ_N] *
        A_1[D_W1 D_S1; D_N1 D_C] *
        E_3[χ_N D_N2; χ_NNE] * C_2[χ_NNE; χ_ENE] * E_4[χ_ENE D_E2; χ_in] *
        A_2[D_C D_in; D_N2 D_E2] *
        Ud[χ_out; χ_SW D_S1]
end
@generated function _right_projector(
        E_1, C_1, E_2, E_3, C_2, E_4, Ud, A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}
    ) where {H}

    Ud_e = _pepo_codomain_projector_expr(:Ud, :out, :SW, :S, H, 1)
    proj_expr = _half_infinite_environment_expr(H)
    P_right_e = _pepo_codomain_projector_expr(:P_right, :out, :SE, :S, H, 2)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $P_right_e := $Ud_e * $proj_expr)
    )
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

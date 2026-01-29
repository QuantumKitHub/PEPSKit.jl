"""
    half_infinite_environment(quadrant1, quadrant2)
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2)
    half_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x, A_1, A_2)
    half_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2)

Contract two quadrants (enlarged corners) to form a half-infinite environment.

```
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant1|    |quadrant2|
    |~~~~~~~~~| -- |~~~~~~~~~|
      |     |        |     |
```

The environment can also be contracted directly from all its constituent tensors.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
```

Alternatively, contract the environment with a vector `x` acting on it

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
                  [~~~x~~~~]
```

or contract the adjoint environment with `x`, e.g. as needed for iterative solvers.
"""
function half_infinite_environment(
        quadrant1::AbstractTensorMap{T, S, N, N}, quadrant2::AbstractTensorMap{T, S, N, N}
    ) where {T, S, N}
    p = (codomainind(quadrant1), domainind(quadrant1))
    return tensorcontract(quadrant1, p, false, quadrant2, p, false, p)
end
function half_infinite_environment(
        C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
    ) where {P <: PEPSSandwich}
    return @autoopt @tensor env[χ_out D_outabove D_outbelow; χ_in D_inabove D_inbelow] :=
        E_1[χ_out D1 D2; χ1] * C_1[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D9 D_outabove D1] * conj(bra(A_1)[d1; D4 D10 D_outbelow D2]) *
        ket(A_2)[d2; D5 D7 D_inabove D9] * conj(bra(A_2)[d2; D6 D8 D_inbelow D10]) *
        E_3[χ3 D5 D6; χ4] * C_2[χ4; χ5] * E_4[χ5 D7 D8; χ_out]
end
function half_infinite_environment(
        C_1, C_2, E_1, E_2, E_3, E_4, x::AbstractTensor{T, S, 3}, A_1::P, A_2::P
    ) where {T, S, P <: PEPSSandwich}
    return @autoopt @tensor env_x[χ_out D_outabove D_outbelow] :=
        E_1[χ_out D1 D2; χ1] * C_1[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D9 D_outabove D1] * conj(bra(A_1)[d1; D4 D10 D_outbelow D2]) *
        ket(A_2)[d2; D5 D7 D11 D9] * conj(bra(A_2)[d2; D6 D8 D12 D10]) *
        E_3[χ3 D5 D6; χ4] * C_2[χ4; χ5] * E_4[χ5 D7 D8; χ6] *
        x[χ6 D11 D12]
end
function half_infinite_environment(
        x::AbstractTensor{T, S, 3}, C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
    ) where {T, S, P <: PEPSSandwich}
    return @autoopt @tensor x_env[χ_in D_inabove D_inbelow] :=
        x[χ1 D1 D2] *
        conj(E_1[χ1 D3 D4; χ2]) * conj(C_1[χ2; χ3]) * conj(E_2[χ3 D5 D6; χ4]) *
        conj(ket(A_1)[d1; D5 D11 D1 D3]) * bra(A_1)[d1; D6 D12 D2 D4] *
        conj(ket(A_2)[d2; D7 D9 D_inabove D11]) * bra(A_2)[d2; D8 D10 D_inbelow D12] *
        conj(E_3[χ4 D7 D8; χ5]) * conj(C_2[χ5; χ6]) * conj(E_4[χ6 D9 D10; χ_in])
end
function half_infinite_environment(
        C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
    ) where {P <: PFTensor}
    return @autoopt @tensor env[χ_out D_out; χ_in D_in] :=
        E_1[χ_out D1; χ1] * C_1[χ1; χ2] * E_2[χ2 D3; χ3] *
        A_1[D1 D_out; D3 D9] *
        A_2[D9 D_in; D5 D7] *
        E_3[χ3 D5; χ4] * C_2[χ4; χ5] * E_4[χ5 D7; χ_in]
end
function half_infinite_environment(
        C_1, C_2, E_1, E_2, E_3, E_4, x::AbstractTensor{T, S, 2}, A_1::P, A::P
    ) where {T, S, P <: PFTensor}
    return @autoopt @tensor env_x[χ_out D_out] :=
        E_1[χ_out D1; χ1] * C_1[χ1; χ2] * E_2[χ2 D3; χ3] *
        A_1[D1 D_out; D3 D9] *
        A_2[D9 D11; D5 D7] *
        E_3[χ3 D5; χ4] * C_2[χ4; χ5] * E_4[χ5 D7; χ6] *
        x[χ6 D11]
end
function half_infinite_environment(
        x::AbstractTensor{T, S, 2}, C_1, C_2, E_1, E_2, E_3, E_4, A_1::P, A_2::P
    ) where {T, S, P <: PFTensor}
    return @autoopt @tensor env_x[χ_in D_in] :=
        x[χ1 D1 D2] *
        conj(E_1[χ1 D3; χ2]) * conj(C_1[χ2; χ3]) * conj(E_2[χ3 D5; χ4]) *
        conj(A_1[D3 D1; D5 D11]) *
        conj(A_2[D11 D_in; D7 D9]) *
        conj(E_3[χ4 D7; χ5]) * conj(C_2[χ5; χ6]) * conj(E_4[χ6 D9; χ_in])
end

## HalfInfiniteEnvironment contractions

# reuse partial multiplication expression; TODO: return quadrants separately?
function _half_infinite_environnment_expr(H)
    # site 1 (codomain)
    C1_e = _corner_expr(:C_1, :WNW, :NNW)
    E1_e = _pepo_edge_expr(:E_1, :SW, :WNW, :W, H, 1)
    E2_e = _pepo_edge_expr(:E_2, :NNW, :NC, :N, H, 1)
    ket1_e, bra1_e, pepo1_es = _pepo_sandwich_expr(:A_1, H, 1; contract_east = :NC)

    # site 2 (domain)
    C2_e = _corner_expr(:C_2, :NNE, :ENE)
    E3_e = _pepo_edge_expr(:E_3, :NC, :NNE, :N, H, 2)
    E4_e = _pepo_edge_expr(:E_4, :ENE, :SE, :E, H, 2)
    ket2_e, bra2_e, pepo2_es = _pepo_sandwich_expr(:A_2, H, 2; contract_west = :NC)

    partial_expr = Expr(
        :call, :*,
        E1_e, C1_e, E2_e,
        ket1_e, Expr(:call, :conj, bra1_e),
        pepo1_es...,
        E3_e, C2_e, E4_e,
        ket2_e, Expr(:call, :conj, bra2_e),
        pepo2_es...,
    )

    return partial_expr
end

@generated function half_infinite_environment(
        C_1, C_2, E_1, E_2, E_3, E_4, A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}
    ) where {H}
    # return projector expression
    env_e = _pepo_env_expr(:env, :SW, :SE, :S, :S, 1, 2, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_e := $rhs))
end
@generated function half_infinite_environment(
        C_1, C_2,
        E_1, E_2, E_3, E_4,
        x::AbstractTensor{T, S, N},
        A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    # codomain vector (output)
    env_x_e = _pepo_env_arg_expr(:env_x, :SW, :S, 1, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    # domain vector (input)
    x_e = _pepo_env_arg_expr(:x, :SE, :S, 2, H)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $env_x_e := $proj_expr * $x_e)
    )
end
@generated function half_infinite_environment(
        x::AbstractTensor{T, S, N},
        C_1, C_2,
        E_1, E_2, E_3, E_4,
        A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3
    # codomain vector (input)
    x_e = _pepo_env_arg_expr(:x, :SW, :S, 1, H)

    # reuse partial multiplication expression
    proj_expr = _half_infinite_environnment_expr(H)

    # domain vector (output)
    x_env_e = _pepo_env_arg_expr(:env_x, :SE, :S, 2, H)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $x_env_e := $x_e * $proj_expr)
    )
end

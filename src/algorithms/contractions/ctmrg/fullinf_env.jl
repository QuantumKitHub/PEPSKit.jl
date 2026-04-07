"""
    full_infinite_environment(quadrant1, quadrant2, quadrant3, quadrant4)
    full_infinite_environment(half1, half2)
    full_infinite_environment(C_1, C_2, C_3, C_4, E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8, A_1, A_2, A_3, A_4)
    full_infinite_environment(C_1, C_2, E_1, E_2, E_3, E_4, x, A_1, A_2, A_3, A_4)
    full_infinite_environment(x, C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2, A_3, A_4)

Contract four quadrants (enlarged corners) to form a full-infinite environment.

```
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant1|    |quadrant2|
    |~~~~~~~~~| -- |~~~~~~~~~|
      |     |        |     |
     out             |     |
                     |     |
     in              |     |
      |     |        |     |
    |~~~~~~~~~| -- |~~~~~~~~~|
    |quadrant4|    |quadrant3|
    |~~~~~~~~~| -- |~~~~~~~~~|
```

In the same manner two halfs can be used to contract the full-infinite environment.

```
    |~~~~~~~~~~~~~~~~~~~~~~~~|
    |         half1          |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
      |     |        |     |
     out             |     |
                     |     |
     in              |     |
      |     |        |     |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
    |         half2          |
    |~~~~~~~~~~~~~~~~~~~~~~~~|
```

The environment can also be contracted directly from all its constituent tensors.

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
    out            |      |
                   |      |
    in      |      |      |
     |      |      |      |
    E_8 -- A_4 -- A_3 -- E_5
     |      |      |      |
    C_4 -- E_7 -- E_6 -- C_3
```

Alternatively, contract the environment with a vector `x` acting on it

```
    C_1 -- E_2 -- E_3 -- C_2
     |      |      |      |
    E_1 -- A_1 -- A_2 -- E_4
     |      |      |      |
    out            |      |
                   |      |
    [~~~~x~~~]     |      |
     |      |      |      |
    E_8 -- A_4 -- A_3 -- E_5
     |      |      |      |
    C_4 -- E_7 -- E_6 -- C_3

```

or contract the adjoint environment with `x`, e.g. as needed for iterative solvers.
"""
@generated function full_infinite_environment(
        quadrant1::AbstractTensorMap{T, S, N, N}, quadrant2::AbstractTensorMap{T, S, N, N},
        quadrant3::AbstractTensorMap{T, S, N, N}, quadrant4::AbstractTensorMap{T, S, N, N},
    ) where {T, S, N}
    env_e = tensorexpr(
        :env,
        (envlabel(:out), ntuple(i -> virtuallabel(:out, i), N - 1)...),
        (envlabel(:in), ntuple(i -> virtuallabel(:in, i), N - 1)...),
    )
    quadrant1_e = tensorexpr(
        :quadrant1,
        (envlabel(:out), ntuple(i -> virtuallabel(:out, i), N - 1)...),
        (envlabel(:NC), ntuple(i -> virtuallabel(:NC, i), N - 1)...),
    )
    quadrant2_e = tensorexpr(
        :quadrant2,
        (envlabel(:NC), ntuple(i -> virtuallabel(:NC, i), N - 1)...),
        (envlabel(:EC), ntuple(i -> virtuallabel(:EC, i), N - 1)...),
    )
    quadrant3_e = tensorexpr(
        :quadrant3,
        (envlabel(:EC), ntuple(i -> virtuallabel(:EC, i), N - 1)...),
        (envlabel(:SC), ntuple(i -> virtuallabel(:SC, i), N - 1)...),
    )
    quadrant4_e = tensorexpr(
        :quadrant4,
        (envlabel(:SC), ntuple(i -> virtuallabel(:SC, i), N - 1)...),
        (envlabel(:in), ntuple(i -> virtuallabel(:in, i), N - 1)...),
    )
    return macroexpand(
        @__MODULE__,
        :(
            return @autoopt @tensor $env_e :=
                $quadrant1_e * $quadrant2_e * $quadrant3_e * $quadrant4_e
        ),
    )
end
function full_infinite_environment(
        half1::AbstractTensorMap{T, S, N}, half2::AbstractTensorMap{T, S, N}
    ) where {T, S, N}
    return half_infinite_environment(half1, half2)
end
function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {P <: PEPSSandwich}
    return @autoopt @tensor env[χ_out D_outabove D_outbelow; χ_in D_inabove D_inbelow] :=
        E_1[χ_out D1 D2; χ1] * C_1[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D11 D_outabove D1] * conj(bra(A_1)[d1; D4 D12 D_outbelow D2]) *
        ket(A_2)[d2; D5 D7 D9 D11] * conj(bra(A_2)[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] * C_2[χ4; χ5] * E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] * C_3[χ7; χ8] * E_6[χ8 D15 D16; χ9] *
        ket(A_3)[d3; D9 D13 D15 D17] * conj(bra(A_3)[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_inabove D17 D19 D21] * conj(bra(A_4)[d4; D_inbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] * C_4[χ10; χ11] * E_8[χ11 D21 D22; χ_in]
end
function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        x::AbstractTensor{T, S, 3},
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {T, S, P <: PEPSSandwich}
    return @autoopt @tensor env_x[χ_out D_outabove D_outbelow] :=
        E_1[χ_out D1 D2; χ1] * C_1[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D11 D_outabove D1] * conj(bra(A_1)[d1; D4 D12 D_outbelow D2]) *
        ket(A_2)[d2; D5 D7 D9 D11] * conj(bra(A_2)[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] * C_2[χ4; χ5] * E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] * C_3[χ7; χ8] * E_6[χ8 D15 D16; χ9] *
        ket(A_3)[d3; D9 D13 D15 D17] * conj(bra_3[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_xabove D17 D19 D21] * conj(bra(A_4)[d4; D_xbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] * C_4[χ10; χ11] * E_8[χ11 D21 D22; χ_x] *
        x[χ_x D_xabove D_xbelow]
end
function full_infinite_environment(
        x::AbstractTensor{T, S, 3},
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {T, S, P <: PEPSSandwich}
    return @autoopt @tensor x_env[χ_in D_inabove D_inbelow] :=
        x[χ_x D_xabove D_xbelow] *
        E_1[χ_x D1 D2; χ1] * C_1[χ1; χ2] * E_2[χ2 D3 D4; χ3] *
        ket(A_1)[d1; D3 D11 D_xabove D1] * conj(bra(A_1)[d1; D4 D12 D_xbelow D2]) *
        ket(A_2)[d2; D5 D7 D9 D11] * conj(bra(A_2)[d2; D6 D8 D10 D12]) *
        E_3[χ3 D5 D6; χ4] * C_2[χ4; χ5] * E_4[χ5 D7 D8; χ6] *
        E_5[χ6 D13 D14; χ7] * C_3[χ7; χ8] * E_6[χ8 D15 D16; χ9] *
        ket(A_3)[d3; D9 D13 D15 D17] * conj(bra(A_3)[d3; D10 D14 D16 D18]) *
        ket(A_4)[d4; D_inabove D17 D19 D21] * conj(bra(A_4)[d4; D_inbelow D18 D20 D22]) *
        E_7[χ9 D19 D20; χ10] * C_4[χ10; χ11] * E_8[χ11 D21 D22; χ_in]
end
function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {P <: PFTensor}
    return @autoopt @tensor env[χ_out D_out; χ_in D_in] :=
        E_1[χ_out D1; χ1] * C_1[χ1; χ2] * E_2[χ2 D3; χ3] *
        A_1[D1 D_out; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] * C_2[χ4; χ5] * E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] * C_3[χ7; χ8] * E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_in D17] *
        E_7[χ9 D19; χ10] * C_4[χ10; χ11] * E_8[χ11 D21; χ_in]
end
function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        x::AbstractTensor{T, S, 2},
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {T, S, P <: PFTensor}
    return @autoopt @tensor env_x[χ_out D_out] :=
        E_1[χ_out D1; χ1] * C_1[χ1; χ2] * E_2[χ2 D3; χ3] *
        A_1[D1 D_out; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] * C_2[χ4; χ5] * E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] * C_3[χ7; χ8] * E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_x D17] *
        E_7[χ9 D19; χ10] * C_4[χ10; χ11] * E_8[χ11 D21; χ_x] *
        x[χ_x D_x]
end
function full_infinite_environment(
        x::AbstractTensor{T, S, 2},
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::P, A_2::P, A_3::P, A_4::P,
    ) where {T, S, P <: PFTensor}
    return @autoopt @tensor x_env[χ_in D_in] :=
        x[χ_x D_x] *
        E_1[χ_x D1; χ1] * C_1[χ1; χ2] * E_2[χ2 D3; χ3] *
        A_1[D1 D_x; D3 D11] *
        A_2[D11 D9; D5 D7] *
        E_3[χ3 D5; χ4] * C_2[χ4; χ5] * E_4[χ5 D7; χ6] *
        E_5[χ6 D13; χ7] * C_3[χ7; χ8] * E_6[χ8 D15; χ9] *
        A_3[D17 D15; D9 D13] *
        A_4[D21 D19; D_in D17] *
        E_7[χ9 D19; χ10] * C_4[χ10; χ11] * E_8[χ11 D21; χ_in]
end


## FullInfiniteEnvironment contractions

# reuse partial multiplication expression; TODO: return quadrants separately?
function _full_infinite_environment_expr(H)
    # site 1 (codomain)
    C1_e = _corner_expr(:C_1, :WNW, :NNW)
    E1_e = _pepo_edge_expr(:E_1, :SW, :WNW, :W, H, 1)
    E2_e = _pepo_edge_expr(:E_2, :NNW, :NC, :N, H, 1)
    ket1_e, bra1_e, pepo1_es = _pepo_sandwich_expr(:A_1, H, 1; contract_east = :NC)

    # site 2
    C2_e = _corner_expr(:C_2, :NNE, :ENE)
    E3_e = _pepo_edge_expr(:E_3, :NC, :NNE, :N, H, 2)
    E4_e = _pepo_edge_expr(:E_4, :ENE, :EC, :E, H, 2)
    ket2_e, bra2_e, pepo2_es = _pepo_sandwich_expr(
        :A_2, H, 2; contract_west = :NC, contract_south = :EC
    )

    # site 3
    C3_e = _corner_expr(:C_3, :WSW, :SSW)
    E5_e = _pepo_edge_expr(:E_5, :EC, :WSW, :E, H, 3)
    E6_e = _pepo_edge_expr(:E_6, :SSW, :SC, :S, H, 3)
    ket3_e, bra3_e, pepo3_es = _pepo_sandwich_expr(
        :A_3, H, 3; contract_north = :EC, contract_west = :SC
    )

    # site 4 (domain)
    C4_e = _corner_expr(:C_4, :SSW, :WSW)
    E7_e = _pepo_edge_expr(:E_7, :SC, :SSW, :S, H, 4)
    E8_e = _pepo_edge_expr(:E_8, :WSW, :NW, :W, H, 4)
    ket4_e, bra4_e, pepo4_es = _pepo_sandwich_expr(:A_4, H, 4; contract_east = :SC)

    partial_expr = Expr(
        :call, :*,
        E1_e, C1_e, E2_e,
        ket1_e, Expr(:call, :conj, bra1_e),
        pepo1_es...,
        E3_e, C2_e, E4_e,
        ket2_e, Expr(:call, :conj, bra2_e),
        pepo2_es...,
        E5_e, C3_e, E6_e,
        ket3_e, Expr(:call, :conj, bra3_e),
        pepo3_es...,
        E7_e, C4_e, E8_e,
        ket4_e, Expr(:call, :conj, bra4_e),
        pepo4_es...,
    )

    return partial_expr
end

@generated function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}, A_3::PEPOSandwich{H}, A_4::PEPOSandwich{H},
    ) where {H}
    # return projector expression
    env_e = _pepo_env_expr(:env, :SW, :NW, :S, :N, 1, 4, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_e := $proj_expr))
end
@generated function full_infinite_environment(
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        x::AbstractTensor{T, S, N},
        A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}, A_3::PEPOSandwich{H}, A_4::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    # codomain vecor (output)
    env_x_e = _pepo_env_arg_expr(:env, :SW, :S, 1, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    # domain vector (input)
    x_e = _pepo_env_arg_expr(:env, :NW, :N, 4, N - 1)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $env_x_e := $proj_expr * x_e))
end
@generated function full_infinite_environment(
        x::AbstractTensor{T, S, N},
        C_1, C_2, C_3, C_4,
        E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
        A_1::PEPOSandwich{H}, A_2::PEPOSandwich{H}, A_3::PEPOSandwich{H}, A_4::PEPOSandwich{H},
    ) where {T, S, N, H}
    @assert N == H + 3

    # codomain vecor (input)
    x_e = _pepo_env_arg_expr(:env, :SW, :S, 1, N - 1)

    # reuse partial multiplication expression
    proj_expr = _full_infinite_environment_expr(H)

    # domain vector (output)
    x_env_e = _pepo_env_arg_expr(:env, :NW, :N, 4, N - 1)

    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor $x_env_e := $x_e * $proj_expr)
    )
end

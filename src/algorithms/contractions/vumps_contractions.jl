using MPSKit: GenericMPSTensor, MPSBondTensor

#
# Environment transfer functions
#

## PEPS

function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @autoopt @tensor GL′[χ_SE D_E_above D_E_below; χ_NE] :=
        GL[χ_SW D_W_above D_W_below; χ_NW] *
        conj(Ā[χ_SW D_S_above D_S_below; χ_SE]) *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below]) *
        A[χ_NW D_N_above D_N_below; χ_NE]
end

function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    A::GenericMPSTensor{S,3},
    Ā::GenericMPSTensor{S,3},
) where {S}
    return @autoopt @tensor GR′[χ_NW D_W_above D_W_below; χ_SW] :=
        GR[χ_NE D_E_above D_E_below; χ_SE] *
        conj(Ā[χ_SW D_S_above D_S_below; χ_SE]) *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below]) *
        A[χ_NW D_N_above D_N_below χ_NE]
end

## PEPO

# some plumbing for generic expressions...

# side=:W for argument, side=:E for output, PEPO height H
function _pepo_leftenv_expr(envname, side::Symbol, H::Int)
    return tensorexpr(
        envname,
        (
            envlabel(:S, side),
            virtuallabel(side, :top),
            ntuple(i -> virtuallabel(side, :mid, i), H)...,
            virtuallabel(side, :bot),
        ),
        (envlabel(:N, side),),
    )
end

# side=:E for argument, side=:W for output, PEPO height H
function _pepo_rightenv_expr(envname, side::Symbol, H::Int)
    return tensorexpr(
        envname,
        (
            envlabel(:N, side),
            virtuallabel(side, :top),
            ntuple(i -> virtuallabel(side, :mid, i), H)...,
            virtuallabel(side, :bot),
        ),
        (envlabel(:S, side),),
    )
end

# side=:N for ket MPS, side=:S for bra MPS, PEPO height H
function _pepo_mpstensor_expr(tensorname, side::Symbol, H::Int)
    return tensorexpr(
        tensorname,
        (
            envlabel(side, :W),
            virtuallabel(side, :top),
            ntuple(i -> virtuallabel(side, :mid, i), H)...,
            virtuallabel(side, :bot),
        ),
        (envlabel(side, :E),),
    )
end

# layer=:top for ket PEPS, layer=:bot for bra PEPS, connects to PEPO slice H
function _pepo_pepstensor_expr(tensorname, layer::Symbol, h::Int)
    return tensorexpr(
        tensorname,
        (physicallabel(h),),
        (
            virtuallabel(:N, layer),
            virtuallabel(:E, layer),
            virtuallabel(:S, layer),
            virtuallabel(:W, layer),
        ),
    )
end

# PEPO slice h
function _pepo_pepotensor_expr(tensorname, h::Int)
    return tensorexpr(
        tensorname,
        (physicallabel(h + 1), physicallabel(h)),
        (
            virtuallabel(:N, :mid, h),
            virtuallabel(:E, :mid, h),
            virtuallabel(:S, :mid, h),
            virtuallabel(:W, :mid, h),
        ),
    )
end

# specialize simple case
function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S}
    return @autoopt @tensor GL′[χ_SE D_E_above D_E_mid D_E_below; χ_NE] :=
        GL[χ_SW D_W_above D_W_mid D_W_below; χ_NW] *
        conj(Ā[χ_SW D_S_above D_S_mid D_S_below; χ_SE]) *
        ket(O)[d_in; D_N_above D_E_above D_S_above D_W_above] *
        only(pepo(O))[d_out d_in; D_N_mid D_E_mid D_S_mid D_W_mid] *
        conj(bra(O)[d_out; D_N_below D_E_below D_S_below D_W_below]) *
        A[χ_NW D_N_above D_N_mid D_N_below; χ_NE]
end

# general case
@generated function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    GL´_e = _pepo_leftenv_expr(:GL´, :E, H)
    GL_e = _pepo_leftenv_expr(:GL, :W, H)
    A_e = _pepo_mpstensor_expr(:A, :N, H)
    Ā_e = _pepo_mpstensor_expr(:Ā, :S, H)
    ket_e = _pepo_pepstensor_expr(:(ket(O)), :top, 1)
    bra_e = _pepo_pepstensor_expr(:(bra(O)), :bot, H + 1)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo(O)[$h]), h)
    end

    rhs = Expr(
        :call,
        :*,
        GL_e,
        A_e,
        Expr(:call, :conj, Ā_e),
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(@autoopt @tensor $GL´_e := $rhs))
end

# specialize simple case
function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    A::GenericMPSTensor{S,4},
    Ā::GenericMPSTensor{S,4},
) where {S}
    return @tensor GR′[χ_NW D_W_above D_W_mid D_W_below; χ_SW] :=
        GR[χ_NE D_E_above D_E_mid D_E_below; χ_SE] *
        conj(Ā[χ_SW D_S_above D_S_mid D_S_below; χ_SE]) *
        ket(O)[d_in; D_N_above D_E_above D_S_above D_W_above] *
        only(pepo(O))[d_out d_in; D_N_mid D_E_mid D_S_mid D_W_mid] *
        conj(bra(O)[d_out; D_N_below D_E_below D_S_below D_W_below]) *
        A[χ_NW D_N_above D_N_mid D_N_below; χ_NE]
end

# general case
function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    GR´_e = _pepo_rightenv_expr(:GR´, :W, H)
    GR_e = _pepo_rightenv_expr(:GR, :E, H)
    A_e = _pepo_mpstensor_expr(:A, :N, H)
    Ā_e = _pepo_mpstensor_expr(:Ā, :S, H)
    ket_e = _pepo_pepstensor_expr(:(ket(O)), :top, 1)
    bra_e = _pepo_pepstensor_expr(:(bra(O)), :bot, H + 1)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo(O)[$h]), h)
    end

    rhs = Expr(
        :call,
        :*,
        GR_e,
        A_e,
        Expr(:call, :conj, Ā_e),
        ket_e,
        Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(@autoopt @tensor $GR´_e := $rhs))
end

@generated function environment_overlap(
    GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    GL_e = tensorexpr(:GL, (N + 1, (2:N)...), 1)
    GR_e = tensorexpr(:GR, 1:N, N + 1)
    return macroexpand(@__MODULE__, :(return o = @tensor $GL_e * $GR_e))
end

# TODO: properly implement in the most efficient way
function MPSKit.contract_mpo_expval(
    AC::GenericMPSTensor{S,N},
    GL::GenericMPSTensor{S,N},
    O::Union{PEPSSandwich,PEPOSandwich},
    GR::GenericMPSTensor{S,N},
    ACbar::GenericMPSTensor{S,N}=AC,
) where {S,N}
    GL´ = MPSKit.transfer_left(GL, O, AC, ACbar)
    return environment_overlap(GL´, GR)
end

#
# Derivative contractions
#

@generated function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    C´_e = tensorexpr(:C´, -1, -2)
    C_e = tensorexpr(:C, 1, 2)
    GL_e = tensorexpr(:GL, (-1, (3:(N + 1))...), 1)
    GR_e = tensorexpr(:GR, 2:(N + 1), -2)
    return macroexpand(@__MODULE__, :(return @tensor $C´_e := $GL_e * $C_e * $GR_e))
end

## PEPS

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,3},
    O::PEPSSandwich,
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S}
    return @autoopt @tensor AC′[χ_SW D_S_above D_S_below; χ_SE] :=
        GL[χ_SW D_W_above D_W_below; χ_NW] *
        AC[χ_NW D_N_above D_N_below; χ_NE] *
        GR[χ_NE D_E_above D_E_below; χ_SE] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end

# PEPS derivative
function ∂peps(
    AC::GenericMPSTensor{S,3},
    ĀC::GenericMPSTensor{S,3},
    O::PEPSTensor{S},
    GL::GenericMPSTensor{S,3},
    GR::GenericMPSTensor{S,3},
) where {S}
    return @tensor ∂p[d; D_N_below D_E_below D_S_below D_W_below] :=
        GL[χ_SW D_W_above D_W_below; χ_NW] *
        AC[χ_NW D_N_above D_N_below; χ_NE] *
        O[d; D_N_above D_E_above D_S_above D_W_above] *
        GR[χ_NE D_E_above D_E_below; χ_SE] *
        conj(ĀC[χ_SW D_S_above D_S_below; χ_SE])
end

## PEPO

# specialize simple case
function MPSKit.∂AC(
    AC::GenericMPSTensor{S,4},
    O::PEPOSandwich{1},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S}
    return @tensor AC′[χ_SW D_S_above D_S_mid D_S_below; χ_SE] :=
        GL[χ_SW D_W_above D_W_mid D_W_below; χ_NW] *
        AC[χ_NW D_N_above D_N_mid D_N_below; χ_NE] *
        GR[χ_NE D_E_above D_E_mid D_E_below; χ_SE] *
        ket(O)[d_in; D_N_above D_E_above D_S_above D_W_above] *
        only(pepo(O))[d_out d_in; D_N_mid D_E_mid D_S_mid D_W_mid] *
        conj(bra(O)[d_out; D_N_below D_E_below D_S_below D_W_below])
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    AC´_e = _pepo_mpstensor_expr(:AC´, :S, H)
    AC_e = _pepo_mpstensor_expr(:AC, :N, H)
    GL_e = _pepo_mpstensor_expr(:GL, :W, H)
    GR_e = _pepo_mpstensor_expr(:GR, :E, H)
    ket_e = _pepo_pepstensor_expr(:(ket(O)), :top, 1)
    bra_e = _pepo_pepstensor_expr(:(bra(O)), :bot, H + 1)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo(O)[$h]), h)
    end

    rhs = Expr(:call, :*, AC_e, GL_e, GR_e, ket_e, Expr(:call, :conj, bra_e), pepo_es...)

    return macroexpand(@__MODULE__, :(@autoopt @tensor $AC´_e := $rhs))
end

# PEPS derivative

# sandwich with the bottom dropped out...
const ∂PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,Tuple{Vararg{P,N}}}
ket(p::∂PEPOSandwich) = p[1]
pepo(p::∂PEPOSandwich) = p[2]
pepo(p::∂PEPOSandwich, i::Int) = p[2][i]

# specialize simple case
function ∂peps(
    AC::GenericMPSTensor{S,4},
    ĀC::GenericMPSTensor{S,4},
    O::∂PEPOSandwich{1},
    GL::GenericMPSTensor{S,4},
    GR::GenericMPSTensor{S,4},
) where {S}
    return @tensor ∂p[d_out; D_N_below D_E_below D_S_below D_W_below] :=
        GL[χ_SW D_W_above D_W_mid D_W_below; χ_NW] *
        AC[χ_NW D_N_above D_N_mid D_N_below; χ_NE] *
        ket(O)[d_in; D_N_above D_E_above D_S_above D_W_above] *
        only(pepo(O))[d_out d_in; D_N_mid D_E_mid D_S_mid D_W_mid] *
        GR[χ_NE D_E_above D_E_mid D_E_below; χ_SE] *
        conj(ĀC[χ_SW D_S_above D_S_mid D_S_below; χ_SE])
end

function ∂peps(
    AC::GenericMPSTensor{S,N},
    ĀC::GenericMPSTensor{S,N},
    O::∂PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    ∂p_e = _pepo_pepstensor_expr(:∂p, :bot, H + 1)
    AC_e = _pepo_mpstensor_expr(:AC, :N, H)
    ĀC_e = _pepo_mpstensor_expr(:ĀC, :S, H)
    GL_e = _pepo_mpstensor_expr(:GL, :W, H)
    GR_e = _pepo_mpstensor_expr(:GR, :E, H)
    ket_e = _pepo_pepstensor_expr(:(ket(O)), :top, 1)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo(O)[$h]), h)
    end

    rhs = Expr(:call, :*, AC_e, Expr(:call, :conj, ĀC_e), GL_e, GR_e, ket_e, pepo_es...)

    return macroexpand(@__MODULE__, :(@autoopt @tensor $∂p_e := $rhs))
end

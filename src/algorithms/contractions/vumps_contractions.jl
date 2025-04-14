using MPSKit: GenericMPSTensor, MPSBondTensor

#
# Environment transfer functions
#

function MPSKit.transfer_left(
    GL::GenericMPSTensor{S,N}, O, A::GenericMPSTensor{S,N}, Ā::GenericMPSTensor{S,N}
) where {S,N}
    Ā = twistdual(Ā, 2:N)
    return _transfer_left(GL, O, A, Ā)
end

function MPSKit.transfer_right(
    GR::GenericMPSTensor{S,N}, O, A::GenericMPSTensor{S,N}, Ā::GenericMPSTensor{S,N}
) where {S,N}
    Ā = twistdual(Ā, 2:N)
    return _transfer_right(GR, O, A, Ā)
end

## PEPS

function _transfer_left(
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

function _transfer_right(
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

@generated function _transfer_left(
    GL::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    GL´_e = _pepo_edge_expr(:GL´, :SE, :NE, :E, H)
    GL_e = _pepo_edge_expr(:GL, :SW, :NW, :W, H)
    A_e = _pepo_edge_expr(:A, :NW, :NE, :N, H)
    Ā_e = _pepo_edge_expr(:Ā, :SW, :SE, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

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

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $GL´_e := $rhs))
end

@generated function _transfer_right(
    GR::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    A::GenericMPSTensor{S,N},
    Ā::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    GR´_e = _pepo_edge_expr(:GR´, :NW, :SW, :W, H)
    GR_e = _pepo_edge_expr(:GR, :NE, :SE, :E, H)
    A_e = _pepo_edge_expr(:A, :NW, :NE, :N, H)
    Ā_e = _pepo_edge_expr(:Ā, :SW, :SE, :S, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

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

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $GR´_e := $rhs))
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

function MPSKit.∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    GL = twistdual(GL, 1)
    GR = twistdual(GR, numind(GR))
    return _∂C(C, GL, GR)
end
@generated function _∂C(
    C::MPSBondTensor{S}, GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    C´_e = tensorexpr(:C´, -1, -2)
    C_e = tensorexpr(:C, 1, 2)
    GL_e = tensorexpr(:GL, (-1, (3:(N + 1))...), 1)
    GR_e = tensorexpr(:GR, 2:(N + 1), -2)
    return macroexpand(@__MODULE__, :(return @tensor $C´_e := $GL_e * $C_e * $GR_e))
end

function MPSKit.∂AC(
    AC::GenericMPSTensor{S,N}, O, GL::GenericMPSTensor{S,N}, GR::GenericMPSTensor{S,N}
) where {S,N}
    GL = twistdual(GL, 1)
    GR = twistdual(GR, numind(GR))
    return _∂AC(AC, O, GL, GR)
end

## PEPS

function _∂AC(
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

@generated function _∂AC(
    AC::GenericMPSTensor{S,N},
    O::PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    AC´_e = _pepo_edge_expr(:AC´, :SW, :SE, :S, H)
    AC_e = _pepo_edge_expr(:AC, :NW, :NE, :N, H)
    GL_e = _pepo_edge_expr(:GL, :SW, :NW, :W, H)
    GR_e = _pepo_edge_expr(:GR, :NE, :SE, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    rhs = Expr(:call, :*, AC_e, GL_e, GR_e, ket_e, Expr(:call, :conj, bra_e), pepo_es...)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $AC´_e := $rhs))
end

# PEPS derivative

# sandwich with the bottom dropped out...
const ∂PEPOSandwich{N,T<:PEPSTensor,P<:PEPOTensor} = Tuple{T,Vararg{P,N}}
ket(p::∂PEPOSandwich) = p[1]
pepo(p::∂PEPOSandwich) = p[2:end]
pepo(p::∂PEPOSandwich, i::Int) = p[1 + i]

@generated function ∂peps(
    AC::GenericMPSTensor{S,N},
    ĀC::GenericMPSTensor{S,N},
    O::∂PEPOSandwich{H},
    GL::GenericMPSTensor{S,N},
    GR::GenericMPSTensor{S,N},
) where {S,N,H}
    # sanity check
    @assert H == N - 3

    ∂p_e = _pepo_pepstensor_expr(:∂p, :bot, H + 1)
    AC_e = _pepo_edge_expr(:AC, :NW, :NE, :N, H)
    ĀC_e = _pepo_edge_expr(:ĀC, :SW, :SE, :S, H)
    GL_e = _pepo_edge_expr(:GL, :SW, :NW, :W, H)
    GR_e = _pepo_edge_expr(:GR, :NE, :SE, :E, H)
    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    rhs = Expr(:call, :*, AC_e, Expr(:call, :conj, ĀC_e), GL_e, GR_e, ket_e, pepo_es...)

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $∂p_e := $rhs))
end

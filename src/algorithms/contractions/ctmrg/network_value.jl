## Network value contractions
#
# The contractions making up `network_value`: the single site contraction, which
# dispatches on the local sandwich making up the network, and the corner and edge
# contractions which normalize it. The latter only involve environment tensors and are
# therefore independent of the sandwich type.

## Site contraction
function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::CTMRG_PEPS_EdgeTensor, E_east::CTMRG_PEPS_EdgeTensor,
        E_south::CTMRG_PEPS_EdgeTensor, E_west::CTMRG_PEPS_EdgeTensor,
        O::PEPSSandwich,
    )
    return @autoopt @tensor E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::CTMRG_PF_EdgeTensor, E_east::CTMRG_PF_EdgeTensor,
        E_south::CTMRG_PF_EdgeTensor, E_west::CTMRG_PF_EdgeTensor,
        O::PFTensor,
    )
    return @autoopt @tensor E_west[χ_WSW D_W; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        O[D_W D_S; D_N D_E]
end

@generated function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::TE, E_east::TE, E_south::TE, E_west::TE,
        O::PEPOSandwich{H},
    ) where {TE <: CTMRGEdgeTensor, H}
    @assert numout(TE) == H + 3

    C_northwest_e = _corner_expr(:C_northwest, :WNW, :NNW)
    C_northeast_e = _corner_expr(:C_northeast, :NNE, :ENE)
    C_southeast_e = _corner_expr(:C_southeast, :ESE, :SSE)
    C_southwest_e = _corner_expr(:C_southwest, :SSW, :WSW)

    E_north_e = _pepo_edge_expr(:E_north, :NNW, :NNE, :N, H)
    E_east_e = _pepo_edge_expr(:E_east, :ENE, :ESE, :E, H)
    E_south_e = _pepo_edge_expr(:E_south, :SSE, :SSW, :S, H)
    E_west_e = _pepo_edge_expr(:E_west, :WSW, :WNW, :W, H)

    ket_e, bra_e, pepo_es = _pepo_sandwich_expr(:O, H)

    rhs = Expr(
        :call, :*,
        C_northwest_e, C_northeast_e, C_southeast_e, C_southwest_e,
        E_north_e, E_east_e, E_south_e, E_west_e,
        ket_e, Expr(:call, :conj, bra_e),
        pepo_es...,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

## Normalization contractions
function _contract_corners(
        C_northwest::CTMRGCornerTensor, C_northeast::CTMRGCornerTensor,
        C_southeast::CTMRGCornerTensor, C_southwest::CTMRGCornerTensor,
    )
    return @tensor C_northwest[1; 2] * C_northeast[2; 3] *
        C_southeast[3; 4] * C_southwest[4; 1]
end

@generated function _contract_vertical_edges(
        C_northwest::CTMRGCornerTensor, C_northeast::CTMRGCornerTensor,
        C_southeast::CTMRGCornerTensor, C_southwest::CTMRGCornerTensor,
        E_east::CTMRGEdgeTensor{T, S, N},
        E_west::CTMRGEdgeTensor{T, S, N},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:NW),), (envlabel(:N),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:N),), (envlabel(:NE),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:SE),), (envlabel(:S),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:S),), (envlabel(:SW),))

    E_east_e = tensorexpr(
        :E_east, (envlabel(:NE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SE),)
    )
    E_west_e = tensorexpr(
        :E_west, (envlabel(:SW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NW),)
    )

    rhs = Expr(
        :call, :*,
        E_west_e, C_northwest_e, C_northeast_e, E_east_e, C_southeast_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

@generated function _contract_horizontal_edges(
        C_northwest::CTMRGCornerTensor, C_northeast::CTMRGCornerTensor,
        C_southeast::CTMRGCornerTensor, C_southwest::CTMRGCornerTensor,
        E_north::CTMRGEdgeTensor{T, S, N}, E_south::CTMRGEdgeTensor{T, S, N},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:W),), (envlabel(:NW),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:NE),), (envlabel(:E),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:E),), (envlabel(:SE),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SW),), (envlabel(:W),))

    E_north_e = tensorexpr(
        :E_north, (envlabel(:NW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NE),)
    )
    E_south_e = tensorexpr(
        :E_south, (envlabel(:SE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SW),)
    )

    rhs = Expr(
        :call, :*,
        C_northwest_e, E_north_e, C_northeast_e, C_southeast_e, E_south_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

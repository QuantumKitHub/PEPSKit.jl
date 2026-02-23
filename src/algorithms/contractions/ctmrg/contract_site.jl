## Site contraction
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

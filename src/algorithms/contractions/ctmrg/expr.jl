# Expressions
# -----------

## PEPS tensor expressions

function _virtual_labels(dir, layer, args...; contract = nothing)
    return isnothing(contract) ? (dir, layer, args...) : (contract, layer)
end
_north_labels(args...; kwargs...) = _virtual_labels(:N, args...; kwargs...)
_east_labels(args...; kwargs...) = _virtual_labels(:E, args...; kwargs...)
_south_labels(args...; kwargs...) = _virtual_labels(:S, args...; kwargs...)
_west_labels(args...; kwargs...) = _virtual_labels(:W, args...; kwargs...)

# layer=:top for ket PEPS, layer=:bot for bra PEPS, connects to PEPO slice h
function _pepo_pepstensor_expr(
        tensorname, layer::Symbol, h::Int, args...;
        contract_north = nothing, contract_east = nothing,
        contract_south = nothing, contract_west = nothing,
    )
    return tensorexpr(
        tensorname,
        (physicallabel(h, args...),),
        (
            virtuallabel(_north_labels(layer, args...; contract = contract_north)...),
            virtuallabel(_east_labels(layer, args...; contract = contract_east)...),
            virtuallabel(_south_labels(layer, args...; contract = contract_south)...),
            virtuallabel(_west_labels(layer, args...; contract = contract_west)...),
        ),
    )
end

# PEPO slice h
function _pepo_pepotensor_expr(
        tensorname, h::Int, args...;
        contract_north = nothing, contract_east = nothing,
        contract_south = nothing, contract_west = nothing,
    )
    layer = Symbol(:mid, :_, h)
    return tensorexpr(
        tensorname,
        (physicallabel(h + 1, args...), physicallabel(h, args...)),
        (
            virtuallabel(_north_labels(layer, args...; contract = contract_north)...),
            virtuallabel(_east_labels(layer, args...; contract = contract_east)...),
            virtuallabel(_south_labels(layer, args...; contract = contract_south)...),
            virtuallabel(_west_labels(layer, args...; contract = contract_west)...),
        ),
    )
end

# PEPOSandwich
function _pepo_sandwich_expr(sandwichname, H::Int, args...; kwargs...)
    ket_e = _pepo_pepstensor_expr(:(ket($sandwichname)), :top, 1, args...; kwargs...)
    bra_e = _pepo_pepstensor_expr(:(bra($sandwichname)), :bot, H + 1, args...; kwargs...)
    pepo_es = map(1:H) do h
        return _pepo_pepotensor_expr(:(pepo($sandwichname, $h)), h, args...; kwargs...)
    end

    return ket_e, bra_e, pepo_es
end

## Corner expressions

function _corner_expr(cornername, codom_label, dom_label, args...)
    return tensorexpr(
        cornername, (envlabel(codom_label, args...),), (envlabel(dom_label, args...),)
    )
end

## Edge expressions

function _pepo_edge_expr(edgename, codom_label, dom_label, dir, H::Int, args...)
    return tensorexpr(
        edgename,
        (
            envlabel(codom_label),
            virtuallabel(dir, :top, args...),
            ntuple(i -> virtuallabel(dir, :mid, i, args...), H)...,
            virtuallabel(dir, :bot, args...),
        ),
        (envlabel(dom_label),),
    )
end

## Enlarged corner (quadrant) expressions

function _pepo_enlarged_corner_expr(
        cornername, codom_label, dom_label, codom_dir, dom_dir, H::Int, args...
    )
    return tensorexpr(
        cornername,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, args...), H)...,
            virtuallabel(codom_dir, :bot, args...),
        ),
        (
            envlabel(dom_label, args...),
            virtuallabel(dom_dir, :top, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, args...), H)...,
            virtuallabel(dom_dir, :bot, args...),
        ),
    )
end

## Environment expressions

function _pepo_env_expr(
        envname, codom_label, dom_label, codom_dir, dom_dir, codom_site, dom_site, H::Int,
        args...,
    )
    return tensorexpr(
        envname,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, codom_site, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, codom_site, args...), H)...,
            virtuallabel(codom_dir, :bot, codom_site, args...),
        ),
        (
            envlabel(dom_label, args...),
            virtuallabel(dom_dir, :top, dom_site, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, dom_site, args...), H)...,
            virtuallabel(dom_dir, :bot, dom_site, args...),
        ),
    )
end

function _pepo_env_arg_expr(argname, codom_label, codom_dir, codom_site, H::Int, args...)
    return tensorexpr(
        argname,
        (
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, codom_site, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, codom_site, args...), H)...,
            virtuallabel(codom_dir, :bot, codom_site, args...),
        ),
    )
end

## Projector expressions

function _pepo_codomain_projector_expr(
        projname, codom_label, dom_label, dom_dir, H::Int, args...
    )
    return tensorexpr(
        projname,
        (envlabel(codom_label),),
        (
            envlabel(dom_label),
            virtuallabel(dom_dir, :top, args...),
            ntuple(i -> virtuallabel(dom_dir, :mid, i, args...), H)...,
            virtuallabel(dom_dir, :bot, args...),
        ),
    )
end

function _pepo_domain_projector_expr(
        projname, codom_label, codom_dir, dom_label, H::Int, args...
    )
    return tensorexpr(
        projname,
        (
            envlabel(codom_label),
            virtuallabel(codom_dir, :top, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, args...), H)...,
            virtuallabel(codom_dir, :bot, args...),
        ),
        (envlabel(dom_label),),
    )
end

## HalfInfiniteEnv expressions

function _half_infinite_environment_expr_parts(H)
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

    return C1_e, E1_e, E2_e, ket1_e, bra1_e, pepo1_es, C2_e, E3_e, E4_e, ket2_e, bra2_e, pepo2_es
end

function _half_infinite_environment_expr(H)

    C1_e, E1_e, E2_e, ket1_e, bra1_e, pepo1_es, C2_e, E3_e, E4_e, ket2_e, bra2_e, pepo2_es =
        _half_infinite_environment_expr_parts(H)

    partial_expr = Expr(
        :call, :*,
        E1_e, C1_e, E2_e,
        ket1_e, Expr(:call, :conj, bra1_e), pepo1_es...,
        E3_e, C2_e, E4_e,
        ket2_e, Expr(:call, :conj, bra2_e), pepo2_es...,
    )

    return partial_expr
end

function _half_infinite_environment_conj_expr(H)

    C1_e, E1_e, E2_e, ket1_e, bra1_e, pepo1_es, C2_e, E3_e, E4_e, ket2_e, bra2_e, pepo2_es =
        _half_infinite_environment_expr_parts(H)

    partial_expr = Expr(
        :call, :*,
        Expr(:call, :conj, E1_e), Expr(:call, :conj, C1_e), Expr(:call, :conj, E2_e),
        Expr(:call, :conj, ket1_e), bra1_e, map(x -> Expr(:call, :conj, x), pepo1_es)...,
        Expr(:call, :conj, E3_e), Expr(:call, :conj, C2_e), Expr(:call, :conj, E4_e),
        Expr(:call, :conj, ket2_e), bra2_e, map(x -> Expr(:call, :conj, x), pepo2_es)...,
    )

    return partial_expr
end

## FullInfiniteEnv expressions

function _full_infinite_environment_expr_parts(H)
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
    C3_e = _corner_expr(:C_3, :ESE, :SSE)
    E5_e = _pepo_edge_expr(:E_5, :EC, :ESE, :E, H, 3)
    E6_e = _pepo_edge_expr(:E_6, :SSE, :SC, :S, H, 3)
    ket3_e, bra3_e, pepo3_es = _pepo_sandwich_expr(
        :A_3, H, 3; contract_north = :EC, contract_west = :SC
    )

    # site 4 (domain)
    C4_e = _corner_expr(:C_4, :SSW, :WSW)
    E7_e = _pepo_edge_expr(:E_7, :SC, :SSW, :S, H, 4)
    E8_e = _pepo_edge_expr(:E_8, :WSW, :NW, :W, H, 4)
    ket4_e, bra4_e, pepo4_es = _pepo_sandwich_expr(:A_4, H, 4; contract_east = :SC)

    return (
        E1_e, C1_e, E2_e,
        ket1_e, bra1_e, pepo1_es,
        E3_e, C2_e, E4_e,
        ket2_e, bra2_e, pepo2_es,
        E5_e, C3_e, E6_e,
        ket3_e, bra3_e, pepo3_es,
        E7_e, C4_e, E8_e,
        ket4_e, bra4_e, pepo4_es,
    )
end

function _full_infinite_environment_expr(H)
    (
        E1_e, C1_e, E2_e,
        ket1_e, bra1_e, pepo1_es,
        E3_e, C2_e, E4_e,
        ket2_e, bra2_e, pepo2_es,
        E5_e, C3_e, E6_e,
        ket3_e, bra3_e, pepo3_es,
        E7_e, C4_e, E8_e,
        ket4_e, bra4_e, pepo4_es,
    ) = _full_infinite_environment_expr_parts(H)

    partial_expr = Expr(
        :call, :*,
        E1_e, C1_e, E2_e,
        ket1_e, Expr(:call, :conj, bra1_e), pepo1_es...,
        E3_e, C2_e, E4_e,
        ket2_e, Expr(:call, :conj, bra2_e), pepo2_es...,
        E5_e, C3_e, E6_e,
        ket3_e, Expr(:call, :conj, bra3_e), pepo3_es...,
        E7_e, C4_e, E8_e,
        ket4_e, Expr(:call, :conj, bra4_e), pepo4_es...,
    )

    return partial_expr
end

function _full_infinite_environment_conj_expr(H)
    (
        E1_e, C1_e, E2_e,
        ket1_e, bra1_e, pepo1_es,
        E3_e, C2_e, E4_e,
        ket2_e, bra2_e, pepo2_es,
        E5_e, C3_e, E6_e,
        ket3_e, bra3_e, pepo3_es,
        E7_e, C4_e, E8_e,
        ket4_e, bra4_e, pepo4_es,
    ) = _full_infinite_environment_expr_parts(H)

    partial_expr = Expr(
        :call, :*,
        Expr(:call, :conj, E1_e), Expr(:call, :conj, C1_e), Expr(:call, :conj, E2_e),
        Expr(:call, :conj, ket1_e), bra1_e, map(x -> Expr(:call, :conj, x), pepo1_es)...,
        Expr(:call, :conj, E3_e), Expr(:call, :conj, C2_e), Expr(:call, :conj, E4_e),
        Expr(:call, :conj, ket2_e), bra2_e, map(x -> Expr(:call, :conj, x), pepo2_es)...,
        Expr(:call, :conj, E5_e), Expr(:call, :conj, C3_e), Expr(:call, :conj, E6_e),
        Expr(:call, :conj, ket3_e), bra3_e, map(x -> Expr(:call, :conj, x), pepo3_es)...,
        Expr(:call, :conj, E7_e), Expr(:call, :conj, C4_e), Expr(:call, :conj, E8_e),
        Expr(:call, :conj, ket4_e), bra4_e, map(x -> Expr(:call, :conj, x), pepo4_es)...,
    )

    return partial_expr
end

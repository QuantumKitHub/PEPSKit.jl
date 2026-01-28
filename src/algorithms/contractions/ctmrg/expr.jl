#
# Expressions
#

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
            envlabel(codom_label, args...),
            virtuallabel(dir, :top, args...),
            ntuple(i -> virtuallabel(dir, :mid, i, args...), H)...,
            virtuallabel(dir, :bot, args...),
        ),
        (envlabel(dom_label, args...),),
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
        (envlabel(codom_label, args...),),
        (
            envlabel(dom_label, args...),
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
            envlabel(codom_label, args...),
            virtuallabel(codom_dir, :top, args...),
            ntuple(i -> virtuallabel(codom_dir, :mid, i, args...), H)...,
            virtuallabel(codom_dir, :bot, args...),
        ),
        (envlabel(dom_label, args...),),
    )
end

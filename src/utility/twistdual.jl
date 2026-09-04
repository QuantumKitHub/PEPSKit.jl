"""
$(SIGNATURES)

Twist the i-th leg of a tensor `t` in place if it represents a dual space.
"""
function twistdual!(t::AbstractTensorMap, i::Int)
    isdual(space(t, i)) || return t
    return twist!(t, i)
end
function twistdual!(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist!(t, is′)
end

"""
$(SIGNATURES)

Twist the i-th leg of a tensor `t` if it represents a dual space.
"""
function twistdual(t::AbstractTensorMap, i::Int)
    isdual(space(t, i)) || return t
    return twist(t, i)
end
function twistdual(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist(t, is′)
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(twistdual), t::AbstractTensorMap, i
    )
    tout, twistdual_pullback = rrule_via_ad(config, twistdual, t, i)
    return tout, twistdual_pullback
end

"""
$(SIGNATURES)

Twist the i-th leg of a tensor `t` in place if it represents a non-dual space.
"""
function twistnondual!(t::AbstractTensorMap, i::Int)
    !isdual(space(t, i)) || return t
    return twist!(t, i)
end
function twistnondual!(t::AbstractTensorMap, is)
    is′ = filter(i -> !isdual(space(t, i)), is)
    return twist!(t, is′)
end

"""
$(SIGNATURES)

Twist the i-th leg of a tensor `t` if it represents a non-dual space.
"""
function twistnondual(t::AbstractTensorMap, i::Int)
    !isdual(space(t, i)) || return t
    return twist(t, i)
end
function twistnondual(t::AbstractTensorMap, is)
    is′ = filter(i -> !isdual(space(t, i)), is)
    return twist(t, is′)
end
function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(twistnondual), t::AbstractTensorMap, i
    )
    tout, twistnondual_pullback = rrule_via_ad(config, twistnondual, t, i)
    return tout, twistnondual_pullback
end

"""
$(SIGNATURES)

Apply a twist to domain or codomain indices that correspond to dual spaces.
"""
function _linearmap_twist!(t::AbstractTensorMap)
    twistdual!(t, 1:numout(t))
    twistnondual!(t, (numout(t) + 1):numind(t))
    return t
end

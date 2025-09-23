#=
In order to use the machinery of AD but still have the option of controlling the algorithm
for computing gradients, it is possible to hook into the pullback of a function and define
a custom rrule. This is achieved via a wrapper function that takes an optional keyword
argument `alg_rrule` which is consequently used to customize the pullback.
In order to be able to specialize on this, we wrap the rrule in a function `_rrule` which
receives this as its first positional argument.
=#

"""
    hook_pullback(f, args...; alg_rrule=nothing, kwargs...)

Wrapper function to customize the pullback of a function `f`. This function is equivalent to
`f(args...; kwargs...)`, but the pullback can be customized by implementing the following
function:

    _rrule(alg_rrule, config, f, args...; kwargs...) -> NoTangent(), ∂f, ∂args...

This function can specialize on its first argument in order to customize the pullback. If no
specialization is needed, the default `alg_rrule=nothing` results in the default AD
pullback.

See also [`_rrule`](@ref).
"""
function hook_pullback(@nospecialize(f), args...; alg_rrule = nothing, kwargs...)
    return f(args...; kwargs...)
end

function ChainRulesCore.rrule(
        config::RuleConfig, ::typeof(hook_pullback), f, args...; alg_rrule = nothing, kwargs...
    )
    # Need to add ∂hook_pullback
    y, f_pullback = _rrule(alg_rrule, config, f, args...; kwargs...)
    return y, Δ -> (NoTangent(), f_pullback(Δ)...)
end

"""
    _rrule(alg_rrule, config, f, args...; kwargs...) -> ∂f, ∂args...

Customize the pullback of a function `f`. This function can specialize on its first argument
in order to have multiple implementations for a pullback. If no specialization is needed,
the default `alg_rrule=nothing` results in the default AD pullback.

!!! warning
    No tangent is expected for the `alg_rrule` argument
"""
function _rrule(::Nothing, config::RuleConfig, f, args...; kwargs...)
    return rrule_via_ad(config, f, args...; kwargs...)
end

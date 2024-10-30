"""
    dtmap(args...; kwargs...)

Differentiable wrapper around `OhMyThreads.tmap`.

All calls of `dtmap` inside of PEPSKit use the threading keyword arguments stored
inside `Default.threading_kwargs` which can be modified using `set_threading_kwargs!`.
"""
dtmap(args...; scheduler=Defaults.scheduler[]) = tmap(args...; scheduler)

# Follows the `map` rrule from ChainRules.jl but specified for the case of one AbstractArray that is being mapped
# https://github.com/JuliaDiff/ChainRules.jl/blob/e245d50a1ae56ce46fc8c1f0fe9b925964f1146e/src/rulesets/Base/base.jl#L243
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(dtmap), f, A::AbstractArray; kwargs...
)
    el_rrules = tmap(A; kwargs...) do a
        rrule_via_ad(config, f, a)
    end
    y = map(first, el_rrules)
    f_projector = ProjectTo(f)
    A_projectors = map(ProjectTo, A)

    function dtmap_pullback(dy_raw)
        dys = unthunk(dy_raw)
        backevals = tmap(el_rrules, dys; kwargs...) do el_rrule, dy
            last(el_rrule)(dy)
        end
        df = f_projector(sum(first, backevals))
        dA = map((Pa, backeval) -> Pa(last(backeval)), A_projectors, backevals)
        return NoTangent(), df, dA
    end

    return y, dtmap_pullback
end

"""
    set_threading_kwargs!(; kwargs...)

Modify multi-threading keyword arguments that are supplied to every call of `dtmap`,
i.e. the differentiable version of `OhMyThreads.tmap`.

To see all available keyword arguments, check the
[`Scheduler` page](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#OhMyThreads.Schedulers.Scheduler)
from the `OhMyThreads` docs.
"""
function set_threading_kwargs!(; kwargs...)
    length(kwargs) > 0 || throw(ArgumentError("need at least one keyword argument"))
    return merge!(THREADING_KWARGS, Dict(kwargs...))
end

"""
    @fwdthreads(ex)

Apply `Threads.@threads` only in the forward pass of the program.

It works by wrapping the for-loop expression in an if statement where in the forward pass
the loop in computed in parallel using `Threads.@threads`, whereas in the backwards pass
the `Threads.@threads` is omitted in order to make the expression differentiable.
"""
macro fwdthreads(ex)
    @assert ex.head === :for "@fwdthreads expects a for loop:\n$ex"

    diffable_ex = quote
        if Zygote.isderiving()
            $ex
        else
            Threads.@threads $ex
        end
    end

    return esc(diffable_ex)
end

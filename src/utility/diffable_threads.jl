"""
    dtmap(args...; kwargs...)

Differentiable wrapper around [OhMyThreads.tmap].

All calls of `dtmap` inside of PEPSKit use the threading scheduler stored inside
`Defaults.scheduler` which can be modified using `set_scheduler!`.
"""
dtmap(args...; scheduler = Defaults.scheduler[]) = tmap(args...; scheduler)

dtmap!!(args...; scheduler = Defaults.scheduler[]) = tmap!(args...; scheduler)

"""
    scheduler_allocator(x)

Default allocator for contractions on `x` in work scheduled on `Defaults.scheduler`, which is
what PEPSKit's parallel maps such as [`dtmap`](@ref) run on. A `SerialScheduler` gets an
allocator owned by a single task (e.g. a buffer), any other scheduler gets an allocator
that can be shared between tasks; see `MPSKit.default_allocator`.
"""
scheduler_allocator(x) = default_allocator(x, Defaults.scheduler[])

# Follows the `map` rrule from ChainRules.jl but specified for the case of one AbstractArray that is being mapped
# https://github.com/JuliaDiff/ChainRules.jl/blob/e245d50a1ae56ce46fc8c1f0fe9b925964f1146e/src/rulesets/Base/base.jl#L243
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(dtmap), f, A::AbstractArray;
        scheduler = Defaults.scheduler[]
    )
    # only infer the pullback for the concrete scheduler, inferring it for an abstract
    # `Scheduler` makes the first Zygote call compile for tens of minutes
    return Base.inferencebarrier(_dtmap_rrule)(config, f, A, scheduler)
end
function _dtmap_rrule(config, f, A, scheduler)
    el_rrules = tmap(A; scheduler) do a
        rrule_via_ad(config, f, a)
    end
    y = map(first, el_rrules)
    f_projector = ProjectTo(f)
    A_projectors = map(ProjectTo, A)

    function dtmap_pullback(dy_raw)
        dys = unthunk(dy_raw)
        backevals = tmap(el_rrules, dys; scheduler) do el_rrule, dy
            last(el_rrule)(dy)
        end
        df = f_projector(sum(first, backevals))
        dA = map((Pa, backeval) -> Pa(last(backeval)), A_projectors, backevals)
        return NoTangent(), df, dA
    end

    return y, dtmap_pullback
end

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(dtmap!!),
        f, C′::AbstractArray, A::AbstractArray;
        kwargs...,
    )
    C, dtmap_pullback = rrule(config, dtmap, f, A; kwargs...)
    function dtmap!!_pullback(dy)
        dtmap, df, dA = dtmap_pullback(dy)
        return dtmap, df, NoTangent, dA
    end
    return C, dtmap!!_pullback
end

"""
    dtmap(args...; kwargs...)

Differentiable wrapper around `OhMyThreads.tmap`.

All calls of `dtmap` inside of PEPSKit use the threading scheduler stored inside
`Defaults.scheduler` which can be modified using `set_scheduler!`.
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
    set_scheduler!([scheduler]; kwargs...)
    set_scheduler!()

Set `OhMyThreads` multi-threading scheduler parameters.

The function either accepts a `scheduler` as an `OhMyThreads.Scheduler` or
as a symbol where the corresponding parameters are specificed as keyword arguments.
For instance, a static scheduler that uses four tasks with chunking enabled
can be set via
```
set_scheduler!(StaticScheduler(; ntasks=4, chunking=true))
```
or equivalently with 
```
set_scheduler!(:static; ntasks=4, chunking=true)
```
If no `scheduler` is passed and only kwargs are provided, the `DynamicScheduler`
constructor is used with the provided kwargs.
For a detailed description of all schedulers and their keyword arguments consult the
[`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).

To reset the scheduler to its default value, one calls `set_scheduler!` without passing
arguments which then uses the default `DynamicScheduler()`. If the number of used threads is
just one it falls back to `StaticScheduler()`.
"""
set_scheduler!(scheduler=OhMyThreads.Implementation.NotGiven(); kwargs...) =
    Defaults.set_scheduler!(scheduler; kwargs...)

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

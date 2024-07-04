# settings for determining contraction orders
const PEPS_PHYSICALDIM = TensorOperations.Power{:χ}(2, 0) # 2
const PEPS_BONDDIM = TensorOperations.Power{:χ}(1, 1) # χ
const PEPS_ENVBONDDIM = TensorOperations.Power{:χ}(1, 2) # χ²

"""
    autoopt(ex)

Preprocessor macro for `@tensor` which automatically inserts costs for all symbols that start with a pattern.
In particular, all labels that start with `d`, `D`, or `χ` are automatically inserted with the corresponding
costs.
"""
macro autoopt(ex)
    dump(ex)
    @assert Meta.isexpr(ex, :macrocall) && ex.args[1] === Symbol("@tensor") "@autoopt expects a @tensor expression:\n$ex"

    # extract expression and kwargs
    tensorexpr = ex.args[end]
    kwargs = TensorOperations.parse_tensor_kwargs(ex.args[3:(end - 1)])

    # extract pre-existing opt data
    opt_id = findfirst(kwargs) do param
        param[1] === :opt
    end
    if !isnothing(opt_id)
        _, val = kwargs[opt_id]
        if val isa Expr
            @show optdict = TensorOperations.optdata(val, tensorexpr)
        elseif val isa Bool && val
            @show optdict = TensorOperations.optdata(tensorexpr)
        else
            throw(ArgumentError("Invalid use of `opt`"))
        end
    else
        optdict = TensorOperations.optdata(tensorexpr)
    end

    # insert costs for all labels starting with d, D, or χ
    replace!(optdict) do (label, cost)
        return if startswith(string(label), "d")
            label => PEPS_PHYSICALDIM
        elseif startswith(string(label), "D")
            label => PEPS_BONDDIM
        elseif startswith(string(label), "χ")
            label => PEPS_ENVBONDDIM
        else
            label => cost
        end
    end

    # insert opt data into tensor expression
    optexpr = Expr(:tuple, (Expr(:call, :(=>), k, v) for (k, v) in optdict)...)
    if isnothing(opt_id)
        insert!(ex.args, length(ex.args), :(opt = $optexpr))
    else
        ex.args[opt_id + 2] = :(opt = $optexpr)
    end
    return esc(ex)
end

# TODO: this definition should be in TensorOperations
TensorOperations.parsecost(x::TensorOperations.Power) = x

"""
    @diffset assign

Helper macro which allows in-place operations in the forward-pass of Zygote, but
resorts to non-mutating operations in the backwards-pass. The expression `assign`
should assign an object to an pre-existing `AbstractArray` and the use of updating
operators is also possible. This is especially needed when in-place assigning
tensors to unit-cell arrays of environments.
"""
macro diffset(ex)
    return esc(parse_ex(ex))
end
parse_ex(ex) = ex
function parse_ex(ex::Expr)
    oppheads = (:(./=), :(.*=), :(.+=), :(.-=))
    opprep = (:(./), :(.*), :(.+), :(.-))
    if ex.head=== :macrocall
        parse_ex(macroexpand(PEPSKit, ex))
    elseif ex.head in (:(.=), :(=)) && length(ex.args) == 2 && is_indexing(ex.args[1])
        lhs = ex.args[1]
        rhs = ex.args[2]

        vname = lhs.args[1]
        args = lhs.args[2:end]
        quote
            $vname = _setindex($vname, $rhs, $(args...))
        end
    elseif ex.head in oppheads && length(ex.args) == 2 && is_indexing(ex.args[1])
        hit = findfirst(x -> x == ex.head, oppheads)
        rep = opprep[hit]

        lhs = ex.args[1]
        rhs = ex.args[2]

        vname = lhs.args[1]
        args = lhs.args[2:end]

        quote
            $vname = _setindex($vname, $(rep)($lhs, $rhs), $(args...))
        end
    else
        return Expr(ex.head, parse_ex.(ex.args)...)
    end
end

is_indexing(ex) = false
is_indexing(ex::Expr) = ex.head=== :ref

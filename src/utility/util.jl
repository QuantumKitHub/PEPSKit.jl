function sdiag_inv_sqrt(S::AbstractTensorMap)
    toret = similar(S)

    if sectortype(S) == Trivial
        copyto!(toret.data, LinearAlgebra.diagm(LinearAlgebra.diag(S.data) .^ (-1 / 2)))
    else
        for (k, b) in blocks(S)
            copyto!(
                blocks(toret)[k], LinearAlgebra.diagm(LinearAlgebra.diag(b) .^ (-1 / 2))
            )
        end
    end

    return toret
end
function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt), S::AbstractTensorMap)
    toret = sdiag_inv_sqrt(S)
    return toret,
    c̄ -> (ChainRulesCore.NoTangent(), -1 / 2 * _elementwise_mult(c̄, toret'^3))
end
function _elementwise_mult(a::AbstractTensorMap, b::AbstractTensorMap)
    dst = similar(a)
    for (k, block) in blocks(dst)
        copyto!(block, blocks(a)[k] .* blocks(b)[k])
    end
    return dst
end

#rotl90 appeared to lose PeriodicArray'ness, which in turn caused zygote problems
#Base.rotl90(a::Array) = Array(rotl90(a));
#Base.rotr90(a::Array) = Array(rotr90(a));
function ChainRulesCore.rrule(::typeof(rotl90), a::AbstractMatrix)
    function pb(x)
        if !iszero(x)
            x = if x isa Tangent
                ChainRulesCore.construct(typeof(a), ChainRulesCore.backing(x))
            else
                x
            end
            x = rotr90(x)
        end

        return (ZeroTangent(), x)
    end
    return rotl90(a), pb
end

structure(t) = codomain(t) ← domain(t);

function _setindex(a::AbstractArray, v, args...)
    b::typeof(a) = copy(a)
    b[args...] = v
    return b
end

function ChainRulesCore.rrule(::typeof(_setindex), a::AbstractArray, tv, args...)
    t = _setindex(a, tv, args...)

    function toret(v)
        if iszero(v)
            backwards_tv = ZeroTangent()
            backwards_a = ZeroTangent()
        else
            v = if v isa Tangent
                ChainRulesCore.construct(typeof(a), ChainRulesCore.backing(v))
            else
                v
            end
            v = typeof(v) != typeof(a) ? convert(typeof(a), v) : v
            #v = convert(typeof(a),v);
            backwards_tv = v[args...]
            backwards_a = copy(v)
            if typeof(backwards_tv) == eltype(a)
                backwards_a[args...] = zero(v[args...])
            else
                backwards_a[args...] = zero.(v[args...])
            end
        end
        return (
            NoTangent(), backwards_a, backwards_tv, fill(ZeroTangent(), length(args))...
        )
    end
    return t, toret
end

macro diffset(ex)
    return esc(parse_ex(ex))
end
parse_ex(ex) = ex
function parse_ex(ex::Expr)
    oppheads = (:(./=), :(.*=), :(.+=), :(.-=))
    opprep = (:(./), :(.*), :(.+), :(.-))
    if ex.head == :macrocall
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
is_indexing(ex::Expr) = ex.head == :ref

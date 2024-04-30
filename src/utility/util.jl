# Get next and previous directional CTM enviroment index, respecting periodicity
_next(i, total) = mod1(i + 1, total)
_prev(i, total) = mod1(i - 1, total)

# Element-wise multiplication of TensorMaps respecting block structure
function _elementwise_mult(a::AbstractTensorMap, b::AbstractTensorMap)
    dst = similar(a)
    for (k, block) in blocks(dst)
        copyto!(block, blocks(a)[k] .* blocks(b)[k])
    end
    return dst
end

# Compute √S⁻¹ for diagonal TensorMaps
function sdiag_inv_sqrt(S::AbstractTensorMap)
    invsq = similar(S)

    if sectortype(S) == Trivial
        copyto!(invsq.data, LinearAlgebra.diagm(LinearAlgebra.diag(S.data) .^ (-1 / 2)))
    else
        for (k, b) in blocks(S)
            copyto!(
                blocks(invsq)[k], LinearAlgebra.diagm(LinearAlgebra.diag(b) .^ (-1 / 2))
            )
        end
    end

    return invsq
end

function ChainRulesCore.rrule(::typeof(sdiag_inv_sqrt), S::AbstractTensorMap)
    invsq = sdiag_inv_sqrt(S)
    function sdiag_inv_sqrt_pullback(c̄)
        return (ChainRulesCore.NoTangent(), -1 / 2 * _elementwise_mult(c̄, invsq'^3))
    end
    return invsq, sdiag_inv_sqrt_pullback
end

# Check whether diagonals contain degenerate values up to absolute or relative tolerance
function is_degenerate_spectrum(
    S; atol::Real=0, rtol::Real=atol > 0 ? 0 : sqrt(eps(scalartype(S)))
)
    for (_, b) in blocks(S)
        s = real(diag(b))
        for i in 1:(length(s) - 1)
            isapprox(s[i], s[i + 1]; atol, rtol) && return true
        end
    end
    return false
end

# Create empty projectors for given state without recomputing transpose
function projector_type(T::DataType, size)
    Pleft = Array{T,length(size)}(undef, size)
    Prtype = tensormaptype(spacetype(T), numin(T), numout(T), storagetype(T))
    Pright = Array{Prtype,length(size)}(undef, size)
    return Pleft, Pright
end

# There are no rrules for rotl90 and rotr90 in ChainRules.jl
function ChainRulesCore.rrule(::typeof(rotl90), a::AbstractMatrix)
    function rotl90_pullback(x)
        if !iszero(x)
            x = if x isa Tangent
                ChainRulesCore.construct(typeof(a), ChainRulesCore.backing(x))
            else
                x
            end
            x = rotr90(x)
        end

        return NoTangent(), x
    end
    return rotl90(a), rotl90_pullback
end

function ChainRulesCore.rrule(::typeof(rotr90), a::AbstractMatrix)
    function rotr90_pullback(x)
        if !iszero(x)
            x = if x isa Tangent
                ChainRulesCore.construct(typeof(a), ChainRulesCore.backing(x))
            else
                x
            end
            x = rotl90(x)
        end

        return NoTangent(), x
    end
    return rotr90(a), rotr90_pullback
end

# Differentiable setindex! alternative
function _setindex(a::AbstractArray, v, args...)
    b::typeof(a) = copy(a)
    b[args...] = v
    return b
end

function ChainRulesCore.rrule(::typeof(_setindex), a::AbstractArray, tv, args...)
    t = _setindex(a, tv, args...)

    function _setindex_pullback(v)
        if iszero(v)
            backwards_tv = ZeroTangent()
            backwards_a = ZeroTangent()
        else
            v = if v isa Tangent
                ChainRulesCore.construct(typeof(a), ChainRulesCore.backing(v))
            else
                v
            end
            # TODO: Fix this for ZeroTangents
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
    return t, _setindex_pullback
end

# Allows in-place operations during AD that copies when differentiating
# Especially needed to set tensors in unit cell of environments
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

"""
    @showtypeofgrad(x)

Macro utility to show to type of the gradient that is about to accumulate for `x`.

See also [`Zygote.@showgrad`](@ref).
"""
macro showtypeofgrad(x)
    return :(
        Zygote.hook($(esc(x))) do x̄
            println($"∂($x) = ", repr(typeof(x̄)))
            x̄
        end
    )
end

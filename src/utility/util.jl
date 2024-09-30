using OhMyThreads: tmap, tforeach

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

"""
    projector_type(T::DataType, size)
    projector_type(edges::Array{<:AbstractTensorMap})

Create two arrays of specified `size` that contain undefined tensors representing
left and right acting projectors, respectively. The projector types are inferred
from the TensorMap type `T` which avoids having to recompute transpose tensors.
Alternatively, supply an array of edge tensors from which left and right projectors
are intialized explicitly with zeros.
"""
function projector_type(T::DataType, size)
    P_left = Array{T,length(size)}(undef, size)
    Prtype = tensormaptype(spacetype(T), numin(T), numout(T), storagetype(T))
    P_right = Array{Prtype,length(size)}(undef, size)
    return P_left, P_right
end
function projector_type(edges::Array{<:AbstractTensorMap})
    P_left = map(e -> TensorMap(zeros, scalartype(e), space(e)), edges)
    P_right = map(e -> TensorMap(zeros, scalartype(e), domain(e), codomain(e)), edges)
    return P_left, P_right
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

# Differentiable wrappers around OhMyThreads functions to avoid type piracy in rrule definition
dtmap(args...; kwargs...) = tmap(args...; kwargs...)
dtforeach(args...; kwargs...) = tforeach(args...; kwargs...)

# Follows the `map` rrule from ChainRules.jl but specified for the case of one AbstractArray that is being mapped
# https://github.com/JuliaDiff/ChainRules.jl/blob/e245d50a1ae56ce46fc8c1f0fe9b925964f1146e/src/rulesets/Base/base.jl#L243
function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode},
    ::typeof(dtmap),
    f::F,
    A::AbstractArray;
    kwargs...,
) where {F}
    el_rrules = tmap(A; kwargs...) do a
        rrule_via_ad(config, f, a)
    end
    y = map(first, el_rrules)
    function map_pullback(dy_raw)
        dy = unthunk(dy_raw)
        backevals = tmap(CartesianIndices(A); kwargs...) do idx
            last(el_rrules[idx])(dy[idx])
        end
        df = ProjectTo(f)(sum(first, backevals))
        dA = map(CartesianIndices(A)) do idx
            ProjectTo(A[idx])(last(backevals[idx]))
        end
        return (NoTangent(), df, dA)
    end
    return y, map_pullback
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

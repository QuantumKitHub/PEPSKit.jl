# Get next and previous directional CTM enviroment index, respecting periodicity
_next(i, total) = mod1(i + 1, total)
_prev(i, total) = mod1(i - 1, total)

# iterator over each coordinates
"""
    eachcoordinate(x, dirs=1:4)

Enumerate all (dir, row, col) pairs.
"""
function eachcoordinate end

@non_differentiable eachcoordinate(args...)

# Element-wise multiplication of TensorMaps respecting block structure
function _elementwise_mult(a::AbstractTensorMap, b::AbstractTensorMap)
    dst = similar(a)
    for (k, block) in blocks(dst)
        copyto!(block, blocks(a)[k] .* blocks(b)[k])
    end
    return dst
end

_safe_pow(a, pow, tol) = (pow < 0 && abs(a) < tol) ? zero(a) : a .^ pow

"""
    sdiag_pow(S::AbstractTensorMap, pow::Real; tol::Real=eps(eltype(S))^(3 / 4))

Compute `S^pow` for diagonal matrices `S`.
"""
function sdiag_pow(S::AbstractTensorMap, pow::Real; tol::Real=eps(eltype(S))^(3 / 4))
    tol *= norm(S, Inf)  # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
    Spow = similar(S)
    for (k, b) in blocks(S)
        copyto!(
            blocks(Spow)[k],
            LinearAlgebra.diagm(_safe_pow.(LinearAlgebra.diag(b), pow, tol)),
        )
    end
    return Spow
end

"""
    absorb_s(u::AbstractTensorMap, s::AbstractTensorMap, vh::AbstractTensorMap)

Given `tsvd` result `u`, `s` and `vh`, absorb singular values `s` into `u` and `vh` by:
```
    u -> u * sqrt(s), vh -> sqrt(s) * vh
```
"""
function absorb_s(u::AbstractTensorMap, s::AbstractTensorMap, vh::AbstractTensorMap)
    sqrt_s = sdiag_pow(s, 0.5)
    return u * sqrt_s, sqrt_s * vh
end

function ChainRulesCore.rrule(
    ::typeof(sdiag_pow), S::AbstractTensorMap, pow::Real; tol::Real=eps(eltype(S))^(3 / 4)
)
    tol *= norm(S, Inf)
    spow = sdiag_pow(S, pow; tol)
    spow2 = sdiag_pow(S, pow - 1; tol)
    function sdiag_pow_pullback(c̄)
        return (ChainRulesCore.NoTangent(), pow * _elementwise_mult(c̄, spow2))
    end
    return spow, sdiag_pow_pullback
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

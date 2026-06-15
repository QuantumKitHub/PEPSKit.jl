# Element-wise multiplication of TensorMaps respecting block structure
function _elementwise_mult(a₁::AbstractTensorMap, a₂::AbstractTensorMap)
    dst = similar(a₁)
    for (k, b) in blocks(dst)
        copyto!(b, block(a₁, k) .* block(a₂, k))
    end
    return dst
end

_safe_pow(a::Number, pow::Real, tol::Real) = (pow < 0 && abs(a) < tol) ? zero(a) : a^pow

"""
    sdiag_pow(s, pow::Real; tol::Real=eps(real(scalartype(s)))^(3 / 4))

Compute `s^pow` for a diagonal matrix `s`.
"""
function sdiag_pow(s::DiagonalTensorMap, pow::Real; tol::Real = eps(real(scalartype(s)))^(3 / 4))
    # Relative tol w.r.t. largest abs value of `s` (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = DiagonalTensorMap(_safe_pow.(s.data, pow, tol), space(s, 1))
    return spow
end
function sdiag_pow(
        s::AbstractTensorMap{T, S, 1, 1}, pow::Real; tol::Real = eps(real(scalartype(s)))^(3 / 4)
    ) where {T, S}
    # Relative tol w.r.t. largest abs value of `s` (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = similar(s)
    for (k, b) in blocks(s)
        copyto!(
            block(spow, k), LinearAlgebra.diagm(_safe_pow.(LinearAlgebra.diag(b), pow, tol))
        )
    end
    return spow
end

function ChainRulesCore.rrule(
        ::typeof(sdiag_pow), s::AbstractTensorMap, pow::Real;
        tol::Real = eps(real(scalartype(s)))^(3 / 4),
    )
    tol *= norm(s, Inf)
    spow = sdiag_pow(s, pow; tol)
    spow_minus1_conj = scale!(sdiag_pow(s', pow - 1; tol), pow)
    function sdiag_pow_pullback(c̄_)
        c̄ = unthunk(c̄_)
        return (ChainRulesCore.NoTangent(), _elementwise_mult(c̄, spow_minus1_conj))
    end
    return spow, sdiag_pow_pullback
end

"""
    absorb_s(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)

Given SVD result `U`, `S` and `V`, absorb singular values `S` into `U` and `V` by:
```
    U -> U * sqrt(S), V -> sqrt(S) * V
```
"""
function absorb_s(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)
    @assert !isdual(space(S, 1))
    sqrt_S = sdiag_pow(S, 0.5)
    return U * sqrt_S, sqrt_S * V
end

_fliptwist_s(s::DiagonalTensorMap) = twist!(DiagonalTensorMap(flip(s, 1:2)), 1)

# Check whether diagonals contain degenerate values up to absolute or relative tolerance
function is_degenerate_spectrum(
        S; atol::Real = 0, rtol::Real = atol > 0 ? 0 : sqrt(eps(scalartype(S)))
    )
    for (_, b) in blocks(S)
        s = real(diag(b))
        for i in 1:(length(s) - 1)
            isapprox(s[i], s[i + 1]; atol, rtol) && return true
        end
    end
    return false
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

# TODO: link to Zygote.showgrad once they update documenter.jl
"""
    @showtypeofgrad(x)

Macro utility to show to type of the gradient that is about to accumulate for `x`.

See also `Zygote.@showgrad`.
"""
macro showtypeofgrad(x)
    return :(
        Zygote.hook($(esc(x))) do x̄
            println($"∂($x) = ", repr(typeof(x̄)))
            x̄
        end
    )
end

"""
Randomly take the dual of `ElementarySpace`s in `Vs` with propability `p`
"""
function random_dual!(Vs::AbstractMatrix{E}; p = 0.7) where {E <: ElementarySpace}
    for (i, V) in enumerate(Vs)
        (rand() < p) && (Vs[i] = V')
    end
    return Vs
end

"""
    _permute_to_last(axes::NTuple{N, Int}, ax::Int) where {N}

Returns `(1, 2, ..., N)` but with `ax` moved to the end,
and the corresponding permutation for `axes` (with `ax` as the only domain index).
"""
function _permute_to_last(axes::NTuple{N, Int}, ax::Int) where {N}
    codomain_axes = TupleTools.deleteat(ntuple(identity, N), ax)
    q = invperm(axes)
    biperm = (map(i -> q[i], codomain_axes), (q[ax],))
    new_axes = (ntuple(i -> axes[biperm[1][i]], N - 1)..., ax)
    return new_axes, biperm
end

"""
    stablemap(f, A)

Type-stable replacement for `map(f, A)` on the CTMRG differentiation path.

`Base.map` builds a `Base.Generator` whose element type is unknown, so `_collect`
falls back to a type-widening loop over untyped storage. Enzyme cannot statically
prove the element type through that and bails out with an `EnzymeNoTypeError`.
Inferring the element type up front and filling a concretely typed destination
keeps the same semantics without the widening machinery.
"""
@inline function stablemap(f::F, A) where {F}
    T = Base.promote_op(f, eltype(A))
    if !isconcretetype(T)              # inference failed: fall back to Base
        return map(f, A)
    end
    dst = similar(A, T)
    @inbounds for (i, a) in zip(eachindex(dst), A)
        dst[i] = f(a)
    end
    return dst
end

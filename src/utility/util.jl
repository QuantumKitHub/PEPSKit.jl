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
    _maxabs(data)

Largest absolute value in `data`, equal to `norm(_, Inf)` for the diagonal storage of a
`DiagonalTensorMap`.

Returns a scalar by default.
"""
function _maxabs(data::AbstractArray)
    isempty(data) && return zero(real(eltype(data)))
    # Largest abs value of `data` (use norm(∘, Inf) to make differentiable)
    return norm(data, Inf)
end
_maxabs(s::AbstractTensorMap) = _maxabs(s.data)

"""
    sdiag_pow(s, pow::Real; tol::Real=eps(real(scalartype(s)))^(3 / 4))

Compute `s^pow` for a diagonal matrix `s`.
The result preserves the scalar and storage types of `s`.
"""
function sdiag_pow(s::DiagonalTensorMap, pow::Real; tol::Real = eps(real(scalartype(s)))^(3 / 4))
    tol *= _maxabs(s)
    spow = similar(s)
    spow.data .= _safe_pow.(s.data, pow, tol)
    return spow
end
function sdiag_pow(
        s::AbstractTensorMap{T, S, 1, 1}, pow::Real; tol::Real = eps(real(scalartype(s)))^(3 / 4)
    ) where {T, S}
    tol *= _maxabs(s)
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
    tol *= _maxabs(s)
    spow = sdiag_pow(s, pow; tol)
    spow_minus1_conj = scale!(sdiag_pow(s', pow - 1; tol), pow)
    function sdiag_pow_pullback(c̄_)
        c̄ = unthunk(c̄_)
        # pow is a fixed algorithm parameter in PEPSKit, so we omit its derivative.
        return (ChainRulesCore.NoTangent(), _elementwise_mult(c̄, spow_minus1_conj), ChainRulesCore.NoTangent())
    end
    return spow, sdiag_pow_pullback
end

"""
    absorb_s(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)
    absorb_s!(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)

Given SVD result `U`, `S` and `V`, absorb singular values `S` into `U` and `V` by:
```
    U -> U * sqrt(S), V -> sqrt(S) * V
```
The in-place version `absorb_s!` overwrites `U` and `V`.
"""
function absorb_s(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)
    @assert !isdual(space(S, 1))
    sqrt_S = sdiag_pow(S, 0.5)
    return U * sqrt_S, sqrt_S * V
end
function absorb_s!(U::AbstractTensorMap, S::DiagonalTensorMap, V::AbstractTensorMap)
    @assert !isdual(space(S, 1))
    sqrt_S = sdiag_pow(S, 0.5)
    return rmul!(U, sqrt_S), lmul!(sqrt_S, V)
end

# avoids a conversion to a dense TensorMap by flip
function _fliptwist_s!(s::DiagonalTensorMap)
    for (f₁, f₂) in fusiontrees(s)
        data = s[f₁, f₂]
        _, θ = only(flip((f₁, f₂), (1, 2)))
        θ *= twist(f₁.uncoupled[1])
        scale!(data, θ)
    end
    return DiagonalTensorMap(s.data, flip(s.domain))
end

# Check whether diagonals contain degenerate values up to absolute or relative tolerance
function is_degenerate_spectrum(
        S; atol::Real = 0, rtol::Real = atol > 0 ? 0 : sqrt(eps(scalartype(S)))
    )
    approx = (x, y) -> isapprox(real(x), real(y); atol, rtol)
    # we need to distinguish vectors in the subspace
    # to avoid symmetry breaking rotations. vectors in
    # different blocks are safe.
    for (_, b) in blocks(S)
        s = diagview(b)
        n = length(s)
        n < 2 && continue
        left = view(s, 1:(n - 1))
        right = view(s, 2:n)
        mapreduce(approx, |, left, right; init = false) && return true
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

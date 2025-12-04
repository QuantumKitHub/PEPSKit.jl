# Get next and previous directional CTMRG environment index, respecting periodicity
_next(i, total) = mod1(i + 1, total)
_prev(i, total) = mod1(i - 1, total)

# Get next and previous coordinate (direction, row, column), given a direction and going around the environment clockwise
function _next_coordinate((dir, row, col), rowsize, colsize)
    if dir == 1
        return (_next(dir, 4), row, _next(col, colsize))
    elseif dir == 2
        return (_next(dir, 4), _next(row, rowsize), col)
    elseif dir == 3
        return (_next(dir, 4), row, _prev(col, colsize))
    elseif dir == 4
        return (_next(dir, 4), _prev(row, rowsize), col)
    end
end
function _prev_coordinate((dir, row, col), rowsize, colsize)
    if dir == 1
        return (_prev(dir, 4), _next(row, rowsize), col)
    elseif dir == 2
        return (_prev(dir, 4), row, _prev(col, colsize))
    elseif dir == 3
        return (_prev(dir, 4), _prev(row, rowsize), col)
    elseif dir == 4
        return (_prev(dir, 4), row, _next(col, colsize))
    end
end

# iterator over each coordinates
"""
    eachcoordinate(x, [dirs=1:4])

Enumerate all (dir, row, col) pairs.
"""
function eachcoordinate end

@non_differentiable eachcoordinate(args...)

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
    sdiag_pow(s, pow::Real; tol::Real=eps(scalartype(s))^(3 / 4))

Compute `s^pow` for a diagonal matrix `s`.
"""
function sdiag_pow(s::DiagonalTensorMap, pow::Real; tol::Real = eps(scalartype(s))^(3 / 4))
    # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
    tol *= norm(s, Inf)
    spow = DiagonalTensorMap(_safe_pow.(s.data, pow, tol), space(s, 1))
    return spow
end
function sdiag_pow(
        s::AbstractTensorMap{T, S, 1, 1}, pow::Real; tol::Real = eps(scalartype(s))^(3 / 4)
    ) where {T, S}
    # Relative tol w.r.t. largest singular value (use norm(∘, Inf) to make differentiable)
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
        tol::Real = eps(scalartype(s))^(3 / 4),
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

"""
    flip_svd(u::AbstractTensorMap, s::DiagonalTensorMap, vh::AbstractTensorMap)

Given SVD result `u ← s ← vh`, flip the arrow between the three tensors 
to `u2 → s2 → vh2` such that
```
    u * s * vh = (@tensor t2[-1, ...; -2, ...] := u2[-1, ...; 2] * s2[1; 2] * vh2[1; -2, ...])
```
The axis orders for `s`, `s2` are
```
    1 ← s ← 2   2 → s2 → 1
```
"""
function flip_svd(u::AbstractTensorMap, s::DiagonalTensorMap, vh::AbstractTensorMap)
    return flip(u, numind(u)), _flip_s(s), flip(vh, 1)
end
_flip_s(s::DiagonalTensorMap) = permute(DiagonalTensorMap(flip(s, (1, 2))), ((2,), (1,)))

"""
    twistdual(t::AbstractTensorMap, i)
    twistdual!(t::AbstractTensorMap, i)

Twist the i-th leg of a tensor `t` if it represents a dual space.
"""
function twistdual!(t::AbstractTensorMap, i::Int)
    isdual(space(t, i)) || return t
    return twist!(t, i)
end
function twistdual!(t::AbstractTensorMap, is)
    is′ = filter(i -> isdual(space(t, i)), is)
    return twist!(t, is′)
end
twistdual(t::AbstractTensorMap, is) = twistdual!(copy(t), is)

"""
    str(t)

Fermionic supertrace by using `@tensor`.
"""
str(t::AbstractTensorMap) = _str(BraidingStyle(sectortype(t)), t)
_str(::Bosonic, t::AbstractTensorMap) = tr(t)
@generated function _str(::Fermionic, t::AbstractTensorMap{<:Any, <:Any, N, N}) where {N}
    tex = tensorexpr(:t, ntuple(identity, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $tex))
end

"""
    trmul(H, ρ)

Compute `tr(H * ρ)` without forming `H * ρ`.
"""
@generated function trmul(
        H::AbstractTensorMap{<:Any, S, N, N}, ρ::AbstractTensorMap{<:Any, S, N, N}
    ) where {S, N}
    Hex = tensorexpr(:H, ntuple(identity, N), ntuple(i -> i + N, N))
    ρex = tensorexpr(:ρ, ntuple(i -> i + N, N), ntuple(identity, N))
    return macroexpand(@__MODULE__, :(@tensor $Hex * $ρex))
end

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

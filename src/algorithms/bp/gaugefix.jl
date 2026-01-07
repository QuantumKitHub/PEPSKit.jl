"""
    struct BPGauge

Algorithm for gauging PEPS with belief propagation fixed point messages.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct BPGauge
    "Assume BP messages are Hermitian"
    ishermitian::Bool = true
end

"""
$(SIGNATURES)

Fix the gauge of `psi` using fixed point environment `env` of belief propagation.
"""
function gauge_fix(psi::InfinitePEPS, alg::BPGauge, env::BPEnv)
    psi′ = copy(psi)
    XXinv = map(eachcoordinate(psi, 1:2)) do I
        _, X, Xinv = _bp_gauge_fix!(CartesianIndex(I), psi′, env; ishermitian = alg.ishermitian)
        return X, Xinv
    end
    return psi′, XXinv
end

function _sqrt_bp_messages(I::CartesianIndex{3}, env::BPEnv; ishermitian::Bool = true)
    dir, row, col = Tuple(I)
    @assert dir == NORTH || dir == EAST

    M = env[dir, dir == NORTH ? _prev(row, end) : row, dir == EAST ? _next(col, end) : col]
    sqrtM, isqrtM = sqrt_invsqrt(M; ishermitian)
    Mᴴ = env[dir + 2, row, col]
    sqrtMᴴ, isqrtMᴴ = sqrt_invsqrt(transpose(Mᴴ); ishermitian)
    return sqrtM, isqrtM, sqrtMᴴ, isqrtMᴴ
end

"""
    _bp_gauge_fix!(I, psi::InfinitePEPS, env::BPEnv; ishermitian::Bool = true) -> psi, X, X⁻¹

For the bond at direction `I[1]` from site `I[2], I[3]`, we identify the following gauge matrices,
along the canonical direction of the PEPS arrows (`SOUTH ← NORTH` or `WEST ← EAST`):

```math
    I = √M⁻¹ √M √Mᴴ √M⁻ᴴ
      = √M⁻¹ (U Λ Vᴴ) √M⁻ᴴ
      = (√M⁻¹ U √Λ) (√Λ Vᴴ √M⁻ᴴ)
      = X X⁻¹
```

Which are then used to update the gauge of `psi`. Thus, by convention `X` is attached to the `SOUTH`/`WEST` directions
and `X⁻¹` is attached to the `NORTH`/`EAST` directions.
"""
function _bp_gauge_fix!(I::CartesianIndex{3}, psi::InfinitePEPS, env::BPEnv; ishermitian::Bool = true)
    dir, row, col = Tuple(I)
    @assert dir == NORTH || dir == EAST

    sqrtM, isqrtM, sqrtMᴴ, isqrtMᴴ = _sqrt_bp_messages(I, env; ishermitian)
    U, Λ, Vᴴ = svd_compact!(sqrtM * sqrtMᴴ)
    sqrtΛ = sdiag_pow(Λ, 1 / 2)
    X = isqrtM * U * sqrtΛ
    invX = sqrtΛ * Vᴴ * isqrtMᴴ
    if isdual(space(sqrtM, 1))
        X, invX = flip(X, 2), flip(invX, 1)
    end

    if dir == NORTH
        psi[row, col] = absorb_north_message(psi[row, col], X)
        psi[_prev(row, end), col] = absorb_south_message(psi[_prev(row, end), col], transpose(invX))
    elseif dir == EAST
        psi[row, col] = absorb_east_message(psi[row, col], X)
        psi[row, _next(col, end)] = absorb_west_message(psi[row, _next(col, end)], transpose(invX))
    end

    return psi, X, invX
end

"""
    SUWeight(env::BPEnv)

Construct `SUWeight` from belief propagation fixed point environment `env`.
"""
function SUWeight(env::BPEnv)
    wts = map(Iterators.product(1:2, axes(env, 2), axes(env, 3))) do (dir′, row, col)
        I = CartesianIndex(mod1(dir′ + 1, 2), row, col)
        sqrtM, _, sqrtMᴴ, _ = _sqrt_bp_messages(I, env; ishermitian = true)
        Λ = svd_vals!(sqrtM * sqrtMᴴ)
        return isdual(space(sqrtM, 1)) ? _fliptwist_s(Λ) : Λ
    end
    return SUWeight(wts)
end

"""
    BPEnv(wts::SUWeight)

Convert fixed point weights `wts` of trivial simple update
to a belief propagation environment.
"""
function BPEnv(wts::SUWeight)
    messages = map(Iterators.product(1:4, axes(wts, 2), axes(wts, 3))) do (d, r, c)
        wt = if d == NORTH
            twist(wts[2, _next(r, end), c], 1)
        elseif d == EAST
            twist(wts[1, r, _prev(c, end)], 1)
        elseif d == SOUTH
            permute(wts[2, r, c], ((2,), (1,)); copy = true)
        else # WEST
            permute(wts[1, r, c], ((2,), (1,)); copy = true)
        end
        return TensorMap(wt)
    end
    return BPEnv(messages)
end

function sqrt_invsqrt(A; ishermitian::Bool = true)
    if ishermitian
        D, V = eigh_full(A)
        sqrtA = V * sdiag_pow(D, 1 / 2) * V'
        isqrtA = V * sdiag_pow(D, -1 / 2) * V'
    else
        D, V = eig_full(A)
        V⁻¹ = inv(V)
        sqrtA = V * sdiag_pow(D, 1 / 2) * V⁻¹
        isqrtA = V * sdiag_pow(D, -1 / 2) * V⁻¹
        if scalartype(A) <: Real
            sqrtA = real(sqrtA)
            isqrtA = real(isqrtA)
        end
    end
    return sqrtA, isqrtA
end

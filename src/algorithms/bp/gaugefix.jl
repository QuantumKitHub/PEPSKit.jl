"""
    struct BPGauge

Algorithm for gauging PEPS with belief propagation fixed point messages.
"""
@kwdef struct BPGauge
    # TODO: add options
end

"""
    gauge_fix(psi::Union{InfinitePEPS, InfinitePEPO}, alg::BPGauge, env::BPEnv)

Fix the gauge of `psi` (which can be an [`InfinitePEPS`](@ref), or
an [`InfinitePEPO`](@ref) interpreted as purified state with two physical legs) 
using fixed point environment `env` of belief propagation.
"""
function gauge_fix(psi::InfinitePEPS, alg::BPGauge, env::BPEnv)
    psi′ = copy(psi)
    XXinv = map(eachcoordinate(psi, 1:2)) do I
        _, X, Xinv = _bp_gauge_fix!(CartesianIndex(I), psi′, env)
        return X, Xinv
    end
    return psi′, XXinv
end
function gauge_fix(psi::InfinitePEPO, alg::BPGauge, env::BPEnv)
    # convert to iPEPS, and store physical leg fusers
    Nr, Nc, Nh = size(psi)
    @assert Nh == 1
    psi_Fs = map(psi.A) do t
        return fuse_physicalspaces(t)
    end
    psi_Fs = reshape(psi_Fs, (Nr, Nc))
    psi′ = map(Base.Fix2(getindex, 1), psi_Fs)
    Fs = map(Base.Fix2(getindex, 2), psi_Fs)
    psi′, XXinv = gauge_fix(InfinitePEPS(psi′), alg, env)
    # convert back to iPEPO
    psi′ = map(zip(psi′.A, Fs)) do (t, F)
        return F' * t
    end
    psi′ = reshape(psi′, (Nr, Nc, 1))
    return InfinitePEPO(psi′), XXinv
end

function _sqrt_bp_messages(I::CartesianIndex{3}, env::BPEnv)
    dir, row, col = Tuple(I)
    @assert dir == NORTH || dir == EAST
    M12 = env[dir, dir == NORTH ? _prev(row, end) : row, dir == EAST ? _next(col, end) : col]
    sqrtM12, isqrtM12 = sqrt_invsqrt(twist(M12, 1))
    M21 = env[dir + 2, row, col]
    sqrtM21, isqrtM21 = sqrt_invsqrt(M21)
    return sqrtM12, isqrtM12, sqrtM21, isqrtM21
end

"""
    _bp_gauge_fix!(I, psi::InfinitePEPS, env::BPEnv) -> psi, X, X⁻¹

For the bond at direction `I[1]` (which can be `NORTH` or `EAST`)
from site `I[2], I[3]`, we identify the following gauge matrices,
along the canonical direction of the PEPS arrows (`SOUTH ← NORTH` or `WEST ← EAST`):

```math
    I = √M₁₂⁻¹ √M₁₂ √M₂₁ √M₂₁⁻¹
      = √M₁₂⁻¹ (U Λ Vᴴ) √M₂₁⁻¹
      = (√M₁₂⁻¹ U √Λ) (√Λ Vᴴ √M₂₁⁻¹)
      = X X⁻¹
```

Which are then used to update the gauge of `psi`. Thus, by convention `X` is attached to the `SOUTH`/`WEST` directions
and `X⁻¹` is attached to the `NORTH`/`EAST` directions.
"""
function _bp_gauge_fix!(I::CartesianIndex{3}, psi::InfinitePEPS, env::BPEnv)
    dir, row, col = Tuple(I)
    @assert dir == NORTH || dir == EAST

    sqrtM12, isqrtM12, sqrtM21, isqrtM21 = _sqrt_bp_messages(I, env)
    U, Λ, Vᴴ = svd_compact!(sqrtM12 * sqrtM21)
    sqrtΛ = sdiag_pow(Λ, 1 / 2)
    X = isqrtM12 * U * sqrtΛ
    invX = sqrtΛ * Vᴴ * isqrtM21
    if isdual(space(sqrtM12, 1))
        X, invX = twist(flip(X, 2), 1), flip(invX, 1)
    end
    if dir == NORTH
        psi[row, col] = absorb_north_message(psi[row, col], X)
        psi[_prev(row, end), col] = absorb_south_message(psi[_prev(row, end), col], invX)
    elseif dir == EAST
        psi[row, col] = absorb_east_message(psi[row, col], X)
        psi[row, _next(col, end)] = absorb_west_message(psi[row, _next(col, end)], invX)
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
        sqrtM12, _, sqrtM21, _ = _sqrt_bp_messages(I, env)
        Λ = svd_vals!(sqrtM12 * sqrtM21)
        return isdual(space(sqrtM12, 1)) ? _fliptwist_s(Λ) : Λ
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
            copy(wts[2, r, c])
        else # WEST
            copy(wts[1, r, c])
        end
        return TensorMap(wt)
    end
    return BPEnv(messages)
end

function sqrt_invsqrt(A::PEPSMessage)
    if isposdef(A)
        D, V = eigh_full(A)
        sqrtA = V * sdiag_pow(D, 1 / 2) * V'
        isqrtA = V * sdiag_pow(D, -1 / 2) * V'
    else
        D, V = eig_full(A)
        V⁻¹ = inv(V)
        sqrtA = V * sdiag_pow(D, 1 / 2) * V⁻¹
        isqrtA = V * sdiag_pow(D, -1 / 2) * V⁻¹
        if scalartype(A) <: Real
            # TODO: is this valid?
            sqrtA = real(sqrtA)
            isqrtA = real(isqrtA)
        end
    end
    return sqrtA, isqrtA
end

"""
$(SIGNATURES)

Fix the gauge of `psi` using fixed point environment of belief propagation.
"""
function gauge_fix(psi::InfinitePEPS, alg::BeliefPropagation, env::BPEnv = BPEnv(psi))
    env, err = leading_boundary(env, InfiniteSquareNetwork(psi), alg)
    psi′ = copy(psi)
    XXinv = map(eachcoordinate(psi, 1:2)) do I
        _, X, Xinv = _bp_gauge_fix!(CartesianIndex(I), psi′, env; ishermitian = alg.project_hermitian)
        return X, Xinv
    end
    return psi′, XXinv, env
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

    M = env[dir, dir == NORTH ? _prev(row, end) : row, dir == EAST ? _next(col, end) : col]
    sqrtM, isqrtM = sqrt_invsqrt(M; ishermitian)
    Mᴴ = env[dir + 2, row, col]
    sqrtMᴴ, isqrtMᴴ = sqrt_invsqrt(transpose(Mᴴ); ishermitian)

    U, Λ, Vᴴ = svd_compact!(sqrtM * sqrtMᴴ)
    sqrtΛ = sdiag_pow(Λ, 1 / 2)
    X = isqrtM * U * sqrtΛ
    invX = sqrtΛ * Vᴴ * isqrtMᴴ
    if !isdual(space(Mᴴ, 1))
        X, Λ, invX = flip(X, 2), _fliptwist_s(Λ), flip(invX, 1)
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

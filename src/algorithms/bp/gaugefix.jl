"""
$(SIGNATURES)

Fix the gauge of `psi` using fixed point environment of belief propagation.
"""
function gauge_fix(psi::InfinitePEPS, alg::BeliefPropagation, env::BPEnv = BPEnv(psi))
    env, err = bp_fixedpoint(env, InfiniteSquareNetwork(psi), alg)
    psi′, wts = _gauge_fix_bp(psi, env)
    return psi′, wts, env
end

function _get_sqrt_messages(env::BPEnv)
    sqrtmsgs = map(env.messages) do M
        # U = V for positive semi-definite message M
        # TODO: switch to eigh! after enforcing positive semi-definiteness
        U, S, Vᴴ = svd_compact(M)
        sqrtS = sdiag_pow(S, 0.5)
        isqrtS = sdiag_pow(S, -0.5)
        sqrtM = U * sqrtS * Vᴴ
        isqrtM = U * isqrtS * Vᴴ
        return sqrtM, isqrtM
    end
    sqrtMs = map(x -> x[1], sqrtmsgs)
    isqrtMs = map(x -> x[2], sqrtmsgs)
    return sqrtMs, isqrtMs
end

"""
Use BP environment `env` to fix gauge of InfinitePEPS `psi`.
"""
function _gauge_fix_bp(psi::InfinitePEPS, env::BPEnv)
    # Bring PEPS to the Vidal gauge
    sqrtMs, isqrtMs = _get_sqrt_messages(env)
    bond_svds = map(eachcoordinate(psi, 1:2)) do (dir, r, c)
        # TODO: would be more reasonable to define SOUTH as adjoint(NORTH)...
        # TODO: figure out twists for fermion
        #= 
        - dir = 1: x-bond (r,c) → (r,c+1)
            m[(r,c) → (r,c+1)] = env[4, r, c]
            m[(r,c+1) → (r,c)] = env[2, r, c+1]
        - dir = 2: y-bond (r,c) → (r-1,c)
            m[(r,c) → (r-1,c)] = env[3, r, c]
            m[(r-1,c) → (r,c)] = env[1, r-1, c]
        =#
        MM = if dir == 1
            _transpose(sqrtMs[WEST, r, c]) * sqrtMs[EAST, r, _next(c, end)]
        else
            _transpose(sqrtMs[SOUTH, r, c]) * sqrtMs[NORTH, _prev(r, end), c]
        end
        U, S, Vᴴ = svd_compact(MM)
        if isdual(space(U, 1))
            U, S, Vᴴ = flip_svd(U, S, Vᴴ)
        end
        return U, S, Vᴴ
    end
    Us = map(x -> x[1], bond_svds)
    Ss = map(x -> x[2], bond_svds)
    Vs = map(x -> x[3], bond_svds)
    ## bond weights Λ
    wts = SUWeight(Ss)
    ## gauge-fixed state
    psi′ = map(eachcoordinate(psi)) do (r, c)
        isqrtM_north = _transpose(isqrtMs[SOUTH, r, c])
        isqrtM_south = _transpose(isqrtMs[NORTH, r, c])
        isqrtM_east = _transpose(isqrtMs[WEST, r, c])
        isqrtM_west = _transpose(isqrtMs[EAST, r, c])

        U_north = Us[2, r, c]
        U_east = Us[1, r, c]
        Vᴴ_south = Vs[2, _next(r, end), c]
        Vᴴ_west = Vs[1, r, _prev(c, end)]
        # Vertex Γ tensors in Vidal gauge
        @tensor contractcheck = true begin
            Γ[d; DN DE DS DW] ≔
                psi[r, c][d; DN1 DE1 DS1 DW1] *
                (isqrtM_north[DN1; DN2] * U_north[DN2; DN]) *
                (isqrtM_east[DE1; DE2] * U_east[DE2; DE]) *
                (isqrtM_south[DS1; DS2] * Vᴴ_south[DS; DS2]) *
                (isqrtM_west[DW1; DW2] * Vᴴ_west[DW; DW2])
        end
        # convert to symmetric gauge by absorbing sqrt of weights
        return _absorb_weights(Γ, wts, r, c)
    end
    return InfinitePEPS(psi′), wts
end

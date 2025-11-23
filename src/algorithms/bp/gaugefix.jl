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
    return map(env.messages) do M
        # U = V for positive semi-definite message M
        # TODO: switch to eigh! after enforcing positive semi-definiteness
        U, S, Vᴴ = svd_compact!(M)
        sqrtM = U * sdiag_pow(S, 1 / 2) * Vᴴ
        isqrtM = U * sdiag_pow(S, -1 / 2) * Vᴴ
        return sqrtM, isqrtM
    end
end

"""
Use BP environment `env` to fix gauge of InfinitePEPS `psi`.
"""
function _gauge_fix_bp(psi::InfinitePEPS, env::BPEnv)
    # Bring PEPS to the Vidal gauge
    sqrtmsgs = _get_sqrt_messages(env)
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
            transpose(sqrtmsgs[WEST, r, c][1]) * sqrtmsgs[EAST, r, _next(c, end)][1]
        else
            transpose(sqrtmsgs[SOUTH, r, c][1]) * sqrtmsgs[NORTH, _prev(r, end), c][1]
        end
        U, S, Vᴴ = svd_compact!(MM)
        if isdual(space(U, 1))
            U, S, Vᴴ = flip_svd(U, S, Vᴴ)
        end
        return U, S, Vᴴ
    end
    ## bond weights Λ
    wts = SUWeight(
        map(eachcoordinate(psi, 1:2)) do (dir, r, c)
            return bond_svds[dir, r, c][2]
        end
    )
    ## gauge-fixed state
    psi′ = map(eachcoordinate(psi)) do (r, c)
        isqrtM_north = transpose(sqrtmsgs[SOUTH, r, c][2])
        isqrtM_south = transpose(sqrtmsgs[NORTH, r, c][2])
        isqrtM_east = transpose(sqrtmsgs[WEST, r, c][2])
        isqrtM_west = transpose(sqrtmsgs[EAST, r, c][2])

        U_north = bond_svds[2, r, c][1]
        U_east = bond_svds[1, r, c][1]
        Vᴴ_south = bond_svds[2, _next(r, end), c][3]
        Vᴴ_west = bond_svds[1, r, _prev(c, end)][3]
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
        return absorb_weight(Γ, wts, r, c, ntuple(identity, 4))
    end
    return InfinitePEPS(psi′), wts
end

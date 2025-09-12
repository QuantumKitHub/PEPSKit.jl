"""
    maximize_fidelity!(
        pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace;
        maxiter = 5, tol = 1.0e-3, boundary_alg=(; verbosity=1)
    )

Iteratively maximize the fidelity of `pepssrc` and `pepsdst` where the contents of `pepssrc`
are embedded into `pepsdst`. To contract the respective networks, the specified `envspace`
is used on the environment bonds and kept fixed. The CTMRG contraction algorithm is specified
via the `boundary_alg` `NamedTuple`.
"""
function maximize_fidelity!(
        pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace;
        maxiter = 5, tol = 1.0e-3, boundary_alg = (; verbosity = 1)
    )
    @assert size(pepsdst) == size(pepssrc) "incompatible unit cell sizes"

    # absorb src PEPS tensors into dst tensors in-place
    for (pdst, psrc) in zip(unitcell(pepsdst), unitcell(pepssrc))
        embed!(pdst, psrc)
    end

    peps = pepssrc
    peps′ = pepsdst
    env, = leading_boundary(CTMRGEnv(peps, envspace), peps; boundary_alg...)
    peps /= sqrt(norm(peps, env)) # normalize PEPSs to ensure that fidelity is bounded by 1
    fid = 0.0
    for i in 1:maxiter
        # normalize updated PEPS and contract ⟨ψ₁|ψ₂⟩
        env′, = leading_boundary(CTMRGEnv(peps′, envspace), peps′; boundary_alg...)
        peps′ /= sqrt(norm(peps′, env′))
        nw = InfiniteSquareNetwork(peps′, peps)
        envnw, = leading_boundary(CTMRGEnv(nw, envspace), nw; boundary_alg...)

        # remove peps′ from fidelity network and compute fidelity
        ∂nval = ∂network_value(peps′, envnw)
        fid′ = abs2(network_value(peps, ∂nval))
        @info @sprintf("Fidmax. iter %d:   fid = %.4e   Δfid = %.4e", i, fid′, fid′ - fid)
        abs(1 - fid′) ≤ tol && break

        # update PEPSs
        peps = peps′
        peps′ = ∂nval
        fid = fid′
    end

    return peps′
end

function network_value(peps::InfinitePEPS, ∂nval::InfinitePEPS)
    return prod(eachcoordinate(peps)) do (r, c)
        @tensor conj(peps[r, c][d; D_N D_E D_S D_W]) * ∂nval[r, c][d; D_N D_E D_S D_W]
    end
end

"""
    ∂network_value(peps::InfinitePEPS, env::CTMRGEnv)

Compute the `InfinitePEPS` resulting from removing the bra PEPS tensors in `network_value`.
"""
function ∂network_value(peps::InfinitePEPS, env::CTMRGEnv)
    return InfinitePEPS(
        map(eachcoordinate(peps)) do ind
            _∂contract_site(ind, peps, env) * _contract_corners(ind, env) /
                _contract_vertical_edges(ind, env) / _contract_horizontal_edges(ind, env)
        end
    )
end

function _∂contract_site(ind::Tuple{Int, Int}, peps::InfinitePEPS, env::CTMRGEnv)
    r, c = ind
    return _∂contract_site(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
        peps[r, c],
    )
end
function _∂contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::CTMRG_PEPS_EdgeTensor, E_east::CTMRG_PEPS_EdgeTensor,
        E_south::CTMRG_PEPS_EdgeTensor, E_west::CTMRG_PEPS_EdgeTensor, ψ,
    )
    return @autoopt @tensor peps′[d; D_N_below D_E_below D_S_below D_W_below] :=
        E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ψ[d; D_N_above D_E_above D_S_above D_W_above]
end

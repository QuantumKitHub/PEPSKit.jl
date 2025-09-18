"""
    single_site_fidelity_initialize(
        peps₀::InfinitePEPS, envspace, [bondspace = _maxspace(peps₀)];
        noise_amp = 1.0e-2, kwargs...
    )

Generate a single-site unit cell PEPS from a (possibly) multi-site `peps₀` by maximizing
the fidelity w.r.t. `peps₀`. Here, `envspace` determines the virtual environment spaces for
CTMRG contractions. By default, the maximal bond space of `peps₀` is used for all virtual
legs of the single-site PEPS.

The single-site PEPS is intialized with Gaussian nosie and multiplied by `noise_amp`.
The `kwargs...` are passed onto the [`maximize_fidelity!`](@ref) call, refer to the docs
for further details.
"""
function single_site_fidelity_initialize(
        peps₀::InfinitePEPS, envspace, bondspace = _spacemax(peps₀);
        noise_amp = 1.0e-1, kwargs...
    )
    @assert allequal(map(p -> space(p, 1), unitcell(peps₀))) "PEPS must have uniform physical spaces"

    physspace = space(unitcell(peps₀)[1], 1)
    peps_single = noise_amp * InfinitePEPS(randn, scalartype(peps₀), physspace, bondspace) # single-site unit cell with random noise

    peps_uc = InfinitePEPS(fill(only(unitcell(peps_single)), size(peps₀))) # fill peps₀ unit cell with peps_singles
    peps_single, env_single = approximate!(peps_uc, peps₀, envspace; kwargs...) # modifies peps_single in-place

    return peps_single, env_single
end

# maximal virtual space over unit cell
function _spacemax(peps::InfinitePEPS)
    return reduce(supremum, map(p -> supremum(domain(p)[1], domain(p)[2]), unitcell(peps)))
end

"""
    approximate!(
        pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace;
        maxiter = 5, tol = 1.0e-3, boundary_alg=(; verbosity=1)
    )

Approximate `pepssrc` with `pepsdst` by iteratively maximizing their fidelity where the
contents of `pepssrc` are embedded into `pepsdst`. To contract the respective networks, the
specified `envspace` is used on the environment bonds and kept fixed. The CTMRG contraction
algorithm is specified via the `boundary_alg` `NamedTuple`.
"""
function MPSKit.approximate!(
        pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace;
        maxiter = 10, tol = 1.0e-3, boundary_alg = (; verbosity = 1)
    )
    @assert size(pepsdst) == size(pepssrc) "incompatible unit cell sizes"

    # absorb src PEPS tensors into dst tensors in-place
    for (pdst, psrc) in zip(unitcell(pepsdst), unitcell(pepssrc))
        absorb!(pdst, psrc)
    end

    # normalize reference PEPS
    peps₀ = pepssrc # smaller bond spaces
    env₀, = leading_boundary(CTMRGEnv(peps₀, envspace), peps₀; boundary_alg...)
    peps₀ /= sqrt(abs(_local_norm(peps₀, peps₀, env₀))) # normalize to ensure that fidelity is bounded by 1

    # normalize maximizer PEPS
    peps = pepsdst
    env, = leading_boundary(CTMRGEnv(peps, envspace), peps; boundary_alg...)
    peps /= sqrt(abs(_local_norm(peps, peps, env)))

    nw₀ = InfiniteSquareNetwork(peps₀, peps)
    envnw, = leading_boundary(CTMRGEnv(nw₀, envspace), nw₀; boundary_alg...)
    peps′ = _∂local_norm(peps₀, envnw)
    fid = 0.0
    for i in 1:maxiter
        # compute fidelity from ∂norm
        fid′ = abs2(_local_norm(peps, peps′))
        @info @sprintf("Fidmax. iter %d:   fid = %.4e   Δfid = %.4e", i, fid′, fid′ - fid)
        abs(1 - fid′) ≤ tol && break

        # contract boundary of fidelity network
        envnw, = leading_boundary(env, InfiniteSquareNetwork(peps, peps′); boundary_alg...)
        ∂norm = _∂local_norm(peps, envnw)

        # renormalize current PEPS
        peps = peps′
        env, = leading_boundary(env, peps; boundary_alg...)
        peps /= sqrt(abs(_local_norm(peps, peps, env)))

        peps′ = ∂norm
        fid = fid′
    end

    return peps, env
end

"""
$(SIGNATURES)

Sum over `contract_local_norm` values of all unit cell entries.
"""
function _local_norm(ket::InfinitePEPS, bra::InfinitePEPS, env::CTMRGEnv)
    return sum(ind -> contract_local_norm((ind,), ket, bra, env), eachcoordinate(ket))
end
function _local_norm(peps::InfinitePEPS, ∂norm::InfinitePEPS)
    return sum(eachcoordinate(peps)) do (r, c)
        @tensor conj(peps[r, c][d; D_N D_E D_S D_W]) * ∂norm[r, c][d; D_N D_E D_S D_W]
    end
end


"""
$(SIGNATURES)

Compute the `InfinitePEPS` resulting from removing the bra PEPS tensors in `_local_norm`.
"""
function _∂local_norm(peps::InfinitePEPS, env::CTMRGEnv)
    return InfinitePEPS(map(ind -> _∂contract_site(ind, peps, env), eachcoordinate(peps)))
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

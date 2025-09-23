"""
    single_site_fidelity_initialize(
        peps₀::InfinitePEPS, envspace, [bondspace = _maxspace(peps₀)]; kwargs...
    )

Generate a single-site unit cell PEPS from a (possibly) multi-site `peps₀` by maximizing
the fidelity w.r.t. `peps₀`. Here, `envspace` determines the virtual environment spaces for
CTMRG contractions. By default, the maximal bond space of `peps₀` is used for all virtual
legs of the single-site PEPS.

## Keyword arguments

- `noise_amp=1.0-1` : Gaussian noise amplitude of initial single-site PEPS

Additionally, all keyword arguments of [`approximate!`](@ref) can be passed.
"""
function single_site_fidelity_initialize(
        peps₀::InfinitePEPS, envspace, bondspace = _spacemax(peps₀);
        noise_amp = 1.0e-1, kwargs...
    )
    @assert allequal(map(p -> space(p, 1), unitcell(peps₀))) "PEPS must have uniform physical spaces"

    physspace = space(unitcell(peps₀)[1], 1)
    peps_single = noise_amp * InfinitePEPS(randn, scalartype(peps₀), physspace, bondspace) # single-site unit cell with random noise

    # absorb peps₀ tensors into single-site tensors in-place
    peps_uc = InfinitePEPS(fill(only(unitcell(peps_single)), size(peps₀))) # fill peps₀ unit cell with peps_singles
    absorb!(peps_uc[1], peps₀[1]) # absorb (1, 1) tensor of peps₀ (applies to all peps_uc entries since absorb! is mutating)
    peps_single, = approximate!(peps_uc, peps₀, envspace; kwargs...)

    return InfinitePEPS([peps_single[1];;])
end

# maximal virtual space over unit cell
function _spacemax(peps::InfinitePEPS)
    return reduce(supremum, map(p -> supremum(domain(p)[1], domain(p)[2]), unitcell(peps)))
end

@doc """
    approximate(pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace; kwargs...)
    approximate!(pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace; kwargs...)

Approximate `pepssrc` with `pepsdst` by iteratively maximizing their fidelity, using
`pepsdst` as an initial guess. To contract the respective networks, the specified `envspace`
is used on the environment bonds and kept fixed.

## Keyword arguments

- `maxiter=5` : Maximal number of maximization iterations
- `tol=1.0e-3` : Absolute convergence tolerance for the infidelity
- `boundary_alg=(; verbosity=2)` : CTMRG contraction algorithm, either specified as a `NamedTuple` or `CTMRGAlgorithm`
"""
approximate, approximate!

function MPSKit.approximate!(
        pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace;
        maxiter = 10, tol = 1.0e-3, verbosity = 3, boundary_alg = (; verbosity = 1)
    )
    @assert size(pepsdst) == size(pepssrc) "incompatible unit cell sizes"
    @assert all(map((pdst, psrc) -> space(pdst, 1) == space(psrc, 1), unitcell(pepsdst), unitcell(pepssrc))) "incompatible physical spaces"

    log = MPSKit.IterLog("Approx.")
    return LoggingExtras.withlevel(; verbosity) do
        # normalize reference PEPS
        peps₀ = pepssrc # smaller bond spaces
        boundary_alg = _alg_or_nt(CTMRGAlgorithm, boundary_alg)
        env₀, = leading_boundary(CTMRGEnv(peps₀, envspace), peps₀, boundary_alg)
        peps₀ /= sqrt(abs(_local_norm(peps₀, peps₀, env₀))) # normalize to ensure that fidelity is bounded by 1

        # normalize maximizer PEPS
        peps = pepsdst
        env, = leading_boundary(CTMRGEnv(peps, envspace), peps, boundary_alg)
        peps /= sqrt(abs(_local_norm(peps, peps, env)))

        approximate_loginit!(log, one(real(scalartype(peps))), zero(real(scalartype(peps))))
        nw₀ = InfiniteSquareNetwork(peps₀, peps)
        envnw, = leading_boundary(CTMRGEnv(nw₀, envspace), nw₀, boundary_alg)
        peps′ = _∂local_norm(peps₀, envnw)
        for iter in 1:maxiter
            # compute fidelity from ∂norm
            fid = abs2(_local_norm(peps, peps′))
            infid = 1 - fid
            if abs(infid) ≤ tol
                approximate_logfinish!(log, iter, infid, fid)
                break
            end
            if iter == maxiter
                approximate_logcancel!(log, iter, infid, fid)
                break
            else
                approximate_logiter!(log, iter, infid, fid)
            end

            # contract boundary of fidelity network
            envnw, = leading_boundary(env, InfiniteSquareNetwork(peps, peps′), boundary_alg)
            ∂norm = _∂local_norm(peps, envnw)

            # renormalize current PEPS
            peps = peps′
            env, = leading_boundary(env, peps, boundary_alg)
            peps /= sqrt(abs(_local_norm(peps, peps, env)))

            peps′ = ∂norm
        end

        return peps, env
    end
end
function MPSKit.approximate(pepsdst::InfinitePEPS, pepssrc::InfinitePEPS, envspace; kwargs...)
    return approximate!(deepcopy(pepsdst), pepssrc, envspace; kwargs...)
end

# custom fidelity maximization logging
function approximate_loginit!(log, infid, fid)
    return @infov 2 loginit!(log, infid, fid)
end
function approximate_logiter!(log, iter, infid, fid)
    return @infov 3 logiter!(log, iter, infid, fid)
end
function approximate_logfinish!(log, iter, infid, fid)
    return @infov 2 logfinish!(log, iter, infid, fid)
end
function approximate_logcancel!(log, iter, infid, fid)
    return @warnv 1 logcancel!(log, iter, infid, fid)
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

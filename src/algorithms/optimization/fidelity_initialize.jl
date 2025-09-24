"""
$(TYPEDEF)

Abstract super type for tensor approximation algorithms.
"""
abstract type ApproximateAlgorithm end

const APPROXIMATE_SYMBOLS = IdDict{Symbol, Type{<:ApproximateAlgorithm}}()

"""
$(TYPEDEF)

PEPS approximation algorithm maximizing the fidelity by successively applying the fidelity
derivative with respect to the approximator PEPS.

## Constructors

    FidelityMaxCrude(; kwargs...)

Construct the approximation algorithm from the following keyword arguments:

* `tol::Float64=$(Defaults.approximate_tol)` : Infidelity tolerance of the approximation iteration.
* `maxiter::Int=$(Defaults.approximate_maxiter)` : Maximal number of approximation steps.
* `miniter::Int=$(Defaults.approximate_miniter)` : Minimal number of approximation steps.
* `verbosity::Int=$(Defaults.approximate_verbosity)` : Approximator output information verbosity.
* `boundary_alg::Union{<:CTMRGAlgorithm,NamedTuple}` : CTMRG algorithm used for contracting the norm and fidelity networks.
"""
struct FidelityMaxCrude <: ApproximateAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    envspace::ElementarySpace
    boundary_alg::CTMRGAlgorithm
end
function FidelityMaxCrude(; kwargs...)
    return ApproximateAlgorithm(; alg = :fidelitymaxcrude, kwargs...)
end

APPROXIMATE_SYMBOLS[:fidelitymaxcrude] = FidelityMaxCrude

"""
    ApproximateAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `ApproximateAlgorithm` algorithm struct.
"""
function ApproximateAlgorithm(;
        alg = Defaults.approximate_alg,
        tol = Defaults.approximate_tol,
        maxiter = Defaults.approximate_maxiter, miniter = Defaults.approximate_miniter,
        verbosity = Defaults.approximate_verbosity,
        boundary_alg = (; verbosity = maximum(1, verbosity - 2)),  # shouldn't be smaller than one by default
    )
    # replace symbol with projector alg type
    haskey(APPROXIMATE_SYMBOLS, alg) || throw(ArgumentError("unknown approximate algorithm: $alg"))
    alg_type = APPROXIMATE_SYMBOLS[alg]

    boundary_algorithm = _alg_or_nt(CTMRGAlgorithm, boundary_alg)

    return alg_type(tol, maxiter, miniter, verbosity, boundary_algorithm)
end

"""
    single_site_fidelity_initialize(
        peps::InfinitePEPS, [bondspace = _maxspace(peps₀)]; kwargs...
    )

Generate a single-site unit cell PEPS from a (possibly) multi-site `peps` by approximating
the respective entries using [`approximate!`](@ref). By default, the maximal bond space of
`peps₀` is used for all virtual legs of the single-site PEPS.

## Keyword arguments

* `noise_amp=1.0-1` : Gaussian noise amplitude of initial single-site PEPS

All additional keyword arguments will be passed to the [`approximate!`](@ref) call.
"""
function single_site_fidelity_initialize(
        peps::InfinitePEPS, bondspace = _spacemax(peps);
        noise_amp = 1.0e-1, kwargs...
    )
    @assert allequal(map(p -> space(p, 1), unitcell(peps))) "PEPS must have uniform physical spaces"

    physspace = space(unitcell(peps)[1], 1)
    peps_single = noise_amp * InfinitePEPS(randn, scalartype(peps), physspace, bondspace) # single-site unit cell with random noise

    # absorb peps₀ tensors into single-site tensors in-place
    peps_uc = InfinitePEPS(fill(only(unitcell(peps_single)), size(peps))) # fill unit cell with peps_single tensors
    absorb!(peps_uc[1], peps[1]) # absorb (1, 1) tensor of peps₀ (applies to all peps_uc entries since absorb! is mutating)
    peps_single, = approximate!(peps_uc, peps; kwargs...)

    return InfinitePEPS([peps_single[1];;])
end

# maximal virtual space over unit cell
function _spacemax(peps::InfinitePEPS)
    return reduce(supremum, map(p -> supremum(domain(p)[1], domain(p)[2]), unitcell(peps)))
end

@doc """
    approximate(ψ₀::InfinitePEPS, ψ::InfinitePEPS; kwargs...)
    approximate!(ψ₀::InfinitePEPS, ψ::InfinitePEPS; kwargs...)
    # expert versions
    approximate(ψ₀::InfinitePEPS, ψ::InfinitePEPS, alg::ApproximateAlgorithm)
    approximate!(ψ₀::InfinitePEPS, ψ::InfinitePEPS, alg::ApproximateAlgorithm)

Approximate `ψ` from the initial guess `ψ₀`. The approximation algorithm is specified via
the keyword arguments or directly by passing an [`ApproximateAlgorithm`](@ref) struct.

## Keyword arguments

* `alg::Symbol=:$(Defaults.approximate_alg)` : Approximation algorithm, which can be one of the following:
    - `:fidelitymaxcrude` : Maximize the fidelity from the fidelity gradient in a power-method fashion.
* `tol::Float64=$(Defaults.approximate_tol)` : Infidelity tolerance of the approximation iteration.
* `maxiter::Int=$(Defaults.approximate_maxiter)` : Maximal number of approximation steps.
* `miniter::Int=$(Defaults.approximate_miniter)` : Minimal number of approximation steps.
* `verbosity::Int=$(Defaults.approximate_verbosity)` : Approximator output information verbosity.
* `boundary_alg::Union{<:CTMRGAlgorithm,NamedTuple}` : CTMRG algorithm used for contracting the norm and fidelity networks.

## Return values

The final approximator and its environment are returned.
"""
approximate, approximate!


function MPSKit.approximate!(ψ₀::InfinitePEPS, ψ::InfinitePEPS; kwargs...)
    alg = ApproximateAlgorithm(; kwargs...)
    return approximate!(ψ₀, ψ, alg)
end
function MPSKit.approximate!(ψ₀::InfinitePEPS, ψ::InfinitePEPS, alg::FidelityMaxCrude)
    @assert size(ψ₀) == size(ψ) "incompatible unit cell sizes"
    @assert all(map((p₀, p) -> space(p₀, 1) == space(p, 1), unitcell(ψ₀), unitcell(ψ))) "incompatible physical spaces"

    log = MPSKit.IterLog("Approximate")
    return LoggingExtras.withlevel(; alg.verbosity) do
        # normalize reference PEPS
        peps_init = ψ # smaller bond spaces
        envspace = domain(ψ₀[1])
        env₀, = leading_boundary(CTMRGEnv(peps_init, envspace), peps_init, alg.boundary_alg)
        peps_init /= sqrt(abs(_local_norm(peps_init, peps_init, env₀))) # normalize to ensure that fidelity is bounded by 1

        # normalize maximizer PEPS
        peps = ψ₀
        env, = leading_boundary(CTMRGEnv(peps, envspace), peps, alg.boundary_alg)
        peps /= sqrt(abs(_local_norm(peps, peps, env)))

        approximate_loginit!(log, one(real(scalartype(peps))), zero(real(scalartype(peps))))
        nw₀ = InfiniteSquareNetwork(peps_init, peps) # peps₀ has different virtual spaces than peps
        envnw, = leading_boundary(CTMRGEnv(nw₀, envspace), nw₀, alg.boundary_alg)
        peps′ = _∂local_norm(peps_init, envnw)
        for iter in 1:maxiter
            # compute fidelity from ∂norm
            fid = abs2(_local_norm(peps, peps′))
            infid = 1 - fid
            if abs(infid) ≤ alg.tol && iter ≥ alg.miniter
                approximate_logfinish!(log, iter, infid, fid)
                break
            end
            if iter == alg.maxiter
                approximate_logcancel!(log, iter, infid, fid)
                break
            else
                approximate_logiter!(log, iter, infid, fid)
            end

            # contract boundary of fidelity network
            # initialize CTMRG on environment of peps′ (must have matching virtual spaces!)
            envnw, = leading_boundary(env, InfiniteSquareNetwork(peps, peps′), alg.boundary_alg)
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
function MPSKit.approximate(ψ₀::InfinitePEPS, args...; kwargs...)
    return approximate!(deepcopy(ψ₀), args...; kwargs...)
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

# contract CTMRG environment leaving open the bra-PEPS virtual and physical bonds
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

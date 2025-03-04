"""
    CTMRGAlgorithm

Abstract super type for the corner transfer matrix renormalization group (CTMRG) algorithm
for contracting infinite PEPS.
"""
abstract type CTMRGAlgorithm end

"""
    ctmrg_iteration(network, env, alg::CTMRGAlgorithm) -> env′, info

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(network, env, alg::CTMRGAlgorithm) end

"""
    MPSKit.leading_boundary(env₀, network; kwargs...)
    # expert version:
    MPSKit.leading_boundary(env₀, network, alg::CTMRGAlgorithm)

Contract `network` using CTMRG and return the CTM environment. The algorithm can be
supplied via the keyword arguments or directly as an [`CTMRGAlgorithm`](@ref) struct.

## Keyword arguments

### CTMRG iterations

* `tol::Real=$(Defaults.ctmrg_tol)`: Stopping criterium for the CTMRG iterations. This is the norm convergence, as well as the distance in singular values of the corners and edges.
* `miniter::Int=$(Defaults.ctmrg_miniter)`: Minimal number of CTMRG iterations.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)`: Maximal number of CTMRG iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)`: Output verbosity level, should be one of the following:
    0. Suppress all output
    1. Only print warnings
    2. Initialization and convergence info
    3. Iteration info
    4. Debug info
* `alg::Union{Symbol,Type{CTMRGAlgorithm}}=$(Defaults.ctmrg_alg)`: Variant of the CTMRG algorithm. See also [`CTMRGAlgorithm`](@ref).

### Projector algorithm

* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg=$(Defaults.trscheme))`: Truncation scheme for the projector computation, which controls the resulting virtual spaces.
* `svd_alg::Union{<:SVDAdjoint,NamedTuple}`: SVD algorithm for computing projectors. See also [`PEPSKit.tsvd`](@ref).
* `projector_alg::Union{Symbol,Type{ProjectorAlgorithm}}=$(Defaults.projector_alg)`: Variant of the projector algorithm. See also [`ProjectorAlgorithm`](@ref).
"""
function MPSKit.leading_boundary(env₀::CTMRGEnv, network::InfiniteSquareNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return MPSKit.leading_boundary(env₀, network, alg)
end
function MPSKit.leading_boundary(
    env₀::CTMRGEnv, network::InfiniteSquareNetwork, alg::CTMRGAlgorithm
)
    CS = map(x -> tsvd(x)[2], env₀.corners)
    TS = map(x -> tsvd(x)[2], env₀.edges)

    η = one(real(scalartype(network)))
    env = deepcopy(env₀)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    return LoggingExtras.withlevel(; alg.verbosity) do
        ctmrg_loginit!(log, η, network, env₀)
        local info
        for iter in 1:(alg.maxiter)
            env, info = ctmrg_iteration(network, env, alg)  # Grow and renormalize in all 4 directions
            η, CS, TS = calc_convergence(env, CS, TS)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, network, env)
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, network, env)
            else
                ctmrg_logiter!(log, iter, η, network, env)
            end
        end
        return env, info
    end
end
function MPSKit.leading_boundary(env₀::CTMRGEnv, state, args...; kwargs...)
    return MPSKit.leading_boundary(env₀, InfiniteSquareNetwork(state), args...; kwargs...)
end

# custom CTMRG logging
function ctmrg_loginit!(log, η, network, env)
    @infov 2 loginit!(log, η, network_value(network, env))
end
function ctmrg_logiter!(log, iter, η, network, env)
    @infov 3 logiter!(log, iter, η, network_value(network, env))
end
function ctmrg_logfinish!(log, iter, η, network, env)
    @infov 2 logfinish!(log, iter, η, network_value(network, env))
end
function ctmrg_logcancel!(log, iter, η, network, env)
    @warnv 1 logcancel!(log, iter, η, network_value(network, env))
end

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

# Available CTMRG algorithms as Symbols
const ctmrg_symbols = Dict(
    :simultaneous => SimultaneousCTMRG, :sequential => SequentialCTMRG
)

"""
    select_algorithm(::typeof(leading_boundary), env₀::CTMRGEnv; kwargs...) -> CTMRGAlgorithm

Parse and standardize CTMRG keyword arguments, and bundle them into a `CTMRGAlgorithm` struct,
which is passed on to [`leading_boundary`](@ref). See [`leading_boundary`](@ref) for a
description of all keyword arguments.
"""
function select_algorithm(
    ::typeof(leading_boundary),
    env₀::CTMRGEnv;
    alg=Defaults.ctmrg_alg,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    trscheme=(; alg=Defaults.trscheme),
    svd_alg=(;),
    projector_alg=Defaults.projector_alg, # only allows for Symbol/Type{ProjectorAlgorithm} to expose projector kwargs
)
    # extract maximal environment dimensions
    χenv = maximum(env₀.corners) do corner
        return dim(space(corner, 1))
    end
    krylovdim = round(Int, Defaults.krylovdim_factor * χenv)

    # replace symbol with projector alg type
    alg_type = if alg isa Symbol
        projector_symbols[alg]
    else
        alg
    end

    # parse SVD forward & rrule algorithm 
    svd_algorithm = if svd_alg isa SVDAdjoint
        svd_alg
    elseif svd_alg isa NamedTuple
        alg′ = select_algorithm(
            SVDAdjoint; rrule_alg=(; tol=1e1tol, verbosity=verbosity - 2), svd_alg...
        )
        if typeof(alg′.rrule_alg) <: Union{<:GMRES,<:Arnoldi}
            @reset alg′.rrule_alg.krylovdim = krylovdim
        end
    else
        throw(ArgumentError("unknown SVD algorithm: $svd_alg"))
    end

    # parse CTMRG projector algorithm
    projector_algorithm = select_algorithm(
        ProjectorAlgorithm; alg=projector_alg, svd_alg, trscheme, verbosity
    )

    return alg_type(tol, maxiter, miniter, verbosity, projector_algorithm)
end

#=
In order to compute an error measure, we compare the singular values of the current iteration with the previous one.
However, when the virtual spaces change, this comparison is not directly possible.
Instead, we project both tensors into the smaller space and then compare the difference.

TODO: we might want to consider embedding the smaller tensor into the larger space and then compute the difference
=#
function _singular_value_distance((S₁, S₂))
    V₁ = space(S₁, 1)
    V₂ = space(S₂, 1)
    if V₁ == V₂
        return norm(S₁ - S₂)
    else
        V = infimum(V₁, V₂)
        e1 = isometry(V₁, V)
        e2 = isometry(V₂, V)
        return norm(e1' * S₁ * e1 - e2' * S₂ * e2)
    end
end

"""
    calc_convergence(env, CS_old, TS_old)
    calc_convergence(env_new::CTMRGEnv, env_old::CTMRGEnv)

Given a new environment `env`, compute the maximal singular value distance.
This determined either from the previous corner and edge singular values
`CS_old` and `TS_old`, or alternatively, directly from the old environment.
"""
function calc_convergence(env, CS_old, TS_old)
    CS_new = map(x -> tsvd(x)[2], env.corners)
    ΔCS = maximum(_singular_value_distance, zip(CS_old, CS_new))

    TS_new = map(x -> tsvd(x)[2], env.edges)
    ΔTS = maximum(_singular_value_distance, zip(TS_old, TS_new))

    @debug "maxᵢ|Cⁿ⁺¹ - Cⁿ|ᵢ = $ΔCS   maxᵢ|Tⁿ⁺¹ - Tⁿ|ᵢ = $ΔTS"

    return max(ΔCS, ΔTS), CS_new, TS_new
end
function calc_convergence(env_new::CTMRGEnv, env_old::CTMRGEnv)
    CS_old = map(x -> tsvd(x)[2], env_old.corners)
    TS_old = map(x -> tsvd(x)[2], env_old.edges)
    return calc_convergence(env_new, CS_old, TS_old)
end
@non_differentiable calc_convergence(args...)

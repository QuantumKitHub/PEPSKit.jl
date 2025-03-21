"""
$(TYPEDEF)

Abstract super type for the corner transfer matrix renormalization group (CTMRG) algorithm
for contracting infinite PEPS.
"""
abstract type CTMRGAlgorithm end

const CTMRG_SYMBOLS = IdDict{Symbol,Type{<:CTMRGAlgorithm}}()

"""
    CTMRGAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `CTMRGAlgorithm` algorithm struct.
"""
function CTMRGAlgorithm(;
    alg=Defaults.ctmrg_alg,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    trscheme=(; alg=Defaults.trscheme),
    svd_alg=(;),
    projector_alg=Defaults.projector_alg, # only allows for Symbol/NamedTuple to expose projector kwargs
)
    # replace symbol with projector alg type
    haskey(CTMRG_SYMBOLS, alg) || throw(ArgumentError("unknown CTMRG algorithm: $alg"))
    alg_type = CTMRG_SYMBOLS[alg]

    # parse CTMRG projector algorithm

    projector_algorithm = ProjectorAlgorithm(;
        alg=projector_alg, svd_alg, trscheme, verbosity
    )

    return alg_type(tol, maxiter, miniter, verbosity, projector_algorithm)
end

"""
    ctmrg_iteration(network, env, alg::CTMRGAlgorithm) -> env′, info

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(network, env, alg::CTMRGAlgorithm) end

"""
    leading_boundary(env₀, network; kwargs...)
    # expert version:
    leading_boundary(env₀, network, alg::CTMRGAlgorithm)

Contract `network` using CTMRG and return the CTM environment. The algorithm can be
supplied via the keyword arguments or directly as an [`CTMRGAlgorithm`](@ref) struct.

## Keyword arguments

### CTMRG iterations

* `tol::Real=$(Defaults.ctmrg_tol)` : Stopping criterium for the CTMRG iterations. This is the norm convergence, as well as the distance in singular values of the corners and edges.
* `miniter::Int=$(Defaults.ctmrg_miniter)` : Minimal number of CTMRG iterations.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)` : Maximal number of CTMRG iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)` : Output verbosity level, should be one of the following:
    0. Suppress all output
    1. Only print warnings
    2. Initialization and convergence info
    3. Iteration info
    4. Debug info
* `alg::Symbol=:$(Defaults.ctmrg_alg)` : Variant of the CTMRG algorithm. See also [`CTMRGAlgorithm`](@ref).
    - `:simultaneous`: Simultaneous expansion and renormalization of all sides.
    - `:sequential`: Sequential application of left moves and rotations.

### Projector algorithm

* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg::Symbol=:$(Defaults.trscheme))` : Truncation scheme for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerr` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncdim` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:truncbelow` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `svd_alg::Union{<:SVDAdjoint,NamedTuple}` : SVD algorithm for computing projectors. See also [`SVDAdjoint`](@ref). By default, a reverse-rule tolerance of `tol=1e1tol` where the `krylovdim` is adapted to the `env₀` environment dimension.
* `projector_alg::Symbol=:$(Defaults.projector_alg)` : Variant of the projector algorithm. See also [`ProjectorAlgorithm`](@ref).
    - `:halfinfinite` : Projection via SVDs of half-infinite (two enlarged corners) CTMRG environments.
    - `:fullinfinite` : Projection via SVDs of full-infinite (all four enlarged corners) CTMRG environments.

## Return values

The CTMRG routine returns the final CTMRG environment as well as an information `NamedTuple`
containing the following fields:

* `truncation_error` : Last (maximal) SVD truncation error of the CTMRG projectors.
* `condition_number` : Last (maximal) condition number of the enlarged CTMRG environment

In case the `alg` is a `SimultaneousCTMRG`, the last SVD will also be returned:

* `U` : Last unit cell of left singular vectors.
* `S` : Last unit cell of singular values.
* `V` : Last unit cell of right singular vectors.

If, in addition, the specified SVD algorithm computes the full, untruncated SVD, the full
set of vectors and values will be returned as well:

* `U_full` : Last unit cell of all left singular vectors.
* `S_full` : Last unit cell of all singular values.
* `V_full` : Last unit cell of all right singular vectors.
"""
function leading_boundary(env₀::CTMRGEnv, network::InfiniteSquareNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return leading_boundary(env₀, network, alg)
end
function leading_boundary(
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
function leading_boundary(env₀::CTMRGEnv, state, args...; kwargs...)
    return leading_boundary(env₀, InfiniteSquareNetwork(state), args...; kwargs...)
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

# TODO: we might want to consider embedding the smaller tensor into the larger space and then compute the difference
"""
    _singular_value_distance((S₁, S₂))

Compute the singular value distance as an error measure, e.g. for CTMRG iterations.
To that end, the singular values of the current iteration `S₁` are compared with the
previous one `S₂`. When the virtual spaces change, this comparison is not directly possible
such that both tensors are projected into the smaller space and then subtracted.
"""
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
    calc_convergence(env_new, env_old)

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

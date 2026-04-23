"""
$(TYPEDEF)

Abstract super type for the corner transfer matrix renormalization group (CTMRG) algorithm
for contracting infinite PEPS.
"""
abstract type CTMRGAlgorithm end

const CTMRG_SYMBOLS = IdDict{Symbol, Type{<:CTMRGAlgorithm}}()


"""
    CTMRGAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `CTMRGAlgorithm` algorithm struct.
"""
function CTMRGAlgorithm(;
        alg = Defaults.ctmrg_alg,
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter, miniter = Defaults.ctmrg_miniter,
        verbosity = Defaults.ctmrg_verbosity,
        trunc = (; alg = Defaults.trunc),
        decomposition_alg = (;),
        projector_alg = Defaults.projector_alg, # only allows for Symbol/NamedTuple to expose projector kwargs
    )
    # replace symbol with projector alg type
    haskey(CTMRG_SYMBOLS, alg) || throw(ArgumentError("unknown CTMRG algorithm: $alg"))
    alg_type = CTMRG_SYMBOLS[alg]

    # parse CTMRG projector algorithm
    if alg == :c4v && projector_alg == Defaults.projector_alg
        projector_alg = Defaults.projector_alg_c4v
    end
    # check for full decomposition algorithm specification, otherwise interpret as forward alg
    if decomposition_alg isa NamedTuple
        decomposition_alg = (; fwd_alg = decomposition_alg)
    end
    projector_algorithm = ProjectorAlgorithm(;
        alg = projector_alg, decomposition_alg, trunc, verbosity
    )

    return alg_type(tol, maxiter, miniter, verbosity, projector_algorithm)
end

"""
    ctmrg_iteration(network, env, alg::CTMRGAlgorithm) -> env′, info

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(network, env, alg::CTMRGAlgorithm) end

"""
    leading_boundary(env₀, network; kwargs...) -> env, info
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
    - `:simultaneous` : Simultaneous expansion and renormalization of all sides.
    - `:sequential` : Sequential application of left moves and rotations.
    - `:c4v` : CTMRG assuming C₄ᵥ-symmetric PEPS and environment.

### Projector algorithm

* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation strategy for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `projector_alg::Symbol=:$(Defaults.projector_alg)` : Variant of the projector algorithm. See also [`ProjectorAlgorithm`](@ref).
    - `:halfinfinite` : Projection via SVDs of half-infinite (two enlarged corners) CTMRG environments.
    - `:fullinfinite` : Projection via SVDs of full-infinite (all four enlarged corners) CTMRG environments.
    - `:c4v_eigh` : Projection via `eigh` of the Hermitian enlarged corner, works only for [`C4vCTMRG`](@ref).
    - `:c4v_qr` : Projection via QR decomposition of the lower-rank column-enlarged corner, works only for [`C4vCTMRG`](@ref).
* `decomposition_alg::Union{NamedTuple,<:SVDAdjoint,<:EighAdjoint,<:QRAdjoint}` : Tensor
  decomposition algorithm used for computing projectors. When specified as a `NamedTuple`,
  the settings are passed a the forward algorithm to the appropriate decomposition
  for the given projector algorithm. For information on which forward algorithms are
  available, and how to specify them, see [`SVDAdjoint`](@ref), [`EighAdjoint`](@ref) and [`QRAdjoint`](@ref).

## Return values

The `leading_boundary` routine returns the final environment as well as an information `NamedTuple`
that generally contains a `contraction_metrics` `NamedTuple` storing different contents depending
on the chosen `alg`. Depending on the contraction method, the information tuple may also contain
the final tensor decomposition (used in the projectors) including its truncation indices.
"""
function leading_boundary(env₀::CTMRGEnv, network::InfiniteSquareNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return leading_boundary(env₀, network, alg)
end
function leading_boundary(
        env₀::CTMRGEnv, network::InfiniteSquareNetwork, alg::CTMRGAlgorithm
    )
    check_input(leading_boundary, network, env₀, alg)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))
    return LoggingExtras.withlevel(; alg.verbosity) do
        env = deepcopy(env₀)
        CS, TS = ignore_derivatives() do
            return map(svd_vals, env₀.corners), map(svd_vals, env₀.edges)
        end
        η = one(real(scalartype(network)))
        ctmrg_loginit!(log, η, network, env₀)
        local info
        for iter in 1:(alg.maxiter)
            env, info = ctmrg_iteration(network, env, alg)
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

"""
    check_input(::typeof(leading_boundary), network, env, alg::CTMRGAlgorithm)

Check compatibility of a given network and environment with a specified CTMRG algorithm.
"""
function check_input(::typeof(leading_boundary), network, env, alg::CTMRGAlgorithm) end
@non_differentiable check_input(args...)

# custom CTMRG logging
function ctmrg_loginit!(log, η, network, env)
    return @infov 2 loginit!(log, η, network_value(network, env))
end
function ctmrg_logiter!(log, iter, η, network, env)
    return @infov 3 logiter!(log, iter, η, network_value(network, env))
end
function ctmrg_logfinish!(log, iter, η, network, env)
    return @infov 2 logfinish!(log, iter, η, network_value(network, env))
end
function ctmrg_logcancel!(log, iter, η, network, env)
    return @warnv 1 logcancel!(log, iter, η, network_value(network, env))
end

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

"""
    _singular_value_distance(S₁, S₂)

Compute the singular value distance as an error measure, e.g. for CTMRG iterations.
To that end, the singular values of the current iteration `S₁` are compared with the
previous one `S₂`. When the virtual spaces change, this comparison is not directly possible
such that both tensors are projected into the smaller space and then subtracted.
"""
function _singular_value_distance(S₁::SV, S₂::SV) where {SV <: TensorKit.SectorVector}
    # allocate vector for difference - possibly grow
    V₁ = Vect[sectortype(S₁)](c => length(v) for (c, v) in blocks(S₁))
    V₂ = Vect[sectortype(S₂)](c => length(v) for (c, v) in blocks(S₂))
    diff = zerovector!(SV(undef, supremum(V₁, V₂)))

    for (c, b) in blocks(S₁)
        diff[c][1:length(b)] .= b
    end
    for (c, b) in blocks(S₂)
        diff[c][1:length(b)] .-= b
    end

    return norm(diff)
end
_singular_value_distance(S₁::DiagonalTensorMap, S₂::DiagonalTensorMap) =
    _singular_value_distance(diagview(S₁), diagview(S₂))

"""
    calc_convergence(env, CS_old, TS_old)
    calc_convergence(env_new, env_old)

Given a new environment `env`, compute the maximal singular value distance.
This determined either from the previous corner and edge singular values
`CS_old` and `TS_old`, or alternatively, directly from the old environment.
"""
function calc_convergence(env, CS_old, TS_old)
    CS_new = map(svd_vals, env.corners)
    ΔCS = maximum(splat(_singular_value_distance), zip(CS_old, CS_new))

    TS_new = map(svd_vals, env.edges)
    ΔTS = maximum(splat(_singular_value_distance), zip(TS_old, TS_new))

    @debug "maxᵢ|Cⁿ⁺¹ - Cⁿ|ᵢ = $ΔCS   maxᵢ|Tⁿ⁺¹ - Tⁿ|ᵢ = $ΔTS"

    return max(ΔCS, ΔTS), CS_new, TS_new
end
function calc_convergence(env_new::CTMRGEnv, env_old::CTMRGEnv)
    CS_old = map(svd_vals, env_old.corners)
    TS_old = map(svd_vals, env_old.edges)
    return calc_convergence(env_new, CS_old, TS_old)
end
@non_differentiable calc_convergence(args...)

"""
    CTMRGAlgorithm

Abstract super type for the corner transfer matrix renormalization group (CTMRG) algorithm
for contracting infinite PEPS.
"""
abstract type CTMRGAlgorithm end

"""
    ctmrg_iteration(state, env, alg::CTMRGAlgorithm) -> env′, info

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(state, env, alg::CTMRGAlgorithm) end

"""
    MPSKit.leading_boundary([env₀], state; kwargs...)
    # expert version:
    MPSKit.leading_boundary([env₀], state, alg::CTMRGAlgorithm)

Contract `state` using CTMRG and return the CTM environment. Per default, a random
initial environment is used.

The algorithm can be supplied via the keyword arguments or directly as an `CTMRGAlgorithm`
struct. The following keyword arguments are supported:

- `alg=SimultaneousCTMRG`: Variant of the CTMRG algorithm; can be any `CTMRGAlgorithm` type

- `tol=Defaults.ctmrg_tol`: Tolerance checking singular value and norm convergence; also
  sets related tolerances to sensible defaults unless they are explicitly specified

- `maxiter=Defaults.ctmrg_maxiter`: Maximal number of CTMRG iterations per run

- `miniter=Defaults.ctmrg_miniter`: Minimal number of CTMRG carried out

- `verbosity=2`: Overall output information verbosity level, where `0` suppresses
  all output, `1` only prints warnings, `2` gives information at the start and end,
  `3` prints information every iteration, and `4` gives extra debug information

- `trscheme=Defaults.trscheme`: SVD truncation scheme during projector computation; can be
  any `TruncationScheme` supported by the provided SVD algorithm

- `svd_alg=Defaults.svd_fwd_alg`: SVD algorithm used for computing projectors

- `svd_rrule_alg=Defaults.svd_rrule_alg_type`: Algorithm for differentiating SVDs; currently
  supported through KrylovKit where `GMRES`, `BiCGStab` and `Arnoldi` are supported (only
  relevant if `leading_boundary` is differentiated)

- `svd_rrule_tol=1e1tol`: Convergence tolerance for SVD reverse-rule algorithm (only
  relevant if `leading_boundary` is differentiated)

- `projector_alg=Defaults.projector_alg_type`: Projector algorithm type, where any
  `ProjectorAlgorithm` can be used
"""
function MPSKit.leading_boundary(state::InfiniteSquareNetwork; kwargs...)
    return MPSKit.leading_boundary(
        CTMRGEnv(state, oneunit(spacetype(state))), state; kwargs...
    )
end
function MPSKit.leading_boundary(env₀, state::InfiniteSquareNetwork; kwargs...)
    alg = select_algorithm(leading_boundary, env₀; kwargs...)
    return MPSKit.leading_boundary(env₀, state, alg)
end
function MPSKit.leading_boundary(state::InfiniteSquareNetwork, alg::CTMRGAlgorithm)
    return MPSKit.leading_boundary(CTMRGEnv(state, oneunit(spacetype(state))), state, alg)
end
function MPSKit.leading_boundary(env₀, state::InfiniteSquareNetwork, alg::CTMRGAlgorithm)
    CS = map(x -> tsvd(x)[2], env₀.corners)
    TS = map(x -> tsvd(x)[2], env₀.edges)

    η = one(real(scalartype(state)))
    env = deepcopy(env₀)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    return LoggingExtras.withlevel(; alg.verbosity) do
        ctmrg_loginit!(log, η, state, env₀)
        local info
        for iter in 1:(alg.maxiter)
            env, info = ctmrg_iteration(state, env, alg)  # Grow and renormalize in all 4 directions
            η, CS, TS = calc_convergence(env, CS, TS)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, state, env)
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, state, env)
            else
                ctmrg_logiter!(log, iter, η, state, env)
            end
        end
        return env, info
    end
end

# network-specific objective functions
ctmrg_objective(state::InfinitePEPS, env::CTMRGEnv) = norm(state, env)
ctmrg_objective(state::InfinitePartitionFunction, env::CTMRGEnv) = value(state, env)

# custom CTMRG logging
ctmrg_loginit!(log, η, state, env) = @infov 2 loginit!(log, η, ctmrg_objective(state, env))
function ctmrg_logiter!(log, iter, η, state, env)
    @infov 3 logiter!(log, iter, η, ctmrg_objective(state, env))
end
function ctmrg_logfinish!(log, iter, η, state, env)
    @infov 2 logfinish!(log, iter, η, ctmrg_objective(state, env))
end
function ctmrg_logcancel!(log, iter, η, state, env)
    @warnv 1 logcancel!(log, iter, η, ctmrg_objective(state, env))
end

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

"""
    select_algorithm(
        ::typeof(leading_boundary),
        env₀::CTMRGEnv;
        alg=SimultaneousCTMRG,
        tol=Defaults.ctmrg_tol,
        maxiter=Defaults.ctmrg_maxiter,
        miniter=Defaults.ctmrg_miniter,
        verbosity=2,
        trscheme=Defaults.trscheme,
        svd_alg=Defaults.svd_fwd_alg,
        svd_rrule_alg=Defaults.svd_rrule_type,
        svd_rrule_tol=1e1tol,
        projector_alg=Defaults.projector_alg_type,
    )

Parse CTMRG keyword arguments on to the corresponding algorithm structs and return a final
algorithm to be used in `leading_boundary`. For a description of the keyword arguments,
see [`leading_boundary`](@ref).
"""
function select_algorithm(
    ::typeof(leading_boundary),
    env₀::CTMRGEnv;
    alg=SimultaneousCTMRG,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=2,
    trscheme=Defaults.trscheme,
    svd_alg=Defaults.svd_fwd_alg,
    svd_rrule_alg=Defaults.svd_rrule_type,
    svd_rrule_tol=1e1tol,
    projector_alg=Defaults.projector_alg_type,
)
    # extract maximal environment dimenions
    χenv = maximum(env₀.corners) do corner
        return dim(space(corner, 1))
    end

    svd_rrule_algorithm = if svd_rrule_alg <: Union{GMRES,Arnoldi}
        svd_rrule_alg(; tol=svd_rrule_tol, krylovdim=χenv + 24, verbosity=verbosity - 2)
    elseif svd_rrule_alg <: BiCGStab
        svd_rrule_alg(; tol=svd_rrule_tol, verbosity)
    end
    svd_algorithm = SVDAdjoint(; fwd_alg=svd_alg, rrule_alg=svd_rrule_algorithm)
    projector_algorithm = projector_alg(svd_algorithm, trscheme, verbosity)
    return alg(tol, maxiter, miniter, verbosity, projector_algorithm)
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

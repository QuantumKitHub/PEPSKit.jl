"""
    FixedSpaceTruncation <: TensorKit.TruncationScheme

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

"""
    struct HalfInfiniteProjector{S,T}(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.
"""
@kwdef struct HalfInfiniteProjector{S<:SVDAdjoint,T}
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

"""
    struct FullInfiniteProjector{S,T}(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.
"""
@kwdef struct FullInfiniteProjector{S<:SVDAdjoint,T}
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

# TODO: do AbstractProjectorAlg type instead? -> would make it easier for users to implement custom projector alg
const ProjectorAlgs = Union{HalfInfiniteProjector,FullInfiniteProjector}

function svd_algorithm(alg::ProjectorAlgs, (dir, r, c))
    if alg.svd_alg isa SVDAdjoint{<:FixedSVD}
        fwd_alg = alg.svd_alg.fwd_alg
        fix_svd = FixedSVD(fwd_alg.U[dir, r, c], fwd_alg.S[dir, r, c], fwd_alg.V[dir, r, c])
        return SVDAdjoint(; fwd_alg=fix_svd, rrule_alg=alg.svd_alg.rrule_alg)
    else
        return alg.svd_alg
    end
end

function truncation_scheme(alg::ProjectorAlgs, Espace)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(Espace)
    else
        return alg.trscheme
    end
end

"""
    CTMRG(; tol=Defaults.ctmrg_tol, maxiter=Defaults.ctmrg_maxiter,
          miniter=Defaults.ctmrg_miniter, flavor=Defaults.ctmrg_flavor, verbosity=0,
          svd_alg=SVDAdjoint(), trscheme=FixedSpaceTruncation())

Algorithm struct that represents the CTMRG algorithm for contracting infinite PEPS.
Each CTMRG run is converged up to `tol` where the singular value convergence of the
corners as well as the norm is checked. The maximal and minimal number of CTMRG iterations
is set with `maxiter` and `miniter`.

In general, two different flavors of CTMRG can be selected with `flavor` which determine how
CTMRG is implemented. It can either be `:sequential`, where the projectors are succesively
computed on the west side, and then applied and rotated. Or with `:simultaneous` all projectors
are computed and applied simultaneously on all sides, where the corners get contracted with
two projectors at the same time.

Different levels of output information are printed depending on `verbosity`, where `0`
suppresses all output, `1` only prints warnings, `2` gives information at the start and
end, and `3` prints information every iteration.

The projectors are computed from `svd_alg` SVDs where the truncation scheme is set via 
`trscheme`.
"""
struct CTMRG
    tol::Float64
    maxiter::Int
    miniter::Int
    flavor::Symbol
    verbosity::Int
    projector_alg::ProjectorAlgs
end
function CTMRG(;
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    flavor=Defaults.ctmrg_flavor,
    verbosity=2,
    projector_alg=Defaults.projector_alg,
    svd_alg=Defaults.svd_alg,
    trscheme=Defaults.trscheme,
)
    return CTMRG(
        tol,
        maxiter,
        miniter,
        flavor,
        verbosity,
        projector_alg(; svd_alg, trscheme, verbosity),
    )
end

"""
    ctmrg_iteration(state, env, alg::CTMRG)

Perform a single CTMRG iteration in which all directions are being grown and renormalized.
"""
function ctmrg_iteration(state, env, alg::CTMRG)
    if alg.flavor == :simultaneous
        return simultaneous_ctmrg_iter(state, env, alg)
    elseif alg.flavor == :sequential
        return sequential_ctmrg_iter(state, env, alg)
    end
end

"""
    MPSKit.leading_boundary([envinit], state, alg::CTMRG)

Contract `state` using CTMRG and return the CTM environment.
Per default, a random initial environment is used.
"""
function MPSKit.leading_boundary(state, alg::CTMRG)
    return MPSKit.leading_boundary(CTMRGEnv(state, oneunit(spacetype(state))), state, alg)
end
function MPSKit.leading_boundary(envinit, state, alg::CTMRG)
    CS = map(x -> tsvd(x)[2], envinit.corners)
    TS = map(x -> tsvd(x)[2], envinit.edges)

    η = one(real(scalartype(state)))
    N = norm(state, envinit)
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    return LoggingExtras.withlevel(; alg.verbosity) do
        ctmrg_loginit!(log, η, N)
        for iter in 1:(alg.maxiter)
            env, = ctmrg_iteration(state, env, alg)  # Grow and renormalize in all 4 directions
            η, CS, TS = calc_convergence(env, CS, TS)
            N = norm(state, env)

            if η ≤ alg.tol && iter ≥ alg.miniter
                ctmrg_logfinish!(log, iter, η, N)
                break
            end
            if iter == alg.maxiter
                ctmrg_logcancel!(log, iter, η, N)
            else
                ctmrg_logiter!(log, iter, η, N)
            end
        end
        return env
    end
end

# custom CTMRG logging
ctmrg_loginit!(log, η, N) = @infov 2 loginit!(log, η, N)
ctmrg_logiter!(log, iter, η, N) = @infov 3 logiter!(log, iter, η, N)
ctmrg_logfinish!(log, iter, η, N) = @infov 2 logfinish!(log, iter, η, N)
ctmrg_logcancel!(log, iter, η, N) = @warnv 1 logcancel!(log, iter, η, N)

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

"""
    compute_projector(enlarged_corners, coordinate, alg::ProjectorAlgs)

Determine left and right projectors at the bond given determined by the enlarged corners
and the given coordinate using the specified `alg`.
"""
function compute_projector(enlarged_corners, coordinate, alg::HalfInfiniteProjector)
    # SVD half-infinite environment
    halfinf = half_infinite_environment(enlarged_corners...)
    trscheme = truncation_scheme(alg, space(enlarged_corners[2], 1))
    svd_alg = svd_algorithm(alg, coordinate)
    U, S, V, err = PEPSKit.tsvd!(halfinf, svd_alg; trunc=trscheme)

    # Compute SVD truncation error and check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = left_and_right_projector(U, S, V, enlarged_corners...)
    return (P_left, P_right), (; err, U, S, V)
end
function compute_projector(enlarged_corners, coordinate, alg::FullInfiniteProjector)
    # QR left and right half-infinite environments
    halfinf_left = half_infinite_environment(enlarged_corners[1], enlarged_corners[2])
    halfinf_right = half_infinite_environment(enlarged_corners[3], enlarged_corners[4])
    _, R_left = leftorth!(halfinf_left)
    L_right, _ = rightorth!(halfinf_right)

    # SVD product of QRs
    fullinf = R_left * L_right
    trscheme = truncation_scheme(alg, space(enlarged_corners[4], 1))
    svd_alg = svd_algorithm(alg, coordinate)
    U, S, V, err = PEPSKit.tsvd!(fullinf, svd_alg; trunc=trscheme)

    # Compute SVD truncation error and check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = left_and_right_projector(U, S, V, R_left, L_right)
    return (P_left, P_right), (; err, U, S, V)
end

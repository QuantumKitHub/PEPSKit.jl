"""
    FixedSpaceTruncation <: TensorKit.TruncationScheme

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

"""
    struct ProjectorAlg{S}(; svd_alg=TensorKit.SVD(), trscheme=TensorKit.notrunc(),
                           fixedspace=false, verbosity=0)

Algorithm struct collecting all projector related parameters. The truncation scheme has to be
a `TensorKit.TruncationScheme`, and some SVD algorithms might have further restrictions on what
kind of truncation scheme can be used. If `fixedspace` is true, the truncation scheme is set to
`truncspace(V)` where `V` is the environment bond space, adjusted to the corresponding
environment direction/unit cell entry.
"""
@kwdef struct ProjectorAlg{S<:SVDAdjoint,T}
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end
# TODO: add option for different projector styles (half-infinite, full-infinite, etc.)

function truncation_scheme(alg::ProjectorAlg, edge)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(space(edge, 1))
    else
        return alg.trscheme
    end
end

function svd_algorithm(alg::ProjectorAlg, (dir, r, c))
    if alg.svd_alg isa SVDAdjoint{<:FixedSVD}
        fwd_alg = alg.svd_alg.fwd_alg
        fix_svd = FixedSVD(fwd_alg.U[dir, r, c], fwd_alg.S[dir, r, c], fwd_alg.V[dir, r, c])
        return SVDAdjoint(; fwd_alg=fix_svd, rrule_alg=alg.svd_alg.rrule_alg)
    else
        return alg.svd_alg
    end
end

"""
    CTMRG(; tol=Defaults.ctmrg_tol, maxiter=Defaults.ctmrg_maxiter,
          miniter=Defaults.ctmrg_miniter, verbosity=0,
          svd_alg=SVDAdjoint(), trscheme=FixedSpaceTruncation(),
          ctmrgscheme=Defaults.ctmrgscheme)

Algorithm struct that represents the CTMRG algorithm for contracting infinite PEPS.
Each CTMRG run is converged up to `tol` where the singular value convergence of the
corners as well as the norm is checked. The maximal and minimal number of CTMRG iterations
is set with `maxiter` and `miniter`. Different levels of output information are printed
depending on `verbosity`, where `0` suppresses all output, `1` only prints warnings, `2`
gives information at the start and end, and `3` prints information every iteration.

The projectors are computed from `svd_alg` SVDs where the truncation scheme is set via 
`trscheme`.

In general, two different schemes can be selected with `ctmrgscheme` which determine how
CTMRG is implemented. It can either be `:sequential`, where the projectors are succesively
computed on the western side, and then applied and rotated. Or with `:simultaneous` all projectors
are computed and applied simultaneously on all sides, where in particular the corners get
contracted with two projectors at the same time.
"""
struct CTMRG{S}
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::ProjectorAlg
end
function CTMRG(;
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=2,
    svd_alg=Defaults.svd_alg,
    trscheme=Defaults.trscheme,
    ctmrgscheme::Symbol=Defaults.ctmrgscheme,
)
    return CTMRG{ctmrgscheme}(
        tol, maxiter, miniter, verbosity, ProjectorAlg(; svd_alg, trscheme, verbosity)
    )
end

ctmrgscheme(::CTMRG{S}) where {S} = S

# aliases for the different CTMRG schemes
const SequentialCTMRG = CTMRG{:sequential}
const SimultaneousCTMRG = CTMRG{:simultaneous}

# supply correct constructor for Accessors.@set
Accessors.constructorof(::Type{CTMRG{S}}) where {S} = CTMRG{S}

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
            env, = ctmrg_iter(state, env, alg)  # Grow and renormalize in all 4 directions
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

"""
Perform CTMRG left move on the `col`-th column
"""
function ctmrg_leftmove(col::Int, state, envs::CTMRGEnv, alg::SequentialCTMRG)
    #=
        ----> left move
        C1 ← T1 ←   r-1
        ↓    ‖
        T4 = M ==   r
        ↓    ‖
        C4 → T3 →   r+1
        c-1  c 
    =#
    projectors, info = ctmrg_projectors(col, state, envs, alg)
    envs = ctmrg_renormalize(col, projectors, state, envs, alg)
    return envs, info
end
"""
    ctmrg_iter(state, envs::CTMRGEnv, alg::CTMRG) -> envs′, info

Perform one iteration of CTMRG that maps the `state` and `envs` to a new environment,
and also returns the `info` `NamedTuple`.
"""
function ctmrg_iter(state, envs::CTMRGEnv, alg::SequentialCTMRG)
    ϵ = zero(real(scalartype(state)))
    for _ in 1:4 # rotate
        for col in 1:size(state, 2) # left move column-wise
            envs, info = ctmrg_leftmove(col, state, envs, alg)
            ϵ = max(ϵ, info.err)
        end
        state = rotate_north(state, EAST)
        envs = rotate_north(envs, EAST)
    end
    return envs, (; err=ϵ)
end
function ctmrg_iter(state, envs::CTMRGEnv, alg::SimultaneousCTMRG)
    enlarged_envs = ctmrg_expand(eachcoordinate(state, 1:4), state, envs)
    projectors, info = ctmrg_projectors(enlarged_envs, envs, alg)
    envs′ = ctmrg_renormalize(enlarged_envs, projectors, state, envs, alg)
    return envs′, info
end

ctmrg_loginit!(log, η, N) = @infov 2 loginit!(log, η, N)
ctmrg_logiter!(log, iter, η, N) = @infov 3 logiter!(log, iter, η, N)
ctmrg_logfinish!(log, iter, η, N) = @infov 2 logfinish!(log, iter, η, N)
ctmrg_logcancel!(log, iter, η, N) = @warnv 1 logcancel!(log, iter, η, N)

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

# ======================================================================================== #
# Expansion step
# ======================================================================================== #

"""
    ctmrg_expand(coordinates, state, envs)

Expand the environment by absorbing a new PEPS tensor on the given coordinates.
"""
function ctmrg_expand(coordinates, state, envs::CTMRGEnv)
    return dtmap(idx -> TensorMap(EnlargedCorner(state, envs, idx), idx[1]), coordinates)
end

# ======================================================================================== #
# Projector step
# ======================================================================================== #

"""
    ctmrg_projectors(col::Int, enlarged_envs, env, alg::CTMRG{:sequential})
    ctmrg_projectors(enlarged_envs, env, alg::CTMRG{:simultaneous})

Compute the CTMRG projectors based on enlarged environments.
In the `:sequential` mode the projectors are computed for the column `col`, whereas
in the `:simultaneous` mode, all projectors (and corresponding SVDs) are computed in parallel.
"""
function ctmrg_projectors(
    col::Int, state::InfinitePEPS, envs::CTMRGEnv{C,E}, alg::SequentialCTMRG
) where {C,E}
    projector_alg = alg.projector_alg
    ϵ = zero(real(scalartype(envs)))

    # SVD half-infinite environment
    coordinates = eachcoordinate(envs)[:, col]
    projectors = dtmap(coordinates) do (r, c)
        r′ = _prev(r, size(envs.corners, 2))
        Q1 = TensorMap(EnlargedCorner(state, envs, (SOUTHWEST, r, c)), SOUTHWEST)
        Q2 = TensorMap(EnlargedCorner(state, envs, (NORTHWEST, r′, c)), NORTHWEST)
        QQ = halfinfinite_environment(Q1, Q2)
        trscheme = truncation_scheme(projector_alg, envs.edges[WEST, r′, c])
        svd_alg = svd_algorithm(projector_alg, (WEST, r, c))
        U, S, V, ϵ_local = PEPSKit.tsvd!(QQ, svd_alg; trunc=trscheme)
        ϵ = max(ϵ, ϵ_local / norm(S))

        # Compute SVD truncation error and check for degenerate singular values
        Zygote.isderiving() && ignore_derivatives() do
            if alg.verbosity > 0 && is_degenerate_spectrum(S)
                svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
                @warn("degenerate singular values detected: ", svals)
            end
        end

        # Compute projectors
        return build_projectors(U, S, V, Q1, Q2)
    end
    return (map(first, projectors), map(last, projectors)), (; err=ϵ)
end
function ctmrg_projectors(
    enlarged_envs, envs::CTMRGEnv{C,E}, alg::SimultaneousCTMRG
) where {C,E}
    projector_alg = alg.projector_alg
    # pre-allocation
    U, V = Zygote.Buffer.(projector_type(envs.edges))
    # Corner type but with real numbers
    S = Zygote.Buffer(U.data, tensormaptype(spacetype(C), 1, 1, real(scalartype(E))))

    ϵ = zero(real(scalartype(envs)))
    coordinates = eachcoordinate(envs, 1:4)
    projectors = dtmap(coordinates) do (dir, r, c)
        # Row-column index of next enlarged corner
        next_rc = if dir == 1
            (r, _next(c, size(envs.corners, 3)))
        elseif dir == 2
            (_next(r, size(envs.corners, 2)), c)
        elseif dir == 3
            (r, _prev(c, size(envs.corners, 3)))
        elseif dir == 4
            (_prev(r, size(envs.corners, 2)), c)
        end

        # SVD half-infinite environment
        QQ = halfinfinite_environment(
            enlarged_envs[dir, r, c], enlarged_envs[_next(dir, 4), next_rc...]
        )

        trscheme = truncation_scheme(projector_alg, envs.edges[dir, next_rc...])
        svd_alg = svd_algorithm(projector_alg, (dir, r, c))
        U_local, S_local, V_local, ϵ_local = PEPSKit.tsvd!(QQ, svd_alg; trunc=trscheme)
        U[dir, r, c] = U_local
        S[dir, r, c] = S_local
        V[dir, r, c] = V_local
        ϵ = max(ϵ, ϵ_local / norm(S_local))

        # Compute SVD truncation error and check for degenerate singular values
        Zygote.isderiving() && ignore_derivatives() do
            if alg.verbosity > 0 && is_degenerate_spectrum(S_local)
                svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S_local))
                @warn("degenerate singular values detected: ", svals)
            end
        end

        # Compute projectors
        return build_projectors(
            U_local,
            S_local,
            V_local,
            enlarged_envs[dir, r, c],
            enlarged_envs[_next(dir, 4), next_rc...],
        )
    end

    P_left = map(first, projectors)
    P_right = map(last, projectors)
    return (P_left, P_right), (; err=ϵ, U=copy(U), S=copy(S), V=copy(V))
end

"""
    build_projectors(U::AbstractTensorMap{E,3,1}, S::AbstractTensorMap{E,1,1}, V::AbstractTensorMap{E,1,3},
        Q::AbstractTensorMap{E,3,3}, Q_next::AbstractTensorMap{E,3,3}) where {E<:ElementarySpace}
    build_projectors(U::AbstractTensorMap{E,3,1}, S::AbstractTensorMap{E,1,1}, V::AbstractTensorMap{E,1,3},
        Q::EnlargedCorner, Q_next::EnlargedCorner) where {E<:ElementarySpace}

Construct left and right projectors where the higher-dimensional is facing left and right, respectively.
"""
function build_projectors(
    U::AbstractTensorMap{E,3,1},
    S::AbstractTensorMap{E,1,1},
    V::AbstractTensorMap{E,1,3},
    Q::AbstractTensorMap{E,3,3},
    Q_next::AbstractTensorMap{E,3,3},
) where {E<:ElementarySpace}
    isqS = sdiag_pow(S, -0.5)
    P_left = Q_next * V' * isqS
    P_right = isqS * U' * Q
    return P_left, P_right
end
function build_projectors(
    U::AbstractTensorMap{E,3,1},
    S::AbstractTensorMap{E,1,1},
    V::AbstractTensorMap{E,1,3},
    Q::EnlargedCorner,
    Q_next::EnlargedCorner,
) where {E<:ElementarySpace}
    isqS = sdiag_pow(S, -0.5)
    P_left = left_projector(Q.E_1, Q.C, Q.E_2, V, isqS, Q.ket, Q.bra)
    P_right = right_projector(
        Q_next.E_1, Q_next.C, Q_next.E_2, U, isqS, Q_next.ket, Q_next.bra
    )
    return P_left, P_right
end

# ======================================================================================== #
# Renormalization step
# ======================================================================================== #

"""
    ctmrg_renormalize(col::Int, projectors, state, envs, ::CTMRG{:sequential})
    ctmrg_renormalize(enlarged_envs, projectors, state, envs, ::CTMRG{:simultaneous})

Apply projectors to renormalize corners and edges.
The `:sequential` mode renormalizes the environment on the column `col`, where as the
`:simultaneous` mode renormalizes all environment tensors simultaneously.
"""
function ctmrg_renormalize(col::Int, projectors, state, envs, ::SequentialCTMRG)
    corners = Zygote.Buffer(envs.corners)
    edges = Zygote.Buffer(envs.edges)

    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir in [SOUTHWEST, NORTHWEST]) && continue
        corners[dir, r, c] = envs.corners[dir, r, c]
    end
    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = envs.edges[dir, r, c]
    end

    # Apply projectors to renormalize corners and edge
    for row in axes(envs.corners, 2)
        C_southwest = renormalize_bottom_corner((row, col), envs, projectors)
        corners[SOUTHWEST, row, col] = C_southwest / norm(C_southwest)

        C_northwest = renormalize_top_corner((row, col), envs, projectors)
        corners[NORTHWEST, row, col] = C_northwest / norm(C_northwest)

        E_west = renormalize_west_edge((row, col), envs, projectors, state)
        edges[WEST, row, col] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end
function ctmrg_renormalize(enlarged_envs, projectors, state, envs, ::SimultaneousCTMRG)
    P_left, P_right = projectors
    coordinates = eachcoordinate(envs, 1:4)
    corners_edges = dtmap(coordinates) do (dir, r, c)
        if dir == NORTH
            corner = renormalize_northwest_corner((r, c), enlarged_envs, P_left, P_right)
            edge = renormalize_north_edge((r, c), envs, P_left, P_right, state)
        elseif dir == EAST
            corner = renormalize_northeast_corner((r, c), enlarged_envs, P_left, P_right)
            edge = renormalize_east_edge((r, c), envs, P_left, P_right, state)
        elseif dir == SOUTH
            corner = renormalize_southeast_corner((r, c), enlarged_envs, P_left, P_right)
            edge = renormalize_south_edge((r, c), envs, P_left, P_right, state)
        elseif dir == WEST
            corner = renormalize_southwest_corner((r, c), enlarged_envs, P_left, P_right)
            edge = renormalize_west_edge((r, c), envs, P_left, P_right, state)
        end
        return corner / norm(corner), edge / norm(edge)
    end

    return CTMRGEnv(map(first, corners_edges), map(last, corners_edges))
end

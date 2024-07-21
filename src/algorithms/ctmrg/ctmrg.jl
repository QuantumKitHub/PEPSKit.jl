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
    svd_alg::S = SVDAdjoint()
    trscheme::T = FixedSpaceTruncation()
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
computed on the western side, and then applied and rotated. Or with `simultaneous` all projectors
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
    svd_alg=SVDAdjoint(),
    trscheme=FixedSpaceTruncation(),
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

"""
    MPSKit.leading_boundary([envinit], state, alg::CTMRG)

Contract `state` using CTMRG and return the CTM environment.
Per default, a random initial environment is used.
"""
function MPSKit.leading_boundary(state, alg::CTMRG)
    return MPSKit.leading_boundary(CTMRGEnv(state, oneunit(spacetype(state))), state, alg)
end
function MPSKit.leading_boundary(envinit, state, alg::CTMRG)
    CS = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.corners)
    TS = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.edges)

    η = one(real(scalartype(state)))
    N = norm(state, envinit)
    env = deepcopy(envinit)
    log = ignore_derivatives(() -> MPSKit.IterLog("CTMRG"))

    return LoggingExtras.withlevel(; alg.verbosity) do
        ctmrg_loginit!(log, η, N)
        local iter
        for outer iter in 1:(alg.maxiter)
            env, = ctmrg_iter(state, env, alg)  # Grow and renormalize in all 4 directions
            η, CS, TS = calc_convergence(env, CS, TS)
            N = norm(state, env)
            ctmrg_logiter!(log, iter, η, N)

            (iter > alg.miniter && η <= alg.tol) && break
        end

        # Do one final iteration that does not change the spaces
        alg_fixed = CTMRG(;
            verbosity=alg.verbosity,
            svd_alg=alg.projector_alg.svd_alg,
            trscheme=FixedSpaceTruncation(),
            ctmrgscheme=ctmrgscheme(alg),
        )
        env′, = ctmrg_iter(state, env, alg_fixed)
        envfix, = gauge_fix(env, env′)

        η = calc_elementwise_convergence(envfix, env; atol=alg.tol^(1 / 2))
        N = norm(state, envfix)

        if η < alg.tol^(1 / 2)
            ctmrg_logfinish!(log, iter, η, N)
        else
            ctmrg_logcancel!(log, iter, η, N)
        end
        return envfix
    end
end

"""
    ctmrg_iter(state, env::CTMRGEnv, alg::CTMRG{:sequential})
    
Perform one iteration of CTMRG that maps the `state` and `env` to a new environment,
and also return the truncation error.
One CTMRG iteration consists of four `left_move` calls and 90 degree rotations,
such that the environment is grown and renormalized in all four directions.
"""
function ctmrg_iter(state, envs::CTMRGEnv, alg::SequentialCTMRG)
    ϵ = 0.0
    for _ in 1:4
        # left move
        enlarged_envs = ctmrg_expand(state, envs, alg)
        projectors, info = ctmrg_projectors(enlarged_envs, envs, alg)
        envs = ctmrg_renormalize(enlarged_envs, projectors, state, envs, alg)

        # rotate
        state = rotate_north(state, EAST)
        envs = rotate_north(envs, EAST)
        ϵ = max(ϵ, info.err)
    end

    return envs, (; err=ϵ)
end
function ctmrg_iter(state, envs::CTMRGEnv, alg::SimultaneousCTMRG)
    enlarged_envs = ctmrg_expand(state, envs, alg)
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
    ctmrg_expand(state, envs, alg::CTMRG{M})

Expand the environment by absorbing a new PEPS tensor.
There are two modes of expansion: `M = :sequential` and `M = :simultaneous`.
The first mode expands the environment in one direction at a time, for convenience towards
the left. The second mode expands the environment in all four directions simultaneously.
"""
function ctmrg_expand(state, envs::CTMRGEnv{C,T}, ::SequentialCTMRG) where {C,T}
    Qtype = tensormaptype(spacetype(C), 3, 3, storagetype(C))
    Q_nw = Zygote.Buffer(envs.corners, Qtype, axes(state)...)
    Q_sw = Zygote.Buffer(envs.corners, Qtype, axes(state)...)

    directions = collect(Iterators.product(axes(state)...))
    @fwdthreads for (r, c) in directions
        r′ = _next(r, size(state, 1))
        Q_nw[r, c] = northwest_corner((r, c), envs, state)
        Q_sw[r, c] = southwest_corner((r′, c), envs, state)
    end

    return copy(Q_nw), copy(Q_sw)
end
function ctmrg_expand(state, envs::CTMRGEnv{C,T}, ::SimultaneousCTMRG) where {C,T}
    Qtype = tensormaptype(spacetype(C), 3, 3, storagetype(C))
    Q = Zygote.Buffer(Array{Qtype,3}(undef, size(envs.corners)))
    drc_combinations = collect(Iterators.product(axes(envs.corners)...))
    @fwdthreads for (dir, r, c) in drc_combinations
        Q[dir, r, c] = if dir == NORTHWEST
            northwest_corner((r, c), envs, state)
        elseif dir == NORTHEAST
            northeast_corner((r, c), envs, state)
        elseif dir == SOUTHEAST
            southeast_corner((r, c), envs, state)
        elseif dir == SOUTHWEST
            southwest_corner((r, c), envs, state)
        end
    end

    return copy(Q)
end

# ======================================================================================== #
# Projector step
# ======================================================================================== #

"""
    ctmrg_projectors(Q, env, alg::CTMRG{M})
"""
function ctmrg_projectors(
    enlarged_envs, envs::CTMRGEnv{C,E}, alg::SequentialCTMRG
) where {C,E}
    projector_alg = alg.projector_alg
    # pre-allocation
    Prtype = tensormaptype(spacetype(E), numin(E), numout(E), storagetype(E))
    P_bottom = Zygote.Buffer(envs.edges, axes(envs.corners, 2), axes(envs.corners, 3))
    P_top = Zygote.Buffer(envs.edges, Prtype, axes(envs.corners, 2), axes(envs.corners, 3))
    ϵ = 0.0

    directions = collect(Iterators.product(axes(envs.corners, 2), axes(envs.corners, 3)))
    @fwdthreads for (r, c) in directions
        # SVD half-infinite environment
        @autoopt @tensor QQ[χ_EB D_EBabove D_EBbelow; χ_ET D_ETabove D_ETbelow] :=
            enlarged_envs[2][r, c][χ_EB D_EBabove D_EBbelow; χ D1 D2] *
            enlarged_envs[1][r, c][χ D1 D2; χ_ET D_ETabove D_ETbelow]

        trscheme = truncation_scheme(projector_alg, envs.edges[WEST, r, c])
        svd_alg = svd_algorithm(projector_alg, (WEST, r, c))
        U, S, V, ϵ_local = PEPSKit.tsvd!(QQ, svd_alg; trunc=trscheme)
        ϵ = max(ϵ, ϵ_local / norm(S))

        # Compute SVD truncation error and check for degenerate singular values
        ignore_derivatives() do
            if alg.verbosity > 0 && is_degenerate_spectrum(S)
                svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
                @warn("degenerate singular values detected: ", svals)
            end
        end

        # Compute projectors
        P_bottom[r, c], P_top[r, c] = build_projectors(
            U, S, V, enlarged_envs[2][r, c], enlarged_envs[1][r, c]
        )
    end

    return (copy(P_bottom), copy(P_top)), (; err=ϵ)
end
function ctmrg_projectors(
    enlarged_envs, envs::CTMRGEnv{C,E}, alg::SimultaneousCTMRG
) where {C,E}
    projector_alg = alg.projector_alg
    # pre-allocation
    P_left, P_right = Zygote.Buffer.(projector_type(envs.edges))
    U, V = Zygote.Buffer.(projector_type(envs.edges))
    # Corner type but with real numbers
    S = Zygote.Buffer(U.data, tensormaptype(spacetype(C), 1, 1, real(scalartype(E))))

    ϵ = 0.0
    drc_combinations = collect(Iterators.product(axes(envs.corners)...))
    @fwdthreads for (dir, r, c) in drc_combinations
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
        @autoopt @tensor QQ[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
            enlarged_envs[dir, r, c][χ_in D_inabove D_inbelow; χ D1 D2] *
            enlarged_envs[_next(dir, 4), next_rc...][χ D1 D2; χ_out D_outabove D_outbelow]

        trscheme = truncation_scheme(projector_alg, envs.edges[dir, r, c])
        svd_alg = svd_algorithm(projector_alg, (dir, r, c))
        U_local, S_local, V_local, ϵ_local = PEPSKit.tsvd!(QQ, svd_alg; trunc=trscheme)
        U[dir, r, c] = U_local
        S[dir, r, c] = S_local
        V[dir, r, c] = V_local
        ϵ = max(ϵ, ϵ_local / norm(S_local))

        # Compute SVD truncation error and check for degenerate singular values
        ignore_derivatives() do
            if alg.verbosity > 0 && is_degenerate_spectrum(S_local)
                svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S_local))
                @warn("degenerate singular values detected: ", svals)
            end
        end

        # Compute projectors
        P_left[dir, r, c], P_right[dir, r, c] = build_projectors(
            U_local,
            S_local,
            V_local,
            enlarged_envs[dir, r, c],
            enlarged_envs[_next(dir, 4), next_rc...],
        )
    end

    return (copy(P_left), copy(P_right)), (; err=ϵ, U=copy(U), S=copy(S), V=copy(V))
end

# ======================================================================================== #
# Renormalization step
# ======================================================================================== #

"""
    ctmrg_renormalize(enlarged_envs, projectors, state, envs, alg::CTMRG{M})

Apply projectors to renormalize corners and edges.
"""
function ctmrg_renormalize(enlarged_envs, projectors, state, envs, alg::SequentialCTMRG)
    corners = Zygote.Buffer(envs.corners)
    edges = Zygote.Buffer(envs.edges)

    # copy environments that do not participate
    for dir in (NORTHEAST, SOUTHEAST)
        for r in axes(envs.corners, 2), c in axes(envs.corners, 3)
            corners[dir, r, c] = envs.corners[dir, r, c]
        end
    end
    for dir in (NORTH, EAST, SOUTH)
        for r in axes(envs.corners, 2), c in axes(envs.corners, 3)
            edges[dir, r, c] = envs.edges[dir, r, c]
        end
    end

    # Apply projectors to renormalize corners and edges
    coordinates = collect(Iterators.product(axes(state)...))
    @fwdthreads for (r, c) in coordinates
        r′ = _prev(r, size(state, 1))
        c′ = _prev(c, size(state, 2))
        C_sw, C_nw, T_w = grow_env_left(
            state[r, c],
            projectors[1][r′, c],
            projectors[2][r, c],
            envs.corners[SOUTHWEST, r, c′],
            envs.corners[NORTHWEST, r, c′],
            envs.edges[SOUTH, r, c],
            envs.edges[WEST, r, c′],
            envs.edges[NORTH, r, c],
        )
        corners[SOUTHWEST, r, c] = C_sw / norm(C_sw)
        corners[NORTHWEST, r, c] = C_nw / norm(C_nw)
        edges[WEST, r, c] = T_w / norm(T_w)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end
function ctmrg_renormalize(enlarged_envs, projectors, state, envs, alg::SimultaneousCTMRG)
    corners = Zygote.Buffer(envs.corners)
    edges = Zygote.Buffer(envs.edges)
    P_left, P_right = projectors

    coordinates = collect(Iterators.product(axes(state)...))
    @fwdthreads for (r, c) in coordinates
        rprev = _prev(r, size(state, 1))
        rnext = _next(r, size(state, 1))
        cprev = _prev(c, size(state, 2))
        cnext = _next(c, size(state, 2))

        C_northwest = _contract_new_corner(
            P_right[WEST, rnext, c], enlarged_envs[NORTHWEST, r, c], P_left[NORTH, r, c]
        )
        corners[NORTHWEST, r, c] = C_northwest / norm(C_northwest)

        C_northeast = _contract_new_corner(
            P_right[NORTH, r, cprev], enlarged_envs[NORTHEAST, r, c], P_left[EAST, r, c]
        )
        corners[NORTHEAST, r, c] = C_northeast / norm(C_northeast)

        C_southeast = _contract_new_corner(
            P_right[EAST, rprev, c], enlarged_envs[SOUTHEAST, r, c], P_left[SOUTH, r, c]
        )
        corners[SOUTHEAST, r, c] = C_southeast / norm(C_southeast)

        C_southwest = _contract_new_corner(
            P_right[SOUTH, r, cnext], enlarged_envs[SOUTHWEST, r, c], P_left[WEST, r, c]
        )
        corners[SOUTHWEST, r, c] = C_southwest / norm(C_southwest)

        @autoopt @tensor E_north[χ_W D_Sab D_Sbe; χ_E] :=
            envs.edges[NORTH, rprev, c][χ1 D1 D2; χ2] *
            state[r, c][d; D1 D3 D_Sab D5] *
            conj(state[r, c][d; D2 D4 D_Sbe D6]) *
            P_left[NORTH, r, c][χ2 D3 D4; χ_E] *
            P_right[NORTH, r, cprev][χ_W; χ1 D5 D6]
        edges[NORTH, r, c] = E_north / norm(E_north)

        @autoopt @tensor E_east[χ_N D_Wab D_Wbe; χ_S] :=
            envs.edges[EAST, r, cnext][χ1 D1 D2; χ2] *
            state[r, c][d; D5 D1 D3 D_Wab] *
            conj(state[r, c][d; D6 D2 D4 D_Wbe]) *
            P_left[EAST, r, c][χ2 D3 D4; χ_S] *
            P_right[EAST, rprev, c][χ_N; χ1 D5 D6]
        edges[EAST, r, c] = E_east / norm(E_east)

        @autoopt @tensor E_south[χ_E D_Nab D_Nbe; χ_W] :=
            envs.edges[SOUTH, rnext, c][χ1 D1 D2; χ2] *
            state[r, c][d; D_Nab D5 D1 D3] *
            conj(state[r, c][d; D_Nbe D6 D2 D4]) *
            P_left[SOUTH, r, c][χ2 D3 D4; χ_W] *
            P_right[SOUTH, r, cnext][χ_E; χ1 D5 D6]
        edges[SOUTH, r, c] = E_south / norm(E_south)

        @autoopt @tensor E_west[χ_S D_Eab D_Ebe; χ_N] :=
            envs.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            state[r, c][d; D3 D_Eab D5 D1] *
            conj(state[r, c][d; D4 D_Ebe D6 D2]) *
            P_left[WEST, r, c][χ2 D3 D4; χ_N] *
            P_right[WEST, rnext, c][χ_S; χ1 D5 D6]
        edges[WEST, r, c] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

# ======================================================================================== #
# Auxiliary routines
# ======================================================================================== #

# Compute enlarged corners and edges for all directions and unit cell entries
function enlarge_corners_edges(state, env::CTMRGEnv{C,T}) where {C,T}
    Qtype = tensormaptype(spacetype(C), 3, 3, storagetype(C))
    Q = Zygote.Buffer(Array{Qtype,3}(undef, size(env.corners)))
    drc_combinations = collect(Iterators.product(axes(env.corners)...))
    @fwdthreads for (dir, r, c) in drc_combinations
        Q[dir, r, c] = if dir == NORTHWEST
            northwest_corner((r, c), env, state)
        elseif dir == NORTHEAST
            northeast_corner((r, c), env, state)
        elseif dir == SOUTHEAST
            southeast_corner((r, c), env, state)
        elseif dir == SOUTHWEST
            southwest_corner((r, c), env, state)
        end
    end

    return copy(Q)
end

# Enlarged corner contractions (need direction specific methods to avoid PEPS rotations)
function northwest_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_S D_Sabove D_Sbelow; χ_E D_Eabove D_Ebelow] :=
        env.edges[WEST, row, _prev(col, end)][χ_S D1 D2; χ1] *
        env.corners[NORTHWEST, _prev(row, end), _prev(col, end)][χ1; χ2] *
        env.edges[NORTH, _prev(row, end), col][χ2 D3 D4; χ_E] *
        peps_above[row, col][d; D3 D_Eabove D_Sabove D1] *
        conj(peps_below[row, col][d; D4 D_Ebelow D_Sbelow D2])
end
function northeast_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_W D_Wabove D_Wbelow; χ_S D_Sabove D_Sbelow] :=
        env.edges[NORTH, _prev(row, end), col][χ_W D1 D2; χ1] *
        env.corners[NORTHEAST, _prev(row, end), _next(col, end)][χ1; χ2] *
        env.edges[EAST, row, _next(col, end)][χ2 D3 D4; χ_S] *
        peps_above[row, col][d; D1 D3 D_Sabove D_Wabove] *
        conj(peps_below[row, col][d; D2 D4 D_Sbelow D_Wbelow])
end
function southeast_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        env.edges[EAST, row, _next(col, end)][χ_N D1 D2; χ1] *
        env.corners[SOUTHEAST, _next(row, end), _next(col, end)][χ1; χ2] *
        env.edges[SOUTH, _next(row, end), col][χ2 D3 D4; χ_W] *
        peps_above[row, col][d; D_Nabove D1 D3 D_Wabove] *
        conj(peps_below[row, col][d; D_Nbelow D2 D4 D_Wbelow])
end
function southwest_corner((row, col), env, peps_above, peps_below=peps_above)
    return @autoopt @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        env.edges[SOUTH, _next(row, end), col][χ_E D1 D2; χ1] *
        env.corners[SOUTHWEST, _next(row, end), _prev(col, end)][χ1; χ2] *
        env.edges[WEST, row, _prev(col, end)][χ2 D3 D4; χ_N] *
        peps_above[row, col][d; D_Nabove D_Eabove D1 D3] *
        conj(peps_below[row, col][d; D_Nbelow D_Ebelow D2 D4])
end

function _contract_new_corner(P_right, Q, P_left)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] * Q[χ1 D1 D2; χ2 D3 D4] * P_left[χ2 D3 D4; χ_out]
end

# Build projectors from SVD and enlarged SW & NW corners
function build_projectors(
    U::AbstractTensorMap{E,3,1}, S, V::AbstractTensorMap{E,1,3}, Q, Q_next
) where {E<:ElementarySpace}
    isqS = sdiag_inv_sqrt(S)
    P_left = Q_next * V' * isqS
    P_right = isqS * U' * Q
    return P_left, P_right
end

# Apply projectors to entire left half-environment to grow SW & NW corners, and W edge
function grow_env_left(
    peps, P_bottom, P_top, corners_SW, corners_NW, edge_S, edge_W, edge_N
)
    @autoopt @tensor corner_SW′[χ_E; χ_N] :=
        corners_SW[χ1; χ2] * edge_S[χ_E D1 D2; χ1] * P_bottom[χ2 D1 D2; χ_N]
    @autoopt @tensor corner_NW′[χ_S; χ_E] :=
        corners_NW[χ1; χ2] * edge_N[χ2 D1 D2; χ_E] * P_top[χ_S; χ1 D1 D2]
    @autoopt @tensor edge_W′[χ_S D_Eabove D_Ebelow; χ_N] :=
        edge_W[χ1 D1 D2; χ2] *
        peps[d; D3 D_Eabove D5 D1] *
        conj(peps[d; D4 D_Ebelow D6 D2]) *
        P_bottom[χ2 D3 D4; χ_N] *
        P_top[χ_S; χ1 D5 D6]
    return corner_SW′, corner_NW′, edge_W′
end

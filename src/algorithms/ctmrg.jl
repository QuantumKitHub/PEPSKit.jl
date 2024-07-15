"""
    FixedSpaceTruncation <: TensorKit.TruncationScheme

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

# TODO: add option for different projector styles (half-infinite, full-infinite, etc.)
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

# TODO: add abstract Algorithm type?
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
    verbosity=1,
    svd_alg=SVDAdjoint(),
    trscheme=FixedSpaceTruncation(),
    ctmrgscheme=Defaults.ctmrgscheme,
)
    return CTMRG{ctmrgscheme}(
        tol, maxiter, miniter, verbosity, ProjectorAlg(; svd_alg, trscheme, verbosity)
    )
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
        alg_fixed = @set alg.projector_alg.trscheme = FixedSpaceTruncation()
        env′, = ctmrg_iter(state, env, alg_fixed)
        envfix = gauge_fix(env, env′)

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

ctmrg_loginit!(log, η, N) = @infov 2 loginit!(log, η, N)
ctmrg_logiter!(log, iter, η, N) = @infov 3 logiter!(log, iter, η, N)
ctmrg_logfinish!(log, iter, η, N) = @infov 2 logfinish!(log, iter, η, N)
ctmrg_logcancel!(log, iter, η, N) = @warnv 1 logcancel!(log, iter, η, N)

@non_differentiable ctmrg_loginit!(args...)
@non_differentiable ctmrg_logiter!(args...)
@non_differentiable ctmrg_logfinish!(args...)
@non_differentiable ctmrg_logcancel!(args...)

"""
    gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}

Fix the gauge of `envfinal` based on the previous environment `envprev`.
This assumes that the `envfinal` is the result of one CTMRG iteration on `envprev`.
Given that the CTMRG run is converged, the returned environment will be
element-wise converged to `envprev`.
"""
function gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}
    # Check if spaces in envprev and envfinal are the same
    same_spaces = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        space(envfinal.edges[dir, r, c]) == space(envprev.edges[dir, r, c]) &&
            space(envfinal.corners[dir, r, c]) == space(envprev.corners[dir, r, c])
    end
    @assert all(same_spaces) "Spaces of envprev and envfinal are not the same"

    # Try the "general" algorithm from https://arxiv.org/abs/2311.11894
    signs = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        # Gather edge tensors and pretend they're InfiniteMPSs
        if dir == NORTH
            Tsprev = circshift(envprev.edges[dir, r, :], 1 - c)
            Tsfinal = circshift(envfinal.edges[dir, r, :], 1 - c)
        elseif dir == EAST
            Tsprev = circshift(envprev.edges[dir, :, c], 1 - r)
            Tsfinal = circshift(envfinal.edges[dir, :, c], 1 - r)
        elseif dir == SOUTH
            Tsprev = circshift(reverse(envprev.edges[dir, r, :]), c)
            Tsfinal = circshift(reverse(envfinal.edges[dir, r, :]), c)
        elseif dir == WEST
            Tsprev = circshift(reverse(envprev.edges[dir, :, c]), r)
            Tsfinal = circshift(reverse(envfinal.edges[dir, :, c]), r)
        end

        # Random MPS of same bond dimension
        M = map(Tsfinal) do t
            TensorMap(randn, scalartype(t), codomain(t) ← domain(t))
        end

        # Find right fixed points of mixed transfer matrices
        ρinit = TensorMap(
            randn,
            scalartype(T),
            MPSKit._lastspace(Tsfinal[end])' ← MPSKit._lastspace(M[end])',
        )
        ρprev = transfermatrix_fixedpoint(Tsprev, M, ρinit)
        ρfinal = transfermatrix_fixedpoint(Tsfinal, M, ρinit)

        # Decompose and multiply
        Qprev, = leftorth(ρprev)
        Qfinal, = leftorth(ρfinal)

        return Qprev * Qfinal'
    end

    cornersfix, edgesfix = fix_relative_phases(envfinal, signs)

    # Fix global phase
    cornersgfix = map(envprev.corners, cornersfix) do Cprev, Cfix
        return dot(Cfix, Cprev) * Cfix
    end
    edgesgfix = map(envprev.edges, edgesfix) do Tprev, Tfix
        return dot(Tfix, Tprev) * Tfix
    end
    return CTMRGEnv(cornersgfix, edgesgfix)
end

# this is a bit of a hack to get the fixed point of the mixed transfer matrix
# because MPSKit is not compatible with AD
function transfermatrix_fixedpoint(tops, bottoms, ρinit)
    _, vecs, info = eigsolve(ρinit, 1, :LM, Arnoldi()) do ρ
        return foldr(zip(tops, bottoms); init=ρ) do (top, bottom), ρ
            return @tensor ρ′[-1; -2] := top[-1 4 3; 1] * conj(bottom[-2 4 3; 2]) * ρ[1; 2]
        end
    end
    info.converged > 0 || @warn "eigsolve did not converge"
    return first(vecs)
end

# Explicit fixing of relative phases (doing this compactly in a loop is annoying)
function _contract_gauge_corner(corner, σ_in, σ_out)
    @autoopt @tensor corner_fix[χ_in; χ_out] :=
        σ_in[χ_in; χ1] * corner[χ1; χ2] * conj(σ_out[χ_out; χ2])
end
function _contract_gauge_edge(edge, σ_in, σ_out)
    @autoopt @tensor edge_fix[χ_in D_above D_below; χ_out] :=
        σ_in[χ_in; χ1] * edge[χ1 D_above D_below; χ2] * conj(σ_out[χ_out; χ2])
end
function fix_relative_phases(envfinal::CTMRGEnv, signs)
    C1 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[NORTHWEST, r, c],
            signs[WEST, r, c],
            signs[NORTH, r, _next(c, end)],
        )
    end
    T1 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[NORTH, r, c],
            signs[NORTH, r, c],
            signs[NORTH, r, _next(c, end)],
        )
    end
    C2 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[NORTHEAST, r, c],
            signs[NORTH, r, c],
            signs[EAST, _next(r, end), c],
        )
    end
    T2 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[EAST, r, c], signs[EAST, r, c], signs[EAST, _next(r, end), c]
        )
    end
    C3 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[SOUTHEAST, r, c],
            signs[EAST, r, c],
            signs[SOUTH, r, _prev(c, end)],
        )
    end
    T3 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[SOUTH, r, c],
            signs[SOUTH, r, c],
            signs[SOUTH, r, _prev(c, end)],
        )
    end
    C4 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        _contract_gauge_corner(
            envfinal.corners[SOUTHWEST, r, c],
            signs[SOUTH, r, c],
            signs[WEST, _prev(r, end), c],
        )
    end
    T4 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        _contract_gauge_edge(
            envfinal.edges[WEST, r, c], signs[WEST, r, c], signs[WEST, _prev(r, end), c]
        )
    end

    return stack([C1, C2, C3, C4]; dims=1), stack([T1, T2, T3, T4]; dims=1)
end

function calc_convergence(envs, CSold, TSold)
    CSnew = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envs.corners)
    ΔCS = maximum(zip(CSold, CSnew)) do (c_old, c_new)
        # only compute the difference on the smallest part of the spaces
        smallest = infimum(MPSKit._firstspace(c_old), MPSKit._firstspace(c_new))
        e_old = isometry(MPSKit._firstspace(c_old), smallest)
        e_new = isometry(MPSKit._firstspace(c_new), smallest)
        return norm(e_new' * c_new * e_new - e_old' * c_old * e_old)
    end

    TSnew = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envs.edges)
    ΔTS = maximum(zip(TSold, TSnew)) do (t_old, t_new)
        MPSKit._firstspace(t_old) == MPSKit._firstspace(t_new) ||
            return scalartype(t_old)(Inf)
        return norm(t_new - t_old)
    end

    @debug "maxᵢ|Cⁿ⁺¹ - Cⁿ|ᵢ = $ΔCS   maxᵢ|Tⁿ⁺¹ - Tⁿ|ᵢ = $ΔTS"

    return max(ΔCS, ΔTS), CSnew, TSnew
end

@non_differentiable calc_convergence(args...)

"""
    calc_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below some tolerance.
"""
function calc_elementwise_convergence(envfinal::CTMRGEnv, envfix::CTMRGEnv; atol::Real=1e-6)
    ΔC = envfinal.corners .- envfix.corners
    ΔCmax = norm(ΔC, Inf)
    ΔCmean = norm(ΔC)
    @debug "maxᵢⱼ|Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmax   mean |Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmean"

    ΔT = envfinal.edges .- envfix.edges
    ΔTmax = norm(ΔT, Inf)
    ΔTmean = norm(ΔT)
    @debug "maxᵢⱼ|Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmax   mean |Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmean"

    # Check differences for all tensors in unit cell to debug properly
    for (dir, r, c) in Iterators.product(axes(envfinal.edges)...)
        @debug(
            "$((dir, r, c)): all |Cⁿ⁺¹ - Cⁿ|ᵢⱼ < ϵ: ",
            all(x -> abs(x) < atol, convert(Array, ΔC[dir, r, c])),
        )
        @debug(
            "$((dir, r, c)): all |Tⁿ⁺¹ - Tⁿ|ᵢⱼ < ϵ: ",
            all(x -> abs(x) < atol, convert(Array, ΔT[dir, r, c])),
        )
    end

    return max(ΔCmax, ΔTmax)
end

@non_differentiable calc_elementwise_convergence(args...)

"""
    ctmrg_iter(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}
    
Perform one iteration of CTMRG that maps the `state` and `env` to a new environment,
and also return the truncation error.
One CTMRG iteration consists of four `left_move` calls and 90 degree rotations,
such that the environment is grown and renormalized in all four directions.
"""
function ctmrg_iter(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}
    ϵ = 0.0

    for _ in 1:4
        env, info = left_move(state, env, alg.projector_alg)
        state = rotate_north(state, EAST)
        env = rotate_north(env, EAST)
        ϵ = max(ϵ, info.ϵ)
    end

    return env, (; ϵ)
end

"""
    left_move(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}

Grow, project and renormalize the environment `env` in west direction.
Return the updated environment as well as the projectors and truncation error.
"""
function left_move(state, env::CTMRGEnv{C,T}, alg::ProjectorAlg) where {C,T}
    corners::typeof(env.corners) = copy(env.corners)
    edges::typeof(env.edges) = copy(env.edges)
    ϵ = 0.0
    P_bottom, P_top = Zygote.Buffer.(projector_type(T, size(state)))  # Use Zygote.Buffer instead of @diffset to avoid ZeroTangent errors in _setindex

    for col in 1:size(state, 2)
        cprev = _prev(col, size(state, 2))

        # Compute projectors
        for row in 1:size(state, 1)
            # Enlarged corners
            Q_sw = southwest_corner((_next(row, size(state, 1)), col), env, state)
            Q_nw = northwest_corner((row, col), env, state)

            # SVD half-infinite environment
            trscheme = if alg.trscheme isa FixedSpaceTruncation
                truncspace(space(env.edges[WEST, row, col], 1))
            else
                alg.trscheme
            end
            @autoopt @tensor QQ[χ_EB D_EBabove D_EBbelow; χ_ET D_ETabove D_ETbelow] :=
                Q_sw[χ_EB D_EBabove D_EBbelow; χ D1 D2] *
                Q_nw[χ D1 D2; χ_ET D_ETabove D_ETbelow]
            U, S, V, ϵ_local = PEPSKit.tsvd!(QQ, alg.svd_alg; trunc=trscheme)
            ϵ = max(ϵ, ϵ_local / norm(S))
            # TODO: check if we can just normalize enlarged corners s.t. trunc behaves a bit better

            # Compute SVD truncation error and check for degenerate singular values
            ignore_derivatives() do
                if alg.verbosity > 0 && is_degenerate_spectrum(S)
                    svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
                    @warn("degenerate singular values detected: ", svals)
                end
            end

            # Compute projectors
            Pb, Pt = build_projectors(U, S, V, Q_sw, Q_nw)
            P_bottom[row, col] = Pb
            P_top[row, col] = Pt
        end

        # Use projectors to grow the corners & edges
        for row in 1:size(state, 1)
            rprev = _prev(row, size(state, 1))
            C_sw, C_nw, T_w = grow_env_left(
                state[row, col],
                P_bottom[rprev, col],
                P_top[row, col],
                env.corners[SOUTHWEST, row, cprev],
                env.corners[NORTHWEST, row, cprev],
                env.edges[SOUTH, row, col],
                env.edges[WEST, row, cprev],
                env.edges[NORTH, row, col],
            )
            @diffset corners[SOUTHWEST, row, col] = C_sw / norm(C_sw)
            @diffset corners[NORTHWEST, row, col] = C_nw / norm(C_nw)
            @diffset edges[WEST, row, col] = T_w / norm(T_w)
        end
    end

    return CTMRGEnv(corners, edges), (; P_left=copy(P_top), P_right=copy(P_bottom), ϵ)
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

@doc """
    LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)

Compute the norm of a PEPS contracted with a CTM environment.
"""

function LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)
    total = one(scalartype(peps))

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        rprev = _prev(r, size(peps, 1))
        rnext = _next(r, size(peps, 1))
        cprev = _prev(c, size(peps, 2))
        cnext = _next(c, size(peps, 2))
        total *= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.edges[NORTH, rprev, c][χ3 D3 D4; χ4] *
            env.corners[NORTHEAST, rprev, cnext][χ4; χ5] *
            env.edges[EAST, r, cnext][χ5 D5 D6; χ6] *
            env.corners[SOUTHEAST, rnext, cnext][χ6; χ7] *
            env.edges[SOUTH, rnext, c][χ7 D7 D8; χ8] *
            env.corners[SOUTHWEST, rnext, cprev][χ8; χ1] *
            peps[r, c][d; D3 D5 D7 D1] *
            conj(peps[r, c][d; D4 D6 D8 D2])
        total *= tr(
            env.corners[NORTHWEST, rprev, cprev] *
            env.corners[NORTHEAST, rprev, c] *
            env.corners[SOUTHEAST, r, c] *
            env.corners[SOUTHWEST, r, cprev],
        )
        total /= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.corners[NORTHEAST, rprev, c][χ3; χ4] *
            env.edges[EAST, r, c][χ4 D1 D2; χ5] *
            env.corners[SOUTHEAST, rnext, c][χ5; χ6] *
            env.corners[SOUTHWEST, rnext, cprev][χ6; χ1]
        total /= @autoopt @tensor env.corners[NORTHWEST, rprev, cprev][χ1; χ2] *
            env.edges[NORTH, rprev, c][χ2 D1 D2; χ3] *
            env.corners[NORTHEAST, rprev, cnext][χ3; χ4] *
            env.corners[SOUTHEAST, r, cnext][χ4; χ5] *
            env.edges[SOUTH, r, c][χ5 D1 D2; χ6] *
            env.corners[SOUTHWEST, r, cprev][χ6; χ1]
    end

    return total
end

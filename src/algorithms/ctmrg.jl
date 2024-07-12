"""
    FixedSpaceTruncation <: TensorKit.TruncationScheme

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

# TODO: add option for different projector styles (half-infinite, full-infinite, etc.)
"""
    struct ProjectorAlg{S}(; svd_alg = TensorKit.SVD(), trscheme = TensorKit.notrunc(),
                           fixedspace = false, verbosity = 0)

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
          svd_alg=TensorKit.SVD(), trscheme=FixedSpaceTruncation(),
          ctmrgscheme=:AllSides)

Algorithm struct that represents the CTMRG algorithm for contracting infinite PEPS.
Each CTMRG run is converged up to `tol` where the singular value convergence of the
corners as well as the norm is checked. The maximal and minimal number of CTMRG iterations
is set with `maxiter` and `miniter`. Different levels of output information are printed
depending on `verbosity` (0, 1 or 2).

The projectors are computed from `svd_alg` SVDs where the truncation scheme is set via
`trscheme`.

In general, two different schemes can be selected with `ctmrgscheme` which determine how
CTMRG is implemented. It can either be `:LeftMoves`, where the projectors are succesively
computed on the western side, and then applied and rotated. Or with `AllSides`, all projectors
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
    ctmrgscheme=:AllSides,
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
    return MPSKit.leading_boundary(CTMRGEnv(state), state, alg)
end
function MPSKit.leading_boundary(envinit, state, alg::CTMRG{S}) where {S}
    normold = 1.0
    CSold = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.corners)
    TSold = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.edges)
    ϵold = 1.0
    env = deepcopy(envinit)

    for i in 1:(alg.maxiter)
        env, info = ctmrg_iter(state, env, alg)  # Grow and renormalize in all 4 directions

        conv_condition, normold, CSold, TSold, ϵ = ignore_derivatives() do
            # Compute convergence criteria and take max (TODO: How should we handle logging all of this?)
            Δϵ = abs((ϵold - info.ϵ) / ϵold)
            normnew = norm(state, env)
            Δnorm = abs(normold - normnew) / abs(normold)
            CSnew = map(c -> tsvd(c; alg=TensorKit.SVD())[2], env.corners)
            ΔCS = maximum(zip(CSold, CSnew)) do (c_old, c_new)
                # only compute the difference on the smallest part of the spaces
                smallest = infimum(MPSKit._firstspace(c_old), MPSKit._firstspace(c_new))
                e_old = isometry(MPSKit._firstspace(c_old), smallest)
                e_new = isometry(MPSKit._firstspace(c_new), smallest)
                return norm(e_new' * c_new * e_new - e_old' * c_old * e_old)
            end
            TSnew = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env.edges)
            ΔTS = maximum(zip(TSold, TSnew)) do (t_old, t_new)
                MPSKit._firstspace(t_old) == MPSKit._firstspace(t_new) ||
                    return scalartype(t_old)(Inf)
                # TODO: implement when spaces aren't the same
                return norm(t_new - t_old)
            end

            conv_condition = max(Δnorm, ΔCS, ΔTS) < alg.tol && i > alg.miniter

            if alg.verbosity > 1 || (alg.verbosity == 1 && (i == 1 || conv_condition))
                @printf(
                    "CTMRG iter: %3d   norm: %.2e   Δnorm: %.2e   ΔCS: %.2e   ΔTS: %.2e   ϵ: %.2e   Δϵ: %.2e\n",
                    i,
                    abs(normnew),
                    Δnorm,
                    ΔCS,
                    ΔTS,
                    info.ϵ,
                    Δϵ
                )
            end
            alg.verbosity > 0 &&
                i == alg.maxiter &&
                @warn(
                    "CTMRG reached maximal number of iterations at (Δnorm=$Δnorm, ΔCS=$ΔCS, ΔTS=$ΔTS)"
                )
            flush(stdout)  # Flush output to enable live printing on HPC
            flush(stderr)  # Same for @info, @warn, ...
            return conv_condition, normnew, CSnew, TSnew, info.ϵ
        end
        conv_condition && break  # Converge if maximal Δ falls below tolerance
    end

    # Do one final iteration that does not change the spaces
    alg_fixed = CTMRG(;
        verbosity=alg.verbosity,
        svd_alg=alg.projector_alg.svd_alg,
        trscheme=FixedSpaceTruncation(),
        ctmrgscheme=S,
    )
    env′, = ctmrg_iter(state, env, alg_fixed)
    envfix, = gauge_fix(env, env′)
    check_elementwise_convergence(env, envfix; atol=alg.tol^(1 / 2)) ||
        @warn "CTMRG did not converge elementwise."
    return envfix
end

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
            rprev = _prev(row, size(state, 1))
            rnext = _next(row, size(state, 1))

            # Enlarged corners
            Q_sw = southwest_corner(
                env.edges[SOUTH, _next(rnext, end), col],
                env.corners[SOUTHWEST, _next(rnext, end), cprev],
                env.edges[WEST, rnext, cprev],
                state[rnext, col],
            )
            Q_nw = northwest_corner(
                env.edges[WEST, row, cprev],
                env.corners[NORTHWEST, rprev, cprev],
                env.edges[NORTH, rprev, col],
                state[row, col],
            )

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

# Compute enlarged corners
function northwest_corner(edge_W, corner_NW, edge_N, peps_above, peps_below=peps_above)
    @autoopt @tensor corner[χ_S D_Sabove D_Sbelow; χ_E D_Eabove D_Ebelow] :=
        edge_W[χ_S D1 D2; χ1] *
        corner_NW[χ1; χ2] *
        edge_N[χ2 D3 D4; χ_E] *
        peps_above[d; D3 D_Eabove D_Sabove D1] *
        conj(peps_below[d; D4 D_Ebelow D_Sbelow D2])
end
function northeast_corner(edge_N, corner_NE, edge_E, peps_above, peps_below=peps_above)
    @autoopt @tensor corner[χ_W D_Wabove D_Wbelow; χ_S D_Sabove D_Sbelow] :=
        edge_N[χ_W D1 D2; χ1] *
        corner_NE[χ1; χ2] *
        edge_E[χ2 D3 D4; χ_S] *
        peps_above[d; D1 D3 D_Sabove D_Wabove] *
        conj(peps_below[d; D2 D4 D_Sbelow D_Wbelow])
end
function southeast_corner(edge_E, corner_SE, edge_S, peps_above, peps_below=peps_above)
    @autoopt @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        edge_E[χ_N D1 D2; χ1] *
        corner_SE[χ1; χ2] *
        edge_S[χ2 D3 D4; χ_W] *
        peps_above[d; D_Nabove D1 D3 D_Wabove] *
        conj(peps_below[d; D_Nbelow D2 D4 D_Wbelow])
end
function southwest_corner(edge_S, corner_SW, edge_W, peps_above, peps_below=peps_above)
    @autoopt @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        edge_S[χ_E D1 D2; χ1] *
        corner_SW[χ1; χ2] *
        edge_W[χ2 D3 D4; χ_N] *
        peps_above[d; D_Nabove D_Eabove D1 D3] *
        conj(peps_below[d; D_Nbelow D_Ebelow D2 D4])
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

"""
    correlation_length(peps::InfinitePEPS, env::CTMRGEnv; howmany=2)

Compute the PEPS correlation length based on the horizontal and vertical
transfer matrices. Additionally the (normalized) eigenvalue spectrum is
returned. Specify the number of computed eigenvalues with `howmany`.
"""
function correlation_length(peps::InfinitePEPS, env::CTMRGEnv; howmany=2)
    ξ = Array{Float64,3}(undef, (2, size(peps)...))  # First index picks horizontal or vertical direction
    λ = Array{ComplexF64,4}(undef, (2, howmany, size(peps)...))
    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        @autoopt @tensor transferh[χ_LT D_Lab D_Lbe χ_LB; χ_RT D_Rab D_Rbe χ_RB] :=
            env.edges[NORTH, _prev(r, end), c][χ_LT D1 D2; χ_RT] *
            peps[r, c][d; D1 D_Rab D3 D_Lab] *
            conj(peps[r, c][d; D2 D_Rbe D4 D_Lbe]) *
            env.edges[SOUTH, _next(r, end), c][χ_RB D3 D4; χ_LB]
        @autoopt @tensor transferv[χ_TL D_Tab D_Tbe χ_TL; χ_BL D_Bab D_Bbe χ_BR] :=
            env.edges[EAST, r, _next(c, end)][χ_TR D1 D2; χ_BR] *
            peps[r, c][d; D_Tab D1 D_Bab D3] *
            conj(peps[r, c][d; D_Tbe D2 D_Bbe D4]) *
            env.edges[WEST, r, _prev(c, end)][χ_BL D3 D4; χ_TL]

        function lintransfer(v, t)
            @tensor v′[-1 -2 -3 -4] := t[-1 -2 -3 -4; 1 2 3 4] * v[1 2 3 4]
            return v′
        end

        v₀h = Tensor(randn, scalartype(transferh), domain(transferh))
        valsh, = eigsolve(v -> lintransfer(v, transferh), v₀h, howmany, :LM)
        λ[1, :, r, c] = valsh[1:howmany] / abs(valsh[1])  # Normalize largest eigenvalue to 1
        ξ[1, r, c] = -1 / log(abs(λ[1, 2, r, c]))

        v₀v = Tensor(rand, scalartype(transferv), domain(transferv))
        valsv, = eigsolve(v -> lintransfer(v, transferv), v₀v, howmany, :LM)
        λ[2, :, r, c] = valsv[1:howmany] / abs(valsv[1])  # Normalize largest eigenvalue to 1
        ξ[2, r, c] = -1 / log(abs(λ[2, 2, r, c]))
    end

    return ξ, λ
end

# TODO: add abstract Algorithm type?
"""
    struct CTMRG(; trscheme = TensorKit.notrunc(), tol = Defaults.ctmrg_tol,
                 maxiter = Defaults.ctmrg_maxiter, miniter = Defaults.ctmrg_miniter,
                 verbosity = 0, fixedspace = false)

Algorithm struct that represents the CTMRG algorithm for contracting infinite PEPS.
The projector bond dimensions are set via `trscheme` which controls the truncation
properties inside of `TensorKit.tsvd`. Each CTMRG run is converged up to `tol`
where the singular value convergence of the corners as well as the norm is checked.
The maximal and minimal number of CTMRG iterations is set with `maxiter` and `miniter`.
Different levels of output information are printed depending on `verbosity` (0, 1 or 2).
Regardless of the truncation scheme, the space can be kept fixed with `fixedspace`.
"""
@kwdef struct CTMRG
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.ctmrg_tol
    maxiter::Int = Defaults.ctmrg_maxiter
    miniter::Int = Defaults.ctmrg_miniter
    verbosity::Int = 0
    fixedspace::Bool = false
end

"""
    MPSKit.leading_boundary([envinit], state, alg::CTMRG)

Contract `state` using CTMRG and return the CTM environment.
Per default, a random initial environment is used.
"""
function MPSKit.leading_boundary(state, alg::CTMRG)
    return MPSKit.leading_boundary(CTMRGEnv(state), state, alg)
end
function MPSKit.leading_boundary(envinit, state, alg::CTMRG)
    normold = 1.0
    CSold = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.corners)
    TSold = map(x -> tsvd(x; alg=TensorKit.SVD())[2], envinit.edges)
    ϵold = 1.0
    env = deepcopy(envinit)

    for i in 1:(alg.maxiter)
        env, ϵ = ctmrg_iter(state, env, alg)  # Grow and renormalize in all 4 directions
        conv_condition, normold, CSold, TSold, ϵ = ignore_derivatives() do
            # Compute convergence criteria and take max (TODO: How should we handle logging all of this?)
            Δϵ = abs((ϵold - ϵ) / ϵold)
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
                    ϵ,
                    Δϵ
                )
            end
            alg.verbosity > 0 &&
                i == alg.maxiter &&
                @warn(
                    "CTMRG reached maximal number of iterations at (Δnorm=$Δnorm, ΔCS=$ΔCS, ΔTS=$ΔTS)"
                )
            return conv_condition, normnew, CSnew, TSnew, ϵ
        end
        conv_condition && break  # Converge if maximal Δ falls below tolerance
    end

    # Do one final iteration that does not change the spaces
    alg_fixed = CTMRG(;
        alg.trscheme, alg.tol, alg.maxiter, alg.miniter, alg.verbosity, fixedspace=true
    )
    env′, = ctmrg_iter(state, env, alg_fixed)
    envfix = gauge_fix(env, env′)
    check_elementwise_convergence(env, envfix; atol=alg.tol^(1 / 2)) ||
        @warn "CTMRG did not converge elementwise."
    return envfix
end

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
    @tensor corner_fix[χ_in; χ_out] := σ_in[χ_in; χ1] * corner[χ1; χ2] * conj(σ_out[χ_out; χ2])
end
function _contract_gauge_edge(edge, σ_in, σ_out)
    @tensor edge_fix[χ_in D_above D_below; χ_out] :=
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

"""
    check_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below some tolerance.
"""
function check_elementwise_convergence(
    envfinal::CTMRGEnv, envfix::CTMRGEnv; atol::Real=1e-6
)
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

    return isapprox(ΔCmax, 0; atol) && isapprox(ΔTmax, 0; atol)
end

@non_differentiable check_elementwise_convergence(args...)

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
        env, _, _, ϵ₀ = left_move(state, env, alg)
        state = rotate_north(state, EAST)
        env = rotate_north(env, EAST)
        ϵ = max(ϵ, ϵ₀)
    end

    return env, ϵ
end

"""
    left_move(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}

Grow, project and renormalize the environment `env` in west direction.
Return the updated environment as well as the projectors and truncation error.
"""
function left_move(state, env::CTMRGEnv{C,T}, alg::CTMRG) where {C,T}
    corners::typeof(env.corners) = copy(env.corners)
    edges::typeof(env.edges) = copy(env.edges)
    ϵ = 0.0
    Pleft, Pright = Zygote.Buffer.(projector_type(T, size(state)))  # Use Zygote.Buffer instead of @diffset to avoid ZeroTangent errors in _setindex

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
            trscheme = if alg.fixedspace == true
                truncspace(space(env.edges[WEST, row, col], 1))
            else
                alg.trscheme
            end
            @tensor QQ[χ_EB D_EBabove D_EBbelow; χ_ET D_ETabove D_ETbelow] :=
                Q_sw[χ_EB D_EBabove D_EBbelow; χ D1 D2] *
                Q_nw[χ D1 D2; χ_ET D_ETabove D_ETbelow]
            U, S, V, ϵ_local = tsvd!(QQ; trunc=trscheme, alg=TensorKit.SVD())
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
            Pl, Pr = build_projectors(U, S, V, Q_sw, Q_nw)
            Pleft[row, col] = Pl
            Pright[row, col] = Pr
        end

        # Use projectors to grow the corners & edges
        for row in 1:size(state, 1)
            rprev = _prev(row, size(state, 1))
            C_sw, C_nw, T_w = grow_env_left(
                state[row, col],
                Pleft[rprev, col],
                Pright[row, col],
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

    return CTMRGEnv(corners, edges), copy(Pleft), copy(Pright), ϵ
end

# Compute enlarged corners
function northwest_corner(edge_W, corner_NW, edge_N, peps_above, peps_below=peps_above)
    @tensor corner[χ_S D_Sabove D_Sbelow; χ_E D_Eabove D_Ebelow] :=
        edge_W[χ_S D1 D2; χ1] *
        corner_NW[χ1; χ2] *
        edge_N[χ2 D3 D4; χ_E] *
        peps_above[d; D3 D_Eabove D_Sabove D1] *
        conj(peps_below[d; D4 D_Ebelow D_Sbelow D2])
end
function northeast_corner(edge_N, corner_NE, edge_E, peps_above, peps_below=peps_above)
    @tensor corner[χ_W D_Wabove D_Wbelow; χ_S D_Sabove D_Sbelow] :=
        edge_N[χ_W D1 D2; χ1] *
        corner_NE[χ1; χ2] *
        edge_E[χ2 D3 D4; χ_S] *
        peps_above[d; D1 D3 D_Sabove D_Wabove] *
        conj(peps_below[d; D2 D4 D_Sbelow D_Wbelow])
end
function southeast_corner(edge_E, corner_SE, edge_S, peps_above, peps_below=peps_above)
    @tensor corner[χ_N D_Nabove D_Nbelow; χ_W D_Wabove D_Wbelow] :=
        edge_E[χ_N D1 D2; χ1] *
        corner_SE[χ1; χ2] *
        edge_S[χ2 D3 D4; χ_W] *
        peps_above[d; D_Nabove D1 D3 D_Wabove] *
        conj(peps_below[d; D_Nbelow D2 D4 D_Wbelow])
end
function southwest_corner(edge_S, corner_SW, edge_W, peps_above, peps_below=peps_above)
    @tensor corner[χ_E D_Eabove D_Ebelow; χ_N D_Nabove D_Nbelow] :=
        edge_S[χ_E D1 D2; χ1] *
        corner_SW[χ1; χ2] *
        edge_W[χ2 D3 D4; χ_N] *
        peps_above[d; D_Nabove D_Eabove D1 D3] *
        conj(peps_below[d; D_Nbelow D_Ebelow D2 D4])
end

# Build projectors from SVD and enlarged SW & NW corners
function build_projectors(
    U::AbstractTensorMap{E,3,1}, S, V::AbstractTensorMap{E,1,3}, Q_SW, Q_NW
) where {E<:ElementarySpace}
    isqS = sdiag_inv_sqrt(S)
    P_bottom = Q_NW * V' * isqS
    P_top = isqS * U' * Q_SW
    return P_bottom, P_top
end

# Apply projectors to entire left half-environment to grow SW & NW corners, and W edge
function grow_env_left(
    peps, P_bottom, P_top, corners_SW, corners_NW, edge_S, edge_W, edge_N
)
    @tensor corner_SW′[χ_E; χ_N] :=
        corners_SW[χ1; χ2] * edge_S[χ_E D1 D2; χ1] * P_bottom[χ2 D1 D2; χ_N]
    @tensor corner_NW′[χ_S; χ_E] :=
        corners_NW[χ1; χ2] * edge_N[χ2 D1 D2; χ_E] * P_top[χ_S; χ1 D1 D2]
    @tensor edge_W′[χ_S D_Eabove D_Ebelow; χ_N] :=
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
        total *= @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
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
        total /= @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.corners[NORTHEAST, rprev, c][χ3; χ4] *
            env.edges[EAST, r, c][χ4 D1 D2; χ5] *
            env.corners[SOUTHEAST, rnext, c][χ5; χ6] *
            env.corners[SOUTHWEST, rnext, cprev][χ6; χ1]
        total /= @tensor env.corners[NORTHWEST, rprev, cprev][χ1; χ2] *
            env.edges[NORTH, rprev, c][χ2 D1 D2; χ3] *
            env.corners[NORTHEAST, rprev, cnext][χ3; χ4] *
            env.corners[SOUTHEAST, r, cnext][χ4; χ5] *
            env.edges[SOUTH, r, c][χ5 D1 D2; χ6] *
            env.corners[SOUTHWEST, r, cprev][χ6; χ1]
    end

    return total
end
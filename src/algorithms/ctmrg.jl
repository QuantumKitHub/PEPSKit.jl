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
        ignore_derivatives() do # Print verbose info
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
        end
        conv_condition && break  # Converge if maximal Δ falls below tolerance

        # Update convergence criteria
        normold = normnew
        CSold = CSnew
        TSold = TSnew
        ϵold = ϵ
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
        ρprev = eigsolve(TransferMatrix(Tsprev, M), ρinit, 1, :LM)[2][1]
        ρfinal = eigsolve(TransferMatrix(Tsfinal, M), ρinit, 1, :LM)[2][1]

        # Decompose and multiply
        Up, _, Vp = tsvd(ρprev)
        Uf, _, Vf = tsvd(ρfinal)
        Qprev = Up * Vp
        Qfinal = Uf * Vf
        σ = Qprev * Qfinal'

        return σ
    end

    cornersfix, edgesfix = fix_relative_phases(envfinal, signs)

    # Fix global phase
    cornersgfix = map(zip(envprev.corners, cornersfix)) do (Cprev, Cfix)
        φ = dot(Cprev, Cfix)
        φ' * Cfix
    end
    edgesgfix = map(zip(envprev.edges, edgesfix)) do (Tprev, Tfix)
        φ = dot(Tprev, Tfix)
        φ' * Tfix
    end
    envfix = CTMRGEnv(cornersgfix, edgesgfix)

    return envfix
end

# Explicit fixing of relative phases (doing this compactly in a loop is annoying)
function fix_relative_phases(envfinal::CTMRGEnv, signs)
    C1 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            signs[WEST, _prev(r, end), c][-1 1] *
            envfinal.corners[NORTHWEST, r, c][1; 2] *
            conj(signs[NORTH, r, c][-2 2])
    end
    T1 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            signs[NORTH, r, c][-1 1] *
            envfinal.edges[NORTH, r, c][1 -2 -3; 2] *
            conj(signs[NORTH, r, _next(c, end)][-4 2])
    end

    C2 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            signs[NORTH, r, _next(c, end)][-1 1] *
            envfinal.corners[NORTHEAST, r, c][1; 2] *
            conj(signs[EAST, r, c][-2 2])
    end
    T2 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            signs[EAST, r, c][-1 1] *
            envfinal.edges[EAST, r, c][1 -2 -3; 2] *
            conj(signs[EAST, _next(r, end), c][-4 2])
    end

    C3 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            signs[EAST, _next(r, end), c][-1 1] *
            envfinal.corners[SOUTHEAST, r, c][1; 2] *
            conj(signs[SOUTH, r, c][-2 2])
    end
    T3 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            signs[SOUTH, r, c][-1 1] *
            envfinal.edges[SOUTH, r, c][1 -2 -3; 2] *
            conj(signs[SOUTH, r, _prev(c, end)][-4 2])
    end

    C4 = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            signs[SOUTH, r, _prev(c, end)][-1 1] *
            envfinal.corners[SOUTHWEST, r, c][1; 2] *
            conj(signs[WEST, r, c][-2 2])
    end
    T4 = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            signs[WEST, r, c][-1 1] *
            envfinal.edges[WEST, r, c][1 -2 -3; 2] *
            conj(signs[WEST, _prev(r, end), c][-4 2])
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
        cnext = _next(col, size(state, 2))

        # Compute projectors
        for row in 1:size(state, 1)
            rnext = _next(row, size(state, 1))
            state_nw = state[row, col]
            state_sw = rotate_north(state[rnext, col], WEST)

            # Enlarged corners
            Q_sw = northwest_corner(
                env.edges[SOUTH, _next(row, end), col],
                env.corners[SOUTHWEST, _next(row, end), col],
                env.edges[WEST, _next(row, end), col],
                state_sw,
            )
            Q_nw = northwest_corner(
                env.edges[WEST, row, col],
                env.corners[NORTHWEST, row, col],
                env.edges[NORTH, row, col],
                state_nw,
            )

            # SVD half-infinite environment
            trscheme = if alg.fixedspace == true
                truncspace(space(env.edges[WEST, row, cnext], 1))
            else
                alg.trscheme
            end
            @tensor QQ[-1 -2 -3;-4 -5 -6] := Q_sw[-1 -2 -3;1 2 3] * Q_nw[1 2 3;-4 -5 -6]
            U, S, V, ϵ_local = tsvd!(QQ; trunc=trscheme, alg=TensorKit.SVD())
            #U, S, V, ϵ_local = tsvd!(Q_sw * Q_nw; trunc=trscheme, alg=TensorKit.SVD())  # TODO: Add field in CTMRG to choose SVD function
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
            rnext = _next(row, size(state, 1))
            C_sw, C_nw, T_w = grow_env_left(
                state[row, col],
                Pleft[_prev(row, end), col],
                Pright[row, col],
                env.corners[SOUTHWEST, rprev, col],
                env.corners[NORTHWEST, rnext, col],
                env.edges[SOUTH, rprev, col],
                env.edges[WEST, row, col],
                env.edges[NORTH, rnext, col],
            )
            @diffset corners[SOUTHWEST, rprev, cnext] = C_sw
            @diffset corners[NORTHWEST, rnext, cnext] = C_nw
            @diffset edges[WEST, row, cnext] = T_w
        end

        @diffset corners[SOUTHWEST, :, cnext] ./= norm.(corners[SOUTHWEST, :, cnext])
        @diffset corners[NORTHWEST, :, cnext] ./= norm.(corners[NORTHWEST, :, cnext])
        @diffset edges[WEST, :, cnext] ./= norm.(edges[WEST, :, cnext])
    end

    return CTMRGEnv(corners, edges), copy(Pleft), copy(Pright), ϵ
end

# Compute enlarged NW corner
function northwest_corner(E4, C1, E1, peps_above, peps_below=peps_above)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E4[-1 1 2; 3] *
        C1[3; 4] *
        E1[4 5 6; -4] *
        peps_above[7; 5 -5 -2 1] *
        conj(peps_below[7; 6 -6 -3 2])
end

# Compute enlarged NE corner
function northeast_corner(E1, C2, E2, peps_above, peps_below=peps_above)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E1[-1 1 2; 3] *
        C2[3; 4] *
        E2[4 5 6; -4] *
        peps_above[7; 1 5 -5 -2] *
        conj(peps_below[7; 2 6 -6 -3])
end

# Compute enlarged SE corner
function southeast_corner(E2, C3, E3, peps_above, peps_below=peps_above)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E2[-1 1 2; 3] *
        C3[3; 4] *
        E3[4 5 6; -4] *
        peps_above[7; -2 1 5 -5] *
        conj(peps_below[7; -3 2 6 -6])
end

# Build projectors from SVD and enlarged SW & NW corners
function build_projectors(
    U::AbstractTensorMap{E,3,1}, S, V::AbstractTensorMap{E,1,3}, Q_sw, Q_nw
) where {E<:ElementarySpace}
    isqS = sdiag_inv_sqrt(S)
    Pl = Q_nw*V'*isqS
    Pr = isqS*U'*Q_sw
    #@tensor Pl[-1 -2 -3; -4] := Q_nw[-1 -2 -3; 1 2 3] * conj(V[4; 1 2 3]) * isqS[4; -4]
    #@tensor Pr[-1; -2 -3 -4] := isqS[-1; 1] * conj(U[2 3 4; 1]) * Q_sw[2 3 4; -2 -3 -4]
    return Pl, Pr
end

# Apply projectors to entire left half-environment to grow SW & NW corners, and W edge
function grow_env_left(peps, Pl, Pr, C_sw, C_nw, T_s, T_w, T_n)
    @tensor C_sw′[-1; -2] := C_sw[1; 4] * T_s[-1 2 3; 1] * Pl[4 2 3; -2]
    @tensor C_nw′[-1; -2] := C_nw[1; 2] * T_n[2 3 4; -2] * Pr[-1; 1 3 4]
    @tensor T_w′[-1 -2 -3; -4] :=
        T_w[1 2 3; 4] *
        peps[9; 5 -2 7 2] *
        conj(peps[9; 6 -3 8 3]) *
        Pl[4 5 6; -4] *
        Pr[-1; 1 7 8]
    return C_sw′, C_nw′, T_w′
end

@doc """
    LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)

Compute the norm of a PEPS contracted with a CTM environment.
"""

function LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)
    total = one(scalartype(peps))

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        total *= @tensor env.edges[WEST, r, c][1 2 3; 4] *
            env.corners[NORTHWEST, r, c][4; 5] *
            env.edges[NORTH, r, c][5 6 7; 8] *
            env.corners[NORTHEAST, r, c][8; 9] *
            env.edges[EAST, r, c][9 10 11; 12] *
            env.corners[SOUTHEAST, r, c][12; 13] *
            env.edges[SOUTH, r, c][13 14 15; 16] *
            env.corners[SOUTHWEST, r, c][16; 1] *
            peps[r, c][17; 6 10 14 2] *
            conj(peps[r, c][17; 7 11 15 3])

        total *=  @tensor env.corners[NORTHWEST, r, c][1; 2] *
            env.corners[NORTHEAST, r, mod1(c - 1, end)][2; 3] *
            env.corners[SOUTHEAST, mod1(r - 1, end), mod1(c - 1, end)][3; 4] *
            env.corners[SOUTHWEST, mod1(r - 1, end), c][4; 1]

        total /= @tensor env.edges[WEST, r, c][1 10 11; 4] *
            env.corners[NORTHWEST, r, c][4; 5] *
            env.corners[NORTHEAST, r, mod1(c - 1, end)][5; 6] *
            env.edges[EAST, r, mod1(c - 1, end)][6 10 11; 7] *
            env.corners[SOUTHEAST, r, mod1(c - 1, end)][7; 8] *
            env.corners[SOUTHWEST, r, c][8; 1]

        total /= @tensor env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 10 11; 3] *
            env.corners[NORTHEAST, r, c][3; 4] *
            env.corners[SOUTHEAST, mod1(r - 1, end), c][4; 5] *
            env.edges[SOUTH, mod1(r - 1, end), c][5 10 11; 6] *
            env.corners[SOUTHWEST, mod1(r - 1, end), c][6; 1]
    end

    return total
end

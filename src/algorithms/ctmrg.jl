# TODO: add abstract Algorithm type?
@kwdef struct CTMRG
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.ctmrg_tol
    maxiter::Int = Defaults.ctmrg_maxiter
    miniter::Int = Defaults.ctmrg_miniter
    verbosity::Int = 0
    fixedspace::Bool = false
end

# Compute CTMRG environment for a given state
# function MPSKit.leading_boundary(state, alg::CTMRG, envinit=CTMRGEnv(state))
function MPSKit.leading_boundary(state, alg::CTMRG, envinit=CTMRGEnv(state))
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
        ΔCS = maximum(norm.(CSnew - CSold))
        TSnew = map(t -> tsvd(t; alg=TensorKit.SVD())[2], env.edges)
        ΔTS = maximum(norm.(TSnew - TSold))
        (max(Δnorm, ΔCS, ΔTS) < alg.tol && i > alg.miniter) && break  # Converge if maximal Δ falls below tolerance

        # Print verbose info
        ignore_derivatives() do
            alg.verbosity > 1 && @printf(
                "CTMRG iter: %3d   norm: %.2e   Δnorm: %.2e   ΔCS: %.2e   ΔTS: %.2e   ϵ: %.2e   Δϵ: %.2e\n",
                i,
                abs(normnew),
                Δnorm,
                ΔCS,
                ΔTS,
                ϵ,
                Δϵ
            )
            alg.verbosity > 0 &&
                i == alg.maxiter &&
                @warn(
                    "CTMRG reached maximal number of iterations at (Δnorm=$Δnorm, ΔCS=$ΔCS, ΔTS=$ΔTS)"
                )
        end

        # Update convergence criteria
        normold = normnew
        CSold = CSnew
        TSold = TSnew
        ϵold = ϵ
    end

    env′, = ctmrg_iter(state, env, alg)
    envfix = gauge_fix(env, env′)
    check_elementwise_convergence(env, envfix) ||
        @warn "CTMRG did not converge elementwise."
    return envfix
end

# Fix gauge of corner end edge tensors from last and second last CTMRG iteration
function gauge_fix(envprev::CTMRGEnv{C,T}, envfinal::CTMRGEnv{C,T}) where {C,T}
    # Compute gauge tensors by comparing signs
    # First fix physical indices to (1, 1)
    Tfixprev = map(x -> convert(Array, x)[:, 1, 1, :], envprev.edges)
    Tfixfinal = map(x -> convert(Array, x)[:, 1, 1, :], envfinal.edges)
    signs = map(Iterators.product(axes(envfinal.edges)...)) do (dir, r, c)
        if isodd(dir)
            seqprev = prod(circshift(Tfixprev[dir, r, :], 1 - c))
            seqfinal = prod(circshift(Tfixfinal[dir, r, :], 1 - c))
        else
            seqprev = prod(circshift(Tfixprev[dir, :, c], 1 - r))
            seqfinal = prod(circshift(Tfixfinal[dir, :, c], 1 - r))
        end

        φ = sum(diag(seqfinal) ./ diag(seqprev)) / size(seqprev, 1)  # Global sequence phase
        σ = sign.(seqfinal[1, :] ./ seqprev[1, :]) * φ'
        Tensor(diagm(σ), space(envprev.edges[1], 1) * space(envprev.edges[1], 1)')
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

# Explicit unrolling of for loop from previous version to fix AD
# TODO: Does not yet properly work for Lx,Ly > 2
function fix_relative_phases(envfinal::CTMRGEnv, signs)
    e1 = envfinal
    σ1 = signs
    C1 = map(Iterators.product(axes(e1.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            σ1[WEST, _prev(r, end), c][-1 1] *
            e1.corners[NORTHWEST, r, c][1; 2] *
            conj(σ1[NORTH, r, c][-2 2])
    end
    T1 = map(Iterators.product(axes(e1.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            σ1[NORTH, r, c][-1 1] *
            e1.edges[NORTH, r, c][1 -2 -3; 2] *
            conj(σ1[NORTH, r, _next(c, end)][-4 2])
    end

    e2 = rotate_north(envfinal, EAST)
    σ2 = rotate_north(signs, EAST)
    C2 = map(Iterators.product(axes(e2.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            σ2[WEST, _prev(r, end), c][-1 1] *
            e2.corners[NORTHWEST, r, c][1; 2] *
            conj(σ2[NORTH, r, c][-2 2])
    end
    C2 = rotate_north(C2, WEST)
    T2 = map(Iterators.product(axes(e2.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            σ2[NORTH, r, c][-1 1] *
            e2.edges[NORTH, r, c][1 -2 -3; 2] *
            conj(σ2[NORTH, r, _next(c, end)][-4 2])
    end
    T2 = rotate_north(T2, WEST)

    e3 = rotate_north(envfinal, SOUTH)
    σ3 = rotate_north(signs, SOUTH)
    C3 = map(Iterators.product(axes(e3.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            σ3[WEST, _prev(r, end), c][-1 1] *
            e3.corners[NORTHWEST, r, c][1; 2] *
            conj(σ3[NORTH, r, c][-2 2])
    end
    C3 = rotate_north(C3, SOUTH)
    T3 = map(Iterators.product(axes(e3.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            σ3[NORTH, r, c][-1 1] *
            e3.edges[NORTH, r, c][1 -2 -3; 2] *
            conj(σ3[NORTH, r, _next(c, end)][-4 2])
    end
    T3 = rotate_north(T3, SOUTH)

    e4 = rotate_north(envfinal, WEST)
    σ4 = rotate_north(signs, WEST)
    C4 = map(Iterators.product(axes(e4.corners)[2:3]...)) do (r, c)
        @tensor Cfix[-1; -2] :=
            σ4[WEST, _prev(r, end), c][-1 1] *
            e4.corners[NORTHWEST, r, c][1; 2] *
            conj(σ4[NORTH, r, c][-2 2])
    end
    C4 = rotate_north(C4, EAST)
    T4 = map(Iterators.product(axes(e4.edges)[2:3]...)) do (r, c)
        @tensor Tfix[-1 -2 -3; -4] :=
            σ4[NORTH, r, c][-1 1] *
            e4.edges[NORTH, r, c][1 -2 -3; 2] *
            conj(σ4[NORTH, r, _next(c, end)][-4 2])
    end
    T4 = rotate_north(T4, EAST)

    return stack([C1, C2, C3, C4]; dims=1), stack([T1, T2, T3, T4]; dims=1)
end

# Semi-working version analogous to left_move with rotations
# function fix_relative_phases(envfinal::CTMRGEnv, signs)
#     cornersfix = deepcopy(envfinal.corners)
#     edgesfix = deepcopy(envfinal.edges)
#     for _ in 1:4
#         corners = map(Iterators.product(axes(envfinal.corners)[2:3]...)) do (r, c)
#             @tensor Cfix[-1; -2] :=
#                 signs[WEST, _prev(r, end), c][-1 1] *
#                 envfinal.corners[NORTHWEST, r, c][1; 2] *
#                 conj(signs[NORTH, r, c][-2 2])
#         end
#         @diffset cornersfix[NORTHWEST, :, :] .= corners
#         edges = map(Iterators.product(axes(envfinal.edges)[2:3]...)) do (r, c)
#             @tensor Tfix[-1 -2 -3; -4] :=
#                 signs[NORTH, r, c][-1 1] *
#                 envfinal.edges[NORTH, r, c][1 -2 -3; 2] *
#                 conj(signs[NORTH, r, _next(c, end)][-4 2])
#         end
#         @diffset edgesfix[NORTH, :, :] .= edges

#         # Rotate east-wards
#         envfinal = rotate_north(envfinal, EAST)
#         cornersfix = rotate_north(cornersfix, EAST)
#         edgesfix = rotate_north(edgesfix, EAST)
#         signs = rotate_north(signs, EAST)  # TODO: Fix AD problem here
#     end
#     return cornersfix, edgesfix
# end

"""
    check_elementwise_convergence(envfinal, envfix; atol=1e-6)

Check if the element-wise difference of the corner and edge tensors of the final and fixed
CTMRG environments are below some tolerance.
"""
function check_elementwise_convergence(envfinal::CTMRGEnv, envfix::CTMRGEnv; atol::Real=1e-6)
    # TODO: do we need both max and mean?
    ΔC = envfinal.corners .- envfix.corners
    ΔCmax = norm(ΔC, Inf)
    ΔCmean = norm(ΔC)
    @debug "maxᵢⱼ|Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmax   mean |Cⁿ⁺¹ - Cⁿ|ᵢⱼ = $ΔCmean"
    
    ΔT = envfinal.edges .- envfix.edges
    ΔTmax = norm(ΔT, Inf)
    ΔTmean = norm(ΔT)
    @debug "maxᵢⱼ|Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmax   mean |Tⁿ⁺¹ - Tⁿ|ᵢⱼ = $ΔTmean"
    
    return isapprox(ΔCmax, 0; atol) && isapprox(ΔTmax, 0; atol)
end

@non_differentiable check_elementwise_convergence(args...)


# One CTMRG iteration x′ = f(A, x)
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

# Grow environment, compute projectors and renormalize
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
            (U, S, V) = tsvd(Q_sw * Q_nw; trunc=trscheme, alg=TensorKit.SVD())  # TODO: Add field in CTMRG to choose SVD function

            # Compute SVD truncation error and check for degenerate singular values
            ignore_derivatives() do
                if alg.verbosity > 0 && is_degenerate_spectrum(S)
                    @warn("degenerate singular values detected: ", diag(S.data))
                end
                n0 = norm(Q_sw * Q_nw)^2
                n1 = norm(U * S * V)^2
                ϵ = max(ϵ, (n0 - n1) / n0)
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

function northeast_corner(E1, C2, E2, peps_above, peps_below=peps_above)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E1[-1 1 2; 3] *
        C2[3; 4] *
        E2[4 5 6; -4] *
        peps_above[7; 1 5 -5 -2] *
        conj(peps_below[7; 2 6 -6 -3])
end

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
    @tensor Pl[-1 -2 -3; -4] := Q_nw[-1 -2 -3; 1 2 3] * conj(V[4; 1 2 3]) * isqS[4; -4]
    @tensor Pr[-1; -2 -3 -4] := isqS[-1; 1] * conj(U[2 3 4; 1]) * Q_sw[2 3 4; -2 -3 -4]
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

# Compute norm of the entire CMTRG enviroment
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

        total *= tr(
            env.corners[NORTHWEST, r, c] *
            env.corners[NORTHEAST, r, mod1(c - 1, end)] *
            env.corners[SOUTHEAST, mod1(r - 1, end), mod1(c - 1, end)] *
            env.corners[SOUTHWEST, mod1(r - 1, end), c],
        )

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

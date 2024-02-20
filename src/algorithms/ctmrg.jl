@kwdef struct CTMRG #<: Algorithm
    trscheme::TruncationScheme = TensorKit.notrunc()
    tol::Float64 = Defaults.ctmrg_tol
    maxiter::Int = Defaults.ctmrg_maxiter
    miniter::Int = Defaults.ctmrg_miniter
    verbose::Int = 0
    fixedspace::Bool = false
end

# Compute CTMRG environment for a given state
function MPSKit.leading_boundary(state, alg::CTMRG, envinit=CTMRGEnv(state))
    normold = 1.0
    CSold = tsvd(envinit.corners[NORTHWEST]; alg=TensorKit.SVD())[2]
    TSold = tsvd(envinit.edges[NORTH]; alg=TensorKit.SVD())[2]
    ϵold = 1.0

    env = deepcopy(envinit)
    for i in 1:(alg.maxiter)
        env, iterinfo = ctmrg_iter(state, env, alg)  # Grow and renormalize in all 4 directions

        # Compute convergence criteria and take max (TODO: How should we handle logging all of this?)
        Δϵ = abs((ϵold - iterinfo.ϵ) / ϵold)
        normnew = contract_ctmrg(state, env)
        Δnorm = abs(normold - normnew)
        CSnew = tsvd(env.corners[NORTHWEST]; alg=TensorKit.SVD())[2]
        ΔCS = norm(CSnew - CSold)
        TSnew = tsvd(env.edges[NORTH]; alg=TensorKit.SVD())[2]
        ΔTS = norm(TSnew - TSold)
        (max(Δnorm, ΔCS, ΔTS) < alg.tol && i > alg.miniter) && break  # Converge if maximal Δ falls below tolerance

        # Print verbose info
        ignore_derivatives() do
            alg.verbose > 1 && @printf(
                "CTMRG iter: %3d   norm: %.2e   Δnorm: %.2e   ΔCS: %.2e   ΔTS: %.2e   ϵ: %.2e   Δϵ: %.2e\n",
                i,
                abs(normnew),
                Δnorm,
                ΔCS,
                ΔTS,
                iterinfo.ϵ,
                Δϵ
            )
            alg.verbose > 0 &&
                i == alg.maxiter &&
                @warn(
                    "CTMRG reached maximal number of iterations at (Δnorm=$Δnorm, ΔCS=$ΔCS, ΔTS=$ΔTS)"
                )
        end

        # Update convergence criteria
        normold = normnew
        CSold = CSnew
        TSold = TSnew
        ϵold = iterinfo.ϵ
    end

    return env
end

# One CTMRG iteration x′ = f(A, x)
function ctmrg_iter(state, env::CTMRGEnv, alg::CTMRG)
    ϵ = 0.0
    Pleft = Vector{Matrix{typeof(env.edges[1])}}(undef, 4)
    Pright = Vector{Matrix{typeof(transpose(env.edges[1]))}}(undef, 4)

    for i in 1:4
        env, info = left_move(state, env, alg)
        state = rotate_north(state, EAST)
        env = rotate_north(env, EAST)
        ϵ = max(ϵ, info.ϵ)
        @diffset Pleft[i] = info.Pleft
        @diffset Pright[i] = info.Pright
    end

    iterinfo = (; ϵ, Pleft=copy(Pleft), Pright=copy(Pright))
    return env, iterinfo
end

# Grow environment, compute projectors and renormalize
function left_move(state, env::CTMRGEnv, alg::CTMRG)
    corners::typeof(env.corners) = copy(env.corners)
    edges::typeof(env.edges) = copy(env.edges)
    ϵ = 0.0

    Pleft = similar(state.A, typeof(env.edges[1]))
    Pright = similar(state.A, typeof(transpose(env.edges[1])))
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
                if unique(x -> round(x; digits=14), diag(S.data)) != diag(S.data) &&
                    alg.verbose > 0
                    println("degenerate singular values detected", diag(S.data))
                end
                n0 = norm(Q_sw * Q_nw)^2
                n1 = norm(U * S * V)^2
                ϵ = max(ϵ, (n0 - n1) / n0)
            end

            # Compute projectors
            Pl, Pr = build_projectors(U, S, V, Q_sw, Q_nw)
            @diffset Pleft[row, col] = Pl
            @diffset Pright[row, col] = Pr
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

    return CTMRGEnv(corners, edges), (; ϵ, Pleft, Pright)
end

# Compute enlarged NW corner
function northwest_corner(E4, C1, E1, peps)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E4[-1 1 2; 3] *
        C1[3; 4] *
        E1[4 5 6; -4] *
        peps[7; 5 -5 -2 1] *
        conj(peps[7; 6 -6 -3 2])
end

function northeast_corner(E1, C2, E2, peps)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E1[-1 1 2; 3] *
        C2[3; 4] *
        E2[4 5 6; -4] *
        peps[7; 1 5 -5 -2] *
        conj(peps[7; 2 6 -6 -3])
end

function southeast_corner(E2, C3, E3, peps)
    @tensor corner[-1 -2 -3; -4 -5 -6] :=
        E2[-1 1 2; 3] *
        C3[3; 4] *
        E3[4 5 6; -4] *
        peps[7; -2 1 5 -5] *
        conj(peps[7; -3 2 6 -6])
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
    # @diffset @tensor corners[NORTHWEST, rop, cop][-1; -2] :=
    #     envs.corners[NORTHWEST, rop, col][1, 2] *
    #     envs.edges[NORTH, rop, col][2, 3, 4, -2] *
    #     Q[-1; 1 3 4]
    # @diffset @tensor corners[SOUTHWEST, rom, cop][-1; -2] :=
    #     envs.corners[SOUTHWEST, rom, col][1, 4] *
    #     envs.edges[SOUTH, rom, col][-1, 2, 3, 1] *
    #     P[4 2 3; -2]
    # @diffset @tensor edges[WEST, row, cop][-1 -2 -3; -4] :=
    #     envs.edges[WEST, row, col][1 2 3; 4] *
    #     peps_above[row, col][9; 5 -2 7 2] *
    #     conj(peps_below[row, col][9; 6 -3 8 3]) *
    #     P[4 5 6; -4] *
    #     Q[-1; 1 7 8]
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
function contract_ctmrg(peps, env::CTMRGEnv)
    total = 1.0 + 0im

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

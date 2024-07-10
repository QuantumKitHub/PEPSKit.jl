# One CTMRG iteration with both-sided application of projectors
function ctmrg_iter(state, env::CTMRGEnv, alg::CTMRG{:AllSides})
    # Compute enlarged corners
    Q = enlarge_corners_edges(state, env)

    # Compute projectors if none are supplied
    P_left, P_right, info = build_projectors(Q, env, alg.projector_alg)

    # Apply projectors and normalize
    corners, edges = renormalize_corners_edges(state, env, Q, P_left, P_right)

    return CTMRGEnv(corners, edges), (; P_left, P_right, info...)
end

# Compute enlarged corners and edges for all directions and unit cell entries
function enlarge_corners_edges(state, env::CTMRGEnv)
    map(Iterators.product(axes(env.corners)...)) do (dir, r, c)
        rprev = _prev(r, size(state, 1))
        rnext = _next(r, size(state, 1))
        cprev = _prev(c, size(state, 2))
        cnext = _next(c, size(state, 2))
        if dir == NORTHWEST
            return northwest_corner(
                env.edges[WEST, r, cprev],
                env.corners[NORTHWEST, rprev, cprev],
                env.edges[NORTH, rprev, c],
                state[r, c],
            )
        elseif dir == NORTHEAST
            return northeast_corner(
                env.edges[NORTH, rprev, c],
                env.corners[NORTHEAST, rprev, cnext],
                env.edges[EAST, r, cnext],
                state[r, c],
            )
        elseif dir == SOUTHEAST
            return southeast_corner(
                env.edges[EAST, r, cnext],
                env.corners[SOUTHEAST, rnext, cnext],
                env.edges[SOUTH, rnext, c],
                state[r, c],
            )
        elseif dir == SOUTHWEST
            return southwest_corner(
                env.edges[SOUTH, rnext, c],
                env.corners[SOUTHWEST, rnext, cprev],
                env.edges[WEST, r, cprev],
                state[r, c],
            )
        end
    end
end

# Build projectors from SVD and enlarged corners
function build_projectors(Q, env::CTMRGEnv, alg::ProjectorAlg{A,T}) where {A,T}  # TODO: Add projector type annotations
    P_left, P_right = Zygote.Buffer.(projector_type(env.edges))
    U, V = Zygote.Buffer.(projector_type(env.edges))
    S = Zygote.Buffer(env.corners)
    ϵ = 0.0
    rsize, csize = size(env.corners)[2:3]
    for dir in 1:4, r in 1:rsize, c in 1:csize
        # Row-column index of next enlarged corner
        next_rc = if dir == 1
            (r, _next(c, csize))
        elseif dir == 2
            (_next(r, rsize), c)
        elseif dir == 3
            (r, _prev(c, csize))
        elseif dir == 4
            (_prev(r, rsize), c)
        end

        # SVD half-infinite environment
        trscheme = if T <: FixedSpaceTruncation
            truncspace(space(env.edges[dir, r, c], 1))
        else
            alg.trscheme
        end
        svd_alg = if A <: SVDAdjoint{<:FixedSVD}
            idx = (dir, r, c)
            fwd_alg = alg.svd_alg.fwd_alg
            fix_svd = FixedSVD(fwd_alg.U[idx...], fwd_alg.S[idx...], fwd_alg.V[idx...])
            return SVDAdjoint(; fwd_alg=fix_svd, rrule_alg=nothing, broadening=nothing)
        else
            alg.svd_alg
        end
        @autoopt @tensor QQ[χ_EB D_EBabove D_EBbelow; χ_ET D_ETabove D_ETbelow] :=
            Q[dir, r, c][χ_EB D_EBabove D_EBbelow; χ D1 D2] *
            Q[_next(dir, 4), next_rc...][χ D1 D2; χ_ET D_ETabove D_ETbelow]
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
        Pl, Pr = build_projectors(
            U_local, S_local, V_local, Q[dir, r, c], Q[_next(dir, 4), next_rc...]
        )
        P_left[dir, r, c] = Pl
        P_right[dir, r, c] = Pr
    end

    return copy(P_left), copy(P_right), (; ϵ, U=copy(U), S=copy(S), V=copy(V))
end

# Apply projectors to renormalize corners and edges
function renormalize_corners_edges(state, env::CTMRGEnv, Q, P_left, P_right)
    corners::typeof(env.corners) = copy(env.corners)
    edges::typeof(env.edges) = copy(env.edges)
    for c in 1:size(state, 2), r in 1:size(state, 1)
        rprev = _prev(r, size(state, 1))
        rnext = _next(r, size(state, 1))
        cprev = _prev(c, size(state, 2))
        cnext = _next(c, size(state, 2))
        @diffset @tensor corners[NORTHWEST, r, c][-1; -2] :=
            P_right[WEST, rnext, c][-1; 1 2 3] *
            Q[NORTHWEST, r, c][1 2 3; 4 5 6] *
            P_left[NORTH, r, c][4 5 6; -2]
        @diffset @tensor corners[NORTHEAST, r, c][-1; -2] :=
            P_right[NORTH, r, cprev][-1; 1 2 3] *
            Q[NORTHEAST, r, c][1 2 3; 4 5 6] *
            P_left[EAST, r, c][4 5 6; -2]
        @diffset @tensor corners[SOUTHEAST, r, c][-1; -2] :=
            P_right[EAST, rprev, c][-1; 1 2 3] *
            Q[SOUTHEAST, r, c][1 2 3; 4 5 6] *
            P_left[SOUTH, r, c][4 5 6; -2]
        @diffset @tensor corners[SOUTHWEST, r, c][-1; -2] :=
            P_right[SOUTH, r, cnext][-1; 1 2 3] *
            Q[SOUTHWEST, r, c][1 2 3; 4 5 6] *
            P_left[WEST, r, c][4 5 6; -2]

        @diffset @tensor edges[NORTH, r, c][-1 -2 -3; -4] :=
            env.edges[NORTH, rprev, c][1 2 3; 4] *
            state[r, c][9; 2 5 -2 7] *
            conj(state[r, c][9; 3 6 -3 8]) *
            P_left[NORTH, r, c][4 5 6; -4] *
            P_right[NORTH, r, cprev][-1; 1 7 8]
        @diffset @tensor edges[EAST, r, c][-1 -2 -3; -4] :=
            env.edges[EAST, r, _next(c, end)][1 2 3; 4] *
            state[r, c][9; 7 2 5 -2] *
            conj(state[r, c][9; 8 3 6 -3]) *
            P_left[EAST, r, c][4 5 6; -4] *
            P_right[EAST, rprev, c][-1; 1 7 8]
        @diffset @tensor edges[SOUTH, r, c][-1 -2 -3; -4] :=
            env.edges[SOUTH, _next(r, end), c][1 2 3; 4] *
            state[r, c][9; -2 7 2 5] *
            conj(state[r, c][9; -3 8 3 6]) *
            P_left[SOUTH, r, c][4 5 6; -4] *
            P_right[SOUTH, r, cnext][-1; 1 7 8]
        @diffset @tensor edges[WEST, r, c][-1 -2 -3; -4] :=
            env.edges[WEST, r, _prev(c, end)][1 2 3; 4] *
            state[r, c][9; 5 -2 7 2] *
            conj(state[r, c][9; 6 -3 8 3]) *
            P_left[WEST, r, c][4 5 6; -4] *
            P_right[WEST, rnext, c][-1; 1 7 8]
    end

    @diffset corners[:, :, :] ./= norm.(corners[:, :, :])
    @diffset edges[:, :, :] ./= norm.(edges[:, :, :])
    return corners, edges
end

# One CTMRG iteration with both-sided application of projectors
function ctmrg_iter(state, env::CTMRGEnv, alg::CTMRG{:AllSides})
    # Compute enlarged corners
    Q = enlarge_corners_edges(state, env)

    # Compute projectors if none are supplied
    Pleft, Pright, info = build_projectors(Q, env, alg)

    # Apply projectors and normalize
    corners, edges = renormalize_corners_edges(state, env, Q, Pleft, Pright)

    return CTMRGEnv(corners, edges), (; Pleft, Pright, info...)
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
function build_projectors(Q, env::CTMRGEnv, alg::ProjectorAlg)  # TODO: Add projector type annotations
    Pleft, Pright = Zygote.Buffer.(projector_type(env.edges))
    U, V = Zygote.Buffer.(projector_type(env.edges))
    S = Zygote.Buffer(env.corners)
    ϵ1, ϵ2, ϵ3, ϵ4 = 0, 0, 0, 0
    for c in 1:size(env.corners, 3), r in 1:size(env.corners, 2)
        Pl1, Pr1, info1 = build_projectors(
            Q[1, r, c],
            Q[2, r, _next(c, end)],
            alg;
            trunc=truncspace(space(env.edges[1, r, c], 1)),
            envindex=(1, r, c),
        )
        Pl2, Pr2, info2 = build_projectors(
            Q[2, r, c],
            Q[3, _next(r, end), c],
            alg;
            trunc=truncspace(space(env.edges[2, r, c], 1)),
            envindex=(2, r, c),
        )
        Pl3, Pr3, info3 = build_projectors(
            Q[3, r, c],
            Q[4, r, _prev(c, end)],
            alg;
            trunc=truncspace(space(env.edges[3, r, c], 1)),
            envindex=(3, r, c),
        )
        Pl4, Pr4, info4 = build_projectors(
            Q[4, r, c],
            Q[1, _prev(r, end), c],
            alg;
            trunc=truncspace(space(env.edges[4, r, c], 1)),
            envindex=(4, r, c),
        )

        Pleft[NORTH, r, c] = Pl1
        Pright[NORTH, r, c] = Pr1
        U[NORTH, r, c] = info1.U
        S[NORTH, r, c] = info1.S
        V[NORTH, r, c] = info1.V

        Pleft[EAST, r, c] = Pl2
        Pright[EAST, r, c] = Pr2
        U[EAST, r, c] = info2.U
        S[EAST, r, c] = info2.S
        V[EAST, r, c] = info2.V

        Pleft[SOUTH, r, c] = Pl3
        Pright[SOUTH, r, c] = Pr3
        U[SOUTH, r, c] = info3.U
        S[SOUTH, r, c] = info3.S
        V[SOUTH, r, c] = info3.V

        Pleft[WEST, r, c] = Pl4
        Pright[WEST, r, c] = Pr4
        U[WEST, r, c] = info4.U
        S[WEST, r, c] = info4.S
        V[WEST, r, c] = info4.V
    end
    return copy(Pleft), copy(Pright), (; ϵ=max(ϵ1, ϵ2, ϵ3, ϵ4), U=copy(U), S=copy(S), V=copy(V))
end
function build_projectors(Qleft, Qright, alg::ProjectorAlg; kwargs...)
    # SVD half-infinite environment
    U, S, V, ϵ = PEPSKit.tsvd!(Qleft * Qright, alg.svd_alg; kwargs...)
    ϵ /= norm(S)

    # Compute SVD truncation error and check for degenerate singular values
    ignore_derivatives() do
        if alg.verbosity > 1 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    # Compute projectors
    isqS = sdiag_inv_sqrt(S)
    @tensor Pl[-1 -2 -3; -4] := Qright[-1 -2 -3; 1 2 3] * conj(V[4; 1 2 3]) * isqS[4; -4]
    @tensor Pr[-1; -2 -3 -4] := isqS[-1; 1] * conj(U[2 3 4; 1]) * Qleft[2 3 4; -2 -3 -4]

    return Pl, Pr, (; ϵ, U, S, V)
end

# Apply projectors to renormalize corners and edges
function renormalize_corners_edges(state, env::CTMRGEnv, Q, Pleft, Pright)
    corners::typeof(env.corners) = copy(env.corners)
    edges::typeof(env.edges) = copy(env.edges)
    for c in 1:size(state, 2), r in 1:size(state, 1)
        rprev = _prev(r, size(state, 1))
        rnext = _next(r, size(state, 1))
        cprev = _prev(c, size(state, 2))
        cnext = _next(c, size(state, 2))
        @diffset @tensor corners[NORTHWEST, r, c][-1; -2] :=
            Pright[WEST, rnext, c][-1; 1 2 3] *
            Q[NORTHWEST, r, c][1 2 3; 4 5 6] *
            Pleft[NORTH, r, c][4 5 6; -2]
        @diffset @tensor corners[NORTHEAST, r, c][-1; -2] :=
            Pright[NORTH, r, cprev][-1; 1 2 3] *
            Q[NORTHEAST, r, c][1 2 3; 4 5 6] *
            Pleft[EAST, r, c][4 5 6; -2]
        @diffset @tensor corners[SOUTHEAST, r, c][-1; -2] :=
            Pright[EAST, rprev, c][-1; 1 2 3] *
            Q[SOUTHEAST, r, c][1 2 3; 4 5 6] *
            Pleft[SOUTH, r, c][4 5 6; -2]
        @diffset @tensor corners[SOUTHWEST, r, c][-1; -2] :=
            Pright[SOUTH, r, cnext][-1; 1 2 3] *
            Q[SOUTHWEST, r, c][1 2 3; 4 5 6] *
            Pleft[WEST, r, c][4 5 6; -2]

        @diffset @tensor edges[NORTH, r, c][-1 -2 -3; -4] :=
            env.edges[NORTH, rprev, c][1 2 3; 4] *
            state[r, c][9; 2 5 -2 7] *
            conj(state[r, c][9; 3 6 -3 8]) *
            Pleft[NORTH, r, c][4 5 6; -4] *
            Pright[NORTH, r, cprev][-1; 1 7 8]
        @diffset @tensor edges[EAST, r, c][-1 -2 -3; -4] :=
            env.edges[EAST, r, _next(c, end)][1 2 3; 4] *
            state[r, c][9; 7 2 5 -2] *
            conj(state[r, c][9; 8 3 6 -3]) *
            Pleft[EAST, r, c][4 5 6; -4] *
            Pright[EAST, rprev, c][-1; 1 7 8]
        @diffset @tensor edges[SOUTH, r, c][-1 -2 -3; -4] :=
            env.edges[SOUTH, _next(r, end), c][1 2 3; 4] *
            state[r, c][9; -2 7 2 5] *
            conj(state[r, c][9; -3 8 3 6]) *
            Pleft[SOUTH, r, c][4 5 6; -4] *
            Pright[SOUTH, r, cnext][-1; 1 7 8]
        @diffset @tensor edges[WEST, r, c][-1 -2 -3; -4] :=
            env.edges[WEST, r, _prev(c, end)][1 2 3; 4] *
            state[r, c][9; 5 -2 7 2] *
            conj(state[r, c][9; 6 -3 8 3]) *
            Pleft[WEST, r, c][4 5 6; -4] *
            Pright[WEST, rnext, c][-1; 1 7 8]
    end

    @diffset corners[:, :, :] ./= norm.(corners[:, :, :])
    @diffset edges[:, :, :] ./= norm.(edges[:, :, :])
    return corners, edges
end

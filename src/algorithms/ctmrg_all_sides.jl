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
function enlarge_corners_edges(state, env::CTMRGEnv{C,T}) where {C,T}
    Qtype = tensormaptype(spacetype(C), 3, 3, storagetype(C))
    Q = Zygote.Buffer(Array{Qtype,3}(undef, size(env.corners)))
    drc_combinations = collect(Iterators.product(axes(env.corners)...))
    @fwdthreads for (dir, r, c) in drc_combinations
        Q[dir, r, c] = if dir == NORTHWEST
            northwest_corner((r, c), state, env)
        elseif dir == NORTHEAST
            northeast_corner((r, c), state, env)
        elseif dir == SOUTHEAST
            southeast_corner((r, c), state, env)
        elseif dir == SOUTHWEST
            southwest_corner((r, c), state, env)
        end
    end

    return copy(Q)
end

# Build projectors from SVD and enlarged corners
function build_projectors(Q, env::CTMRGEnv{C,E}, alg::ProjectorAlg{A,T}) where {C,E,A,T}
    P_left, P_right = Zygote.Buffer.(projector_type(env.edges))
    U, V = Zygote.Buffer.(projector_type(env.edges))
    Stype = tensormaptype(spacetype(C), 1, 1, Matrix{real(scalartype(E))})  # Corner type but with real numbers
    S = Zygote.Buffer(Array{Stype,3}(undef, size(env.corners)))
    ϵ = 0.0
    drc_combinations = collect(Iterators.product(axes(env.corners)...))
    @fwdthreads for (dir, r, c) in drc_combinations
        # Row-column index of next enlarged corner
        next_rc = if dir == 1
            (r, _next(c, size(env.corners, 3)))
        elseif dir == 2
            (_next(r, size(env.corners, 2)), c)
        elseif dir == 3
            (r, _prev(c, size(env.corners, 3)))
        elseif dir == 4
            (_prev(r, size(env.corners, 2)), c)
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
            SVDAdjoint(; fwd_alg=fix_svd, rrule_alg=alg.svd_alg.rrule_alg)
        else
            alg.svd_alg
        end
        @autoopt @tensor QQ[χ_in D_inabove D_inbelow; χ_out D_outabove D_outbelow] :=
            Q[dir, r, c][χ_in D_inabove D_inbelow; χ D1 D2] *
            Q[_next(dir, 4), next_rc...][χ D1 D2; χ_out D_outabove D_outbelow]
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
function _contract_new_corner(P_right, Q, P_left)
    return @autoopt @tensor corner[χ_in; χ_out] :=
        P_right[χ_in; χ1 D1 D2] * Q[χ1 D1 D2; χ2 D3 D4] * P_left[χ2 D3 D4; χ_out]
end
function renormalize_corners_edges(state, env::CTMRGEnv, Q, P_left, P_right)
    corners = Zygote.Buffer(copy(env.corners))
    edges = Zygote.Buffer(copy(env.edges))
    rc_combinations = collect(Iterators.product(axes(state)...))
    @fwdthreads for (r, c) in rc_combinations
        rprev = _prev(r, size(state, 1))
        rnext = _next(r, size(state, 1))
        cprev = _prev(c, size(state, 2))
        cnext = _next(c, size(state, 2))

        corners[NORTHWEST, r, c] = _contract_new_corner(
            P_right[WEST, rnext, c], Q[NORTHWEST, r, c], P_left[NORTH, r, c]
        )
        corners[NORTHEAST, r, c] = _contract_new_corner(
            P_right[NORTH, r, cprev], Q[NORTHEAST, r, c], P_left[EAST, r, c]
        )
        corners[SOUTHEAST, r, c] = _contract_new_corner(
            P_right[EAST, rprev, c], Q[SOUTHEAST, r, c], P_left[SOUTH, r, c]
        )
        corners[SOUTHWEST, r, c] = _contract_new_corner(
            P_right[SOUTH, r, cnext], Q[SOUTHWEST, r, c], P_left[WEST, r, c]
        )

        @autoopt @tensor edges[NORTH, r, c][χ_W D_Sab D_Sbe; χ_E] :=
            env.edges[NORTH, rprev, c][χ1 D1 D2; χ2] *
            state[r, c][d; D1 D3 D_Sab D5] *
            conj(state[r, c][d; D2 D4 D_Sbe D6]) *
            P_left[NORTH, r, c][χ2 D3 D4; χ_E] *
            P_right[NORTH, r, cprev][χ_W; χ1 D5 D6]
        @autoopt @tensor edges[EAST, r, c][χ_N D_Wab D_Wbe; χ_S] :=
            env.edges[EAST, r, cnext][χ1 D1 D2; χ2] *
            state[r, c][d; D5 D1 D3 D_Wab] *
            conj(state[r, c][d; D6 D2 D4 D_Wbe]) *
            P_left[EAST, r, c][χ2 D3 D4; χ_S] *
            P_right[EAST, rprev, c][χ_N; χ1 D5 D6]
        @autoopt @tensor edges[SOUTH, r, c][χ_E D_Nab D_Nbe; χ_W] :=
            env.edges[SOUTH, rnext, c][χ1 D1 D2; χ2] *
            state[r, c][d; D_Nab D5 D1 D3] *
            conj(state[r, c][d; D_Nbe D6 D2 D4]) *
            P_left[SOUTH, r, c][χ2 D3 D4; χ_W] *
            P_right[SOUTH, r, cnext][χ_E; χ1 D5 D6]
        @autoopt @tensor edges[WEST, r, c][χ_S D_Eab D_Ebe; χ_N] :=
            env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            state[r, c][d; D3 D_Eab D5 D1] *
            conj(state[r, c][d; D4 D_Ebe D6 D2]) *
            P_left[WEST, r, c][χ2 D3 D4; χ_N] *
            P_right[WEST, rnext, c][χ_S; χ1 D5 D6]
    end

    corners = copy(corners)
    edges = copy(edges)
    corners[:, :, :] ./= norm.(corners[:, :, :])
    edges[:, :, :] ./= norm.(edges[:, :, :])
    return corners, edges
end

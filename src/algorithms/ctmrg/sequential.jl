"""
    sequential_ctmrg_iter(state, envs::CTMRGEnv, alg::CTMRG) -> envs′, info

Perform one sequential iteration of CTMRG, where one iteration consists of four expansion,
renormalization and rotation steps that are performed sequentially.
"""
function sequential_ctmrg_iter(state, envs::CTMRGEnv, alg::CTMRG)
    ϵ = zero(real(scalartype(state)))
    for _ in 1:4 # rotate
        for col in 1:size(state, 2) # left move column-wise
            projectors, info = sequential_projectors(col, state, envs, alg.projector_alg)
            envs = renormalize_sequentially(col, projectors, state, envs)
            ϵ = max(ϵ, info.err)
        end
        state = rotate_north(state, EAST)
        envs = rotate_north(envs, EAST)
    end

    return envs, (; err=ϵ)
end

"""
    sequential_projectors(col::Int, state::InfinitePEPS, envs::CTMRGEnv, alg::ProjectorAlgs)
    sequential_projectors(coordinate, state::InfinitePEPS, envs::CTMRGEnv, alg::ProjectorAlgs)

Compute CTMRG projectors in the `:sequential` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:sequential` scheme).
"""
function sequential_projectors(
    col::Int, state::InfinitePEPS, envs::CTMRGEnv, alg::ProjectorAlgs
)
    ϵ = zero(real(scalartype(envs)))

    # SVD half-infinite environment column-wise
    coordinates = eachcoordinate(envs)[:, col]
    projectors = dtmap(coordinates) do (r, c)
        proj, info = sequential_projectors((WEST, r, c), state, envs, alg)
        ϵ = max(ϵ, info.err / norm(info.S))
        return proj
    end

    return (map(first, projectors), map(last, projectors)), (; err=ϵ)
end
function sequential_projectors(
    coordinate, state::InfinitePEPS, envs::CTMRGEnv, alg::HalfInfiniteProjector
)
    _, r, c = coordinate
    r′ = _prev(r, size(envs, 2))
    Q1 = TensorMap(EnlargedCorner(state, envs, (SOUTHWEST, r, c)), SOUTHWEST)
    Q2 = TensorMap(EnlargedCorner(state, envs, (NORTHWEST, r′, c)), NORTHWEST)
    return compute_projector((Q1, Q2), coordinate, alg)
end
function sequential_projectors(
    coordinate, state::InfinitePEPS, envs::CTMRGEnv, alg::FullInfiniteProjector
)
    _, r, c = coordinate
    r′ = _next(r, size(envs, 2))
    c′ = _next(c, size(envs, 3))
    Q1 = TensorMap(EnlargedCorner(state, envs, (NORTHWEST, r, c)), NORTHWEST)
    Q2 = TensorMap(EnlargedCorner(state, envs, (NORTHEAST, r, c′)), NORTHEAST)
    Q3 = TensorMap(EnlargedCorner(state, envs, (SOUTHEAST, r′, c′)), SOUTHEAST)
    Q4 = TensorMap(EnlargedCorner(state, envs, (SOUTHWEST, r′, c)), SOUTHWEST)
    return compute_projector((Q1, Q2, Q3, Q4), coordinate, alg)
end

"""
    renormalize_sequentially(col::Int, projectors, state, envs)

Renormalize one column of the CTMRG environment.
"""
function renormalize_sequentially(col::Int, projectors, state, envs)
    corners = Zygote.Buffer(envs.corners)
    edges = Zygote.Buffer(envs.edges)

    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir in [SOUTHWEST, NORTHWEST]) && continue
        corners[dir, r, c] = envs.corners[dir, r, c]
    end
    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = envs.edges[dir, r, c]
    end

    # Apply projectors to renormalize corners and edge
    for row in axes(envs.corners, 2)
        C_southwest = renormalize_bottom_corner((row, col), envs, projectors)
        corners[SOUTHWEST, row, col] = C_southwest / norm(C_southwest)

        C_northwest = renormalize_top_corner((row, col), envs, projectors)
        corners[NORTHWEST, row, col] = C_northwest / norm(C_northwest)

        E_west = renormalize_west_edge((row, col), envs, projectors, state)
        edges[WEST, row, col] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

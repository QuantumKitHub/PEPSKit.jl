"""
    SequentialCTMRG(; tol=Defaults.ctmrg_tol, maxiter=Defaults.ctmrg_maxiter,
                      miniter=Defaults.ctmrg_miniter, verbosity=0,
                      projector_alg=typeof(Defaults.projector_alg),
                      svd_alg=SVDAdjoint(), trscheme=FixedSpaceTruncation())

CTMRG algorithm where the expansions and renormalization is performed sequentially
column-wise. This is implemented as a growing and projecting step to the left, followed by
a clockwise rotation (performed four times). The projectors are computed using
`projector_alg` from `svd_alg` SVDs where the truncation scheme is set via `trscheme`.
"""
struct SequentialCTMRG <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::ProjectorAlgorithm
end
function SequentialCTMRG(;
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=2,
    projector_alg=Defaults.projector_alg_type,
    svd_alg=Defaults.svd_alg,
    trscheme=Defaults.trscheme,
)
    return SequentialCTMRG(
        tol, maxiter, miniter, verbosity, projector_alg(; svd_alg, trscheme, verbosity)
    )
end

"""
    ctmrg_leftmove(col::Int, network, env::CTMRGEnv, alg::SequentialCTMRG)

Perform sequential CTMRG left move on the `col`-th column.
"""
function ctmrg_leftmove(col::Int, network, env::CTMRGEnv, alg::SequentialCTMRG)
    #=
        ----> left move
        C1 ← T1 ←   r-1
        ↓    ‖
        T4 = M ==   r
        ↓    ‖
        C4 → T3 →   r+1
        c-1  c 
    =#
    projectors, info = sequential_projectors(col, network, env, alg.projector_alg)
    env = renormalize_sequentially(col, projectors, network, env)
    return env, info
end

function ctmrg_iteration(network, env::CTMRGEnv, alg::SequentialCTMRG)
    truncation_error = zero(real(scalartype(network)))
    condition_number = zero(real(scalartype(network)))
    for _ in 1:4 # rotate
        for col in 1:size(network, 2) # left move column-wise
            env, info = ctmrg_leftmove(col, network, env, alg)
            truncation_error = max(truncation_error, info.truncation_error)
            condition_number = max(condition_number, info.condition_number)
        end
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
    end
    return env, (; truncation_error, condition_number)
end

"""
    sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm)
    sequential_projectors(coordinate::NTuple{3,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:sequential` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:sequential` scheme).
"""
function sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm)
    coordinates = eachcoordinate(env)[:, col]
    proj_and_info = dtmap(coordinates) do (r, c)
        trscheme = truncation_scheme(alg, env.edges[WEST, _prev(r, size(env, 2)), c])
        proj, info = sequential_projectors(
            (WEST, r, c), network, env, @set(alg.trscheme = trscheme)
        )
        return proj, info
    end
    return _split_proj_and_info(proj_and_info)
end
function sequential_projectors(
    coordinate::NTuple{3,Int}, network, env::CTMRGEnv, alg::HalfInfiniteProjector
)
    _, r, c = coordinate
    r′ = _prev(r, size(env, 2))
    Q1 = TensorMap(EnlargedCorner(network, env, (SOUTHWEST, r, c)), SOUTHWEST)
    Q2 = TensorMap(EnlargedCorner(network, env, (NORTHWEST, r′, c)), NORTHWEST)

    svd_alg = svd_algorithm(alg, coordinate)

    return compute_projector(Q1, Q2, svd_alg, alg)
end
function sequential_projectors(
    coordinate::NTuple{3,Int}, network, env::CTMRGEnv, alg::FullInfiniteProjector
)
    rowsize, colsize = size(env)[2:3]
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    ec = (
        TensorMap(EnlargedCorner(network, env, coordinate_se), SOUTHEAST),
        TensorMap(EnlargedCorner(network, env, coordinate), SOUTHWEST),
        TensorMap(EnlargedCorner(network, env, coordinate_nw), NORTHWEST),
        TensorMap(EnlargedCorner(network, env, coordinate_ne), NORTHEAST),
    )
    svd_alg = svd_algorithm(alg, coordinate)
    Q1, Q2 = ec[1] ⊙ ec[2], ec[3] ⊙ ec[4]
    return compute_projector(Q1, Q2, svd_alg, alg)
end

"""
    renormalize_sequentially(col::Int, projectors, network, env)

Renormalize one column of the CTMRG environment.
"""
function renormalize_sequentially(col::Int, projectors, network, env)
    corners = Zygote.Buffer(env.corners)
    edges = Zygote.Buffer(env.edges)

    for (dir, r, c) in eachcoordinate(network, 1:4)
        (c == col && dir in [SOUTHWEST, NORTHWEST]) && continue
        corners[dir, r, c] = env.corners[dir, r, c]
    end
    for (dir, r, c) in eachcoordinate(network, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = env.edges[dir, r, c]
    end

    # Apply projectors to renormalize corners and edge
    for row in axes(env.corners, 2)
        C_southwest = renormalize_bottom_corner((row, col), env, projectors)
        corners[SOUTHWEST, row, col] = C_southwest / norm(C_southwest)

        C_northwest = renormalize_top_corner((row, col), env, projectors)
        corners[NORTHWEST, row, col] = C_northwest / norm(C_northwest)

        E_west = renormalize_west_edge((row, col), env, projectors, network)
        edges[WEST, row, col] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

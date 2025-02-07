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

function ctmrg_iteration(state, env::CTMRGEnv, alg::SequentialCTMRG)
    truncation_error = zero(real(scalartype(state)))
    condition_number = zero(real(scalartype(state)))
    for _ in 1:4 # rotate
        for col in 1:size(state, 2) # left move column-wise
            projectors, info = sequential_projectors(col, state, env, alg.projector_alg)
            env = renormalize_sequentially(col, projectors, state, env)
            truncation_error = max(truncation_error, info.truncation_error)
            condition_number = max(condition_number, info.condition_number)
        end
        state = rotate_north(state, EAST)
        env = rotate_north(env, EAST)
    end

    return env, (; truncation_error, condition_number)
end

"""
    sequential_projectors(col::Int, state::InfinitePEPS, env::CTMRGEnv, alg::ProjectorAlgorithm)
    sequential_projectors(coordinate::NTuple{3,Int}, state::InfinitePEPS, env::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:sequential` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:sequential` scheme).
"""
function sequential_projectors(
    col::Int, state::InfiniteSquareNetwork, env::CTMRGEnv, alg::ProjectorAlgorithm
)
    # SVD half-infinite environment column-wise
    T = promote_type(real(scalartype(state)), real(scalartype(env)))
    ϵ = Zygote.Buffer(zeros(T, size(env, 2)))
    S = Zygote.Buffer(zeros(size(env, 2)), tensormaptype(spacetype(eltype(state)), 1, 1, T))
    coordinates = eachcoordinate(env)[:, col]
    projectors = dtmap(coordinates) do (r, c)
        trscheme = truncation_scheme(alg, env.edges[WEST, _prev(r, size(env, 2)), c])
        proj, info = sequential_projectors(
            (WEST, r, c), state, env, @set(alg.trscheme = trscheme)
        )
        S[r] = info.S
        ϵ[r] = info.err / norm(info.S)
        return proj
    end

    truncation_error = maximum(copy(ϵ))
    condition_number = maximum(_condition_number, S)
    info = (; truncation_error, condition_number)
    return (map(first, projectors), map(last, projectors)), info
end
function sequential_projectors(
    coordinate::NTuple{3,Int},
    state::InfiniteSquareNetwork,
    env::CTMRGEnv,
    alg::HalfInfiniteProjector,
)
    _, r, c = coordinate
    r′ = _prev(r, size(env, 2))
    Q1 = TensorMap(EnlargedCorner(state, env, (SOUTHWEST, r, c)), SOUTHWEST)
    Q2 = TensorMap(EnlargedCorner(state, env, (NORTHWEST, r′, c)), NORTHWEST)
    return compute_projector((Q1, Q2), coordinate, alg)
end
function sequential_projectors(
    coordinate::NTuple{3,Int},
    state::InfiniteSquareNetwork,
    env::CTMRGEnv,
    alg::FullInfiniteProjector,
)
    rowsize, colsize = size(env)[2:3]
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    ec = (
        TensorMap(EnlargedCorner(state, env, coordinate_se), SOUTHEAST),
        TensorMap(EnlargedCorner(state, env, coordinate), SOUTHWEST),
        TensorMap(EnlargedCorner(state, env, coordinate_nw), NORTHWEST),
        TensorMap(EnlargedCorner(state, env, coordinate_ne), NORTHEAST),
    )
    return compute_projector(ec, coordinate, alg)
end

"""
    renormalize_sequentially(col::Int, projectors, state, env)

Renormalize one column of the CTMRG environment.
"""
function renormalize_sequentially(col::Int, projectors, state, env)
    corners = Zygote.Buffer(env.corners)
    edges = Zygote.Buffer(env.edges)

    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir in [SOUTHWEST, NORTHWEST]) && continue
        corners[dir, r, c] = env.corners[dir, r, c]
    end
    for (dir, r, c) in eachcoordinate(state, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = env.edges[dir, r, c]
    end

    # Apply projectors to renormalize corners and edge
    for row in axes(env.corners, 2)
        C_southwest = renormalize_bottom_corner((row, col), env, projectors)
        corners[SOUTHWEST, row, col] = C_southwest / norm(C_southwest)

        C_northwest = renormalize_top_corner((row, col), env, projectors)
        corners[NORTHWEST, row, col] = C_northwest / norm(C_northwest)

        E_west = renormalize_west_edge((row, col), env, projectors, state)
        edges[WEST, row, col] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

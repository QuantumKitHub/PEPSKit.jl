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

function ctmrg_iteration(state, envs::CTMRGEnv, alg::SequentialCTMRG)
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
    sequential_projectors(col::Int, state::InfinitePEPS, envs::CTMRGEnv, alg::ProjectorAlgorithm)
    sequential_projectors(coordinate::NTuple{3,Int}, state::InfinitePEPS, envs::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:sequential` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:sequential` scheme).
"""
function sequential_projectors(
    col::Int, state::InfiniteSquareNetwork, envs::CTMRGEnv, alg::ProjectorAlgorithm
)
    # SVD half-infinite environment column-wise
    ϵ = Zygote.Buffer(zeros(real(scalartype(envs)), size(envs, 2)))
    coordinates = eachcoordinate(envs)[:, col]
    projectors = dtmap(coordinates) do (r, c)
        trscheme = truncation_scheme(alg, envs.edges[WEST, _prev(r, size(envs, 2)), c])
        proj, info = sequential_projectors(
            (WEST, r, c), state, envs, @set(alg.trscheme = trscheme)
        )
        ϵ[r] = info.err / norm(info.S)
        return proj
    end

    return (map(first, projectors), map(last, projectors)), (; err=maximum(copy(ϵ)))
end
function sequential_projectors(
    coordinate::NTuple{3,Int},
    state::InfiniteSquareNetwork,
    envs::CTMRGEnv,
    alg::HalfInfiniteProjector,
)
    _, r, c = coordinate
    r′ = _prev(r, size(envs, 2))
    Q1 = TensorMap(EnlargedCorner(state, envs, (SOUTHWEST, r, c)), SOUTHWEST)
    Q2 = TensorMap(EnlargedCorner(state, envs, (NORTHWEST, r′, c)), NORTHWEST)
    return compute_projector((Q1, Q2), coordinate, alg)
end
function sequential_projectors(
    coordinate::NTuple{3,Int},
    state::InfiniteSquareNetwork,
    envs::CTMRGEnv,
    alg::FullInfiniteProjector,
)
    rowsize, colsize = size(envs)[2:3]
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    ec = (
        TensorMap(EnlargedCorner(state, envs, coordinate_se), SOUTHEAST),
        TensorMap(EnlargedCorner(state, envs, coordinate), SOUTHWEST),
        TensorMap(EnlargedCorner(state, envs, coordinate_nw), NORTHWEST),
        TensorMap(EnlargedCorner(state, envs, coordinate_ne), NORTHEAST),
    )
    return compute_projector(ec, coordinate, alg)
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

"""
    struct SequentialCTMRG <: CTMRGAlgorithm

CTMRG algorithm where the expansions and renormalization is performed sequentially
column-wise. This is implemented as a growing and projecting step to the left, followed by
a clockwise rotation (performed four times).

## Fields

$(TYPEDFIELDS)

## Constructors

    SequentialCTMRG(; kwargs...)

Construct a sequential CTMRG algorithm struct based on keyword arguments.
For a full description, see [`leading_boundary`](@ref). The supported keywords are:

* `tol::Real=$(Defaults.ctmrg_tol)`
* `maxiter::Int=$(Defaults.ctmrg_maxiter)`
* `miniter::Int=$(Defaults.ctmrg_miniter)`
* `verbosity::Int=$(Defaults.ctmrg_verbosity)`
* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg::Symbol=:$(Defaults.trscheme))`
* `svd_alg::Union{<:SVDAdjoint,NamedTuple}`
* `projector_alg::Symbol=:$(Defaults.projector_alg)`
"""
struct SequentialCTMRG{P<:ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function SequentialCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg=:sequential, kwargs...)
end

CTMRG_SYMBOLS[:sequential] = SequentialCTMRG

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
            @info "left move on $col"
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
    tile = tiling(network)
    tile_slice = tile[:, col]
    coordinates = indexin(unique(tile_slice), tile)
    T_dst = Base.promote_op(
        sequential_projectors, CartesianIndex{3}, typeof(network), typeof(env), typeof(alg)
    )
    proj_and_info′ = similar(coordinates, T_dst)
    proj_and_info::typeof(proj_and_info′) = dtmap!!(proj_and_info′, coordinates) do I
        trscheme = truncation_scheme(alg, env.edges[WEST, I - CartesianIndex(1, 0)])
        I′ = CartesianIndex(WEST, Tuple(I)...)
        return sequential_projectors(I′, network, env, @set(alg.trscheme = trscheme))
    end
    Ps, info = _split_proj_and_info(proj_and_info′)
    Ps_tiled = InfiniteTiledArray.(Ps, Ref(tile_slice))
    return Ps_tiled, info
end
function sequential_projectors(
    coordinate::CartesianIndex{3}, network, env::CTMRGEnv, alg::HalfInfiniteProjector
)
    dir, r, c = Tuple(coordinate)
    @assert dir == WEST
    r′ = _prev(r, size(env, 2))
    Q1 = TensorMap(EnlargedCorner(network, env, (SOUTHWEST, r, c)))
    Q2 = TensorMap(EnlargedCorner(network, env, (NORTHWEST, r′, c)))
    return compute_projector((Q1, Q2), coordinate, alg)
end
function sequential_projectors(
    coordinate::CartesianIndex{3}, network, env::CTMRGEnv, alg::FullInfiniteProjector
)
    rowsize, colsize = size(env, 2), size(env, 3)
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    ec = (
        TensorMap(EnlargedCorner(network, env, coordinate_se)),
        TensorMap(EnlargedCorner(network, env, coordinate)),
        TensorMap(EnlargedCorner(network, env, coordinate_nw)),
        TensorMap(EnlargedCorner(network, env, coordinate_ne)),
    )
    return compute_projector(ec, coordinate, alg)
end

"""
    renormalize_sequentially(col::Int, projectors, network, env)

Renormalize one column of the CTMRG environment.
"""
function renormalize_sequentially(col::Int, projectors, network, env)
    # copy over everything, overwrite afterwards
    corners_data = Zygote.Buffer(tilingdata(env.corners))
    corners_data .= tilingdata(env.corners)
    edges_data = Zygote.Buffer(tilingdata(env.edges))
    edges_data .= tilingdata(env.edges)

    # TODO: this is a very ugly hack to deal with Zygote not being differentiable
    # when mutating elements
    tile = tiling(network)
    rows = unique(row -> tile[row, col], axes(tile, 1))
    env_tile = tiling(env)

    # Apply projectors to renormalize corners and edge
    for row in rows
        I = CartesianIndex(row, col)
        @assert I[2] == col
        C_southwest = renormalize_bottom_corner(I, env, projectors)
        corners_data[env_tile[SOUTHWEST, I]] = normalize(C_southwest)

        C_northwest = renormalize_top_corner(I, env, projectors)
        corners_data[env_tile[NORTHWEST, I]] = normalize(C_northwest)

        E_west = renormalize_west_edge(I, env, projectors, network)
        edges_data[env_tile[WEST, I]] = normalize(E_west)
    end

    corners = InfiniteTiledArray(copy(corners_data), env_tile)
    edges = InfiniteTiledArray(copy(edges_data), env_tile)

    return CTMRGEnv(corners, edges)
end

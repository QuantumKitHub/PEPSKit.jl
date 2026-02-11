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
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))`
* `decomposition_alg::Union{<:SVDAdjoint,NamedTuple}`
* `projector_alg::Symbol=:$(Defaults.projector_alg)`
"""
struct SequentialCTMRG{P <: ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function SequentialCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg = :sequential, kwargs...)
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
    T_dst = Base.promote_op(
        sequential_projectors, NTuple{3, Int}, typeof(network), typeof(env), typeof(alg)
    )
    proj_and_info = similar(coordinates, T_dst)
    proj_and_info′::typeof(proj_and_info) = dtmap!!(proj_and_info, coordinates) do (r, c)
        proj, info = sequential_projectors((WEST, r, c), network, env, alg)
        return proj, info
    end
    return _split_proj_and_info(proj_and_info′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::HalfInfiniteProjector
    )
    _, r, c = coordinate
    r′ = _prev(r, size(env, 2))
    trunc = truncation_strategy(alg, env.edges[WEST, r′, c])
    alg′ = @set alg.trunc = trunc
    Q1 = TensorMap(EnlargedCorner(network, env, (SOUTHWEST, r, c)))
    Q2 = TensorMap(EnlargedCorner(network, env, (NORTHWEST, r′, c)))
    return compute_projector((Q1, Q2), coordinate, alg′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::FullInfiniteProjector
    )
    rowsize, colsize = size(env)[2:3]
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    trunc = truncation_strategy(alg, env.edges[WEST, coordinate_nw[2:3]...])
    alg′ = @set alg.trunc = trunc
    ec = (
        TensorMap(EnlargedCorner(network, env, coordinate_se)),
        TensorMap(EnlargedCorner(network, env, coordinate)),
        TensorMap(EnlargedCorner(network, env, coordinate_nw)),
        TensorMap(EnlargedCorner(network, env, coordinate_ne)),
    )
    return compute_projector(ec, coordinate, alg′)
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
        C_southwest = renormalize_southwest_corner((row, col), env, projectors)
        corners[SOUTHWEST, row, col] = C_southwest / norm(C_southwest)

        C_northwest = renormalize_northwest_corner((row, col), env, projectors)
        corners[NORTHWEST, row, col] = C_northwest / norm(C_northwest)

        E_west = renormalize_west_edge((row, col), env, projectors, network)
        edges[WEST, row, col] = E_west / norm(E_west)
    end

    return CTMRGEnv(copy(corners), copy(edges))
end

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
* `projector_alg::Union{Symbol,NamedTuple}=:$(Defaults.projector_alg)`
"""
struct SequentialCTMRG{P <: ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function SequentialCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg = :SequentialCTMRG, kwargs...)
end

CTMRG_SYMBOLS[:SequentialCTMRG] = SequentialCTMRG

function initialize_projector_cache(
        network, env, alg::SequentialCTMRG{<:SubspaceIterationProjector}
    )
    return initialize_subspace_iteration_cache(network, env, alg.projector_alg)
end

function check_input(
        ::typeof(leading_boundary), network, env,
        alg::SequentialCTMRG{<:SubspaceIterationProjector},
    )
    return _check_subspace_iteration_input(network, env, alg.projector_alg)
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

function ctmrg_leftmove(
        col::Int, network, env::CTMRGEnv,
        alg::SequentialCTMRG{<:SubspaceIterationProjector}, X, Y, recycle::Bool,
    )
    projectors, info = sequential_projectors(
        col, network, env, alg.projector_alg, X, Y, recycle
    )
    env′ = renormalize_sequentially(col, projectors, network, env)
    return env′, info
end

function ctmrg_iteration(
        network, env::CTMRGEnv, alg::SequentialCTMRG,
        cache::EmptyProjectorCache,
    )
    truncation_error = zero(real(scalartype(network)))
    for _ in 1:4 # rotate
        for col in 1:size(network, 2) # left move column-wise
            env, info = ctmrg_leftmove(col, network, env, alg)
            ignore_derivatives() do
                truncation_error = max(truncation_error, info.truncation_error)
            end
        end
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
    end
    return env, (; contraction_metrics = (; truncation_error)), cache
end

function ctmrg_iteration(
        network, env::CTMRGEnv,
        alg::SequentialCTMRG{<:SubspaceIterationProjector},
        cache::SubspaceIterationCache,
    )
    truncation_error = zero(real(scalartype(network)))

    # Work on copies so the input cache remains unchanged throughout this functional update.
    expected_size = size(env)
    cache = if all(grid -> size(grid) == expected_size, (cache.X, cache.Y, cache.U, cache.V))
        copy(cache)
    else
        initialize_projector_cache(network, env, alg)
    end
    Xcache, Ycache = cache.X, cache.Y
    Ucache, Vcache = cache.U, cache.V
    gridsize = size(env)[2:3]

    # Perform four directional moves, expressing each one as a column left-move.
    recycled = true
    for _ in 1:4
        # Select the rangefinders aligned with the current left-move direction.
        Xdirection = Xcache[WEST, :, :]
        Ydirection = Ycache[WEST, :, :]

        # Collect the updated rangefinders and dominant subspaces for this directional phase.
        Xphase = similar(Xdirection)
        Yphase = similar(Ydirection)
        Uphase = similar(Ucache[WEST, :, :])
        Vphase = similar(Vcache[WEST, :, :])
        for col in 1:size(network, 2)
            env, info = ctmrg_leftmove(
                col, network, env, alg, Xdirection, Ydirection, cache.recycle
            )
            Xphase[:, col] .= vec(info.X)
            Yphase[:, col] .= vec(info.Y)
            Uphase[:, col] .= vec(info.U)
            Vphase[:, col] .= vec(info.V)
            recycled &= info.recycled
            ignore_derivatives() do
                truncation_error = max(truncation_error, info.truncation_error)
            end
        end

        Xcache[WEST, :, :] = Xphase
        Ycache[WEST, :, :] = Yphase
        Ucache[WEST, :, :] = Uphase
        Vcache[WEST, :, :] = Vphase

        # Rotate the network, environment, and cache into the frame of the next left move.
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
        rotated_cache = rotl90(
            SubspaceIterationCache(
                cache.iteration, cache.recycle, Xcache, Ycache, Ucache, Vcache
            )
        )
        Xcache, Ycache = rotated_cache.X, rotated_cache.Y
        Ucache, Vcache = rotated_cache.U, rotated_cache.V
        gridsize = reverse(gridsize)
    end

    # Compare successive dominant subspaces and decide whether Eq. (5) can be skipped next time.
    subspace_error = _subspace_error(cache.U, cache.V, Ucache, Vcache)
    iteration = cache.iteration + 1
    recycle = iteration >= alg.projector_alg.min_subspace_iters &&
        subspace_error < alg.projector_alg.subspace_tol
    cache′ = SubspaceIterationCache(iteration, recycle, Xcache, Ycache, Ucache, Vcache)
    info = (; contraction_metrics = (; truncation_error, subspace_error, recycled))
    return env, info, cache′
end

"""
    sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm)
    sequential_projectors(coordinate::NTuple{3,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:SequentialCTMRG` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:SequentialCTMRG` scheme).
"""
function sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm)
    coordinates = eachcoordinate(env)[:, col]
    T_dst = Base.promote_op(
        sequential_projectors, NTuple{3, Int}, typeof(network), typeof(env), typeof(alg)
    )
    proj_and_info = similar(coordinates, T_dst)
    proj_and_info′::typeof(proj_and_info) = dtmap!!(proj_and_info, coordinates) do (r, c)
        return sequential_projectors((WEST, r, c), network, env, alg)
    end
    return _split_proj_and_info(proj_and_info′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::HalfInfiniteProjector
    )
    ec, trunc = enlarged_corner_inputs(network, coordinate, env, alg, Val(2))
    alg´ = _set_decomposition_truncation(alg, trunc)
    return compute_projector(ec, alg´)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::FullInfiniteProjector
    )
    ec, trunc = enlarged_corner_inputs(network, coordinate, env, alg, Val(4))
    alg´ = _set_decomposition_truncation(alg, trunc)
    return compute_projector(ec, alg´)
end

# For SubspaceIterationProjector
function sequential_projectors(
        col::Int, network, env::CTMRGEnv,
        alg::SubspaceIterationProjector, X, Y, recycle::Bool,
    )
    coordinates = eachcoordinate(env)[:, col]
    proj_and_info = similar(coordinates, Any)
    proj_and_info′::typeof(proj_and_info) = dtmap!!(proj_and_info, coordinates) do (r, c)
        return sequential_projectors(
            (WEST, r, c), network, env, alg, X[r, c], Y[r, c], recycle
        )
    end
    return _split_subspace_proj_and_info(proj_and_info′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv,
        alg::SubspaceIterationProjector, X, Y, recycle::Bool,
    )
    ec, trunc = enlarged_corner_inputs(network, coordinate, env, alg, Val(4))
    return compute_projector(ec, alg, (; X, Y, trunc, recycle))
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
        corners[dir, r, c] = corner(env, dir, r, c)
    end
    for (dir, r, c) in eachcoordinate(network, 1:4)
        (c == col && dir == WEST) && continue
        edges[dir, r, c] = edge(env, dir, r, c)
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

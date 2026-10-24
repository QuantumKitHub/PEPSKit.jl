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
        network, env, alg::Union{SimultaneousCTMRG{<:SubspaceIterationProjector}, SequentialCTMRG{<:SubspaceIterationProjector}}
    )
    return initialize_subspace_iteration_cache(network, env, alg.projector_alg)
end

function check_input(
        ::typeof(leading_boundary), network, env,
        alg::Union{SimultaneousCTMRG{<:SubspaceIterationProjector}, SequentialCTMRG{<:SubspaceIterationProjector}},
    )
    return _check_subspace_iteration_input(network, env, alg.projector_alg)
end

"""
    ctmrg_leftmove(col::Int, network, env::CTMRGEnv, alg::SequentialCTMRG)

Perform sequential CTMRG left move on the `col`-th column.
"""
function ctmrg_leftmove(col::Int, network, env::CTMRGEnv, alg::SequentialCTMRG, args...)
    #=
        ----> left move
        C1 ← T1 ←   r-1
        ↓    ‖
        T4 = M ==   r
        ↓    ‖
        C4 → T3 →   r+1
        c-1  c 
    =#
    projectors, info = sequential_projectors(col, network, env, alg.projector_alg, args...)
    env = renormalize_sequentially(col, projectors, network, env)
    return env, info
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

"""Replace the updated direction using projector columns without mutating the old cache."""
function _replace_subspace_phase(grid, columns, key::Symbol)
    isnothing(grid) && return nothing
    return map(CartesianIndices(grid)) do I
        return I[1] == WEST ? getproperty(columns[I[3]], key)[I[2]] : grid[I]
    end
end

function ctmrg_iteration(
        network, env::CTMRGEnv,
        alg::SequentialCTMRG{<:SubspaceIterationProjector},
        cache::SubspaceIterationCache,
    )
    truncation_error = zero(real(scalartype(network)))

    if !_compatible_subspace_cache(cache, env, alg.projector_alg)
        cache = initialize_projector_cache(network, env, alg)
    end
    grids = (cache.X, cache.Y, cache.U, cache.V)

    # Perform four directional moves, expressing each one as a column left-move.
    recycled = true
    freezable = true
    for _ in 1:4
        # Select the rangefinders aligned with the current left-move direction.
        Xdirection = grids[1][WEST, :, :]
        Ydirection = _subspace_grid_entry(grids[2], WEST, :, :)

        columns = ()
        for col in 1:size(network, 2)
            env, info = ctmrg_leftmove(
                col, network, env, alg, Xdirection, Ydirection, cache.recycle
            )
            columns = (columns..., info)
            recycled &= info.recycled
            freezable &= info.freezable
            ignore_derivatives() do
                truncation_error = max(truncation_error, info.truncation_error)
            end
        end

        grids = map(
            (grid, key) -> _rotate_subspace_grid(_replace_subspace_phase(grid, columns, key)),
            grids, (:X, :Y, :U, :V),
        )

        # Rotate the network, environment, and cache into the frame of the next left move.
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
    end

    cache′, subspace_error = _update_subspace_cache(
        cache, grids...,
        alg.projector_alg, freezable
    )
    info = (; contraction_metrics = (; truncation_error, subspace_error, recycled))
    return env, info, cache′
end

"""
    sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm)
    sequential_projectors(coordinate::NTuple{3,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:SequentialCTMRG` scheme either for an entire column `col` or
for a specific `coordinate` (where `dir=WEST` is already implied in the `:SequentialCTMRG` scheme).
"""
function sequential_projectors(col::Int, network, env::CTMRGEnv, alg::ProjectorAlgorithm, args...)
    coordinates = eachcoordinate(env)[:, col]
    T_dst = Base.promote_op(
        sequential_projectors, NTuple{3, Int}, typeof(network), typeof(env), typeof(alg),
        map(typeof, _projector_arguments(first(coordinates), args...))...,
    )
    proj_and_info = similar(coordinates, T_dst)
    proj_and_info′::typeof(proj_and_info) = dtmap!!(proj_and_info, coordinates) do (r, c)
        return sequential_projectors((WEST, r, c), network, env, alg, _projector_arguments((r, c), args...)...)
    end
    return _split_proj_and_info(proj_and_info′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::Union{HalfInfiniteProjector, FullInfiniteProjector}
    )
    ec, trunc = enlarged_corner_inputs(network, coordinate, env, alg, Val(alg isa HalfInfiniteProjector ? 2 : 4))
    alg´ = _set_decomposition_truncation(alg, trunc)
    return compute_projector(ec, alg´)
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

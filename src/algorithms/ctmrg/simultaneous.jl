"""
$(TYPEDEF)

CTMRG algorithm where all sides are grown and renormalized at the same time. In particular,
the projectors are applied to the corners from two sides simultaneously.

## Fields

$(TYPEDFIELDS)

## Constructors

    SimultaneousCTMRG(; kwargs...)

Construct a simultaneous CTMRG algorithm struct based on keyword arguments.
For a full description, see [`leading_boundary`](@ref). The supported keywords are:

* `tol::Real=$(Defaults.ctmrg_tol)`
* `maxiter::Int=$(Defaults.ctmrg_maxiter)`
* `miniter::Int=$(Defaults.ctmrg_miniter)`
* `verbosity::Int=$(Defaults.ctmrg_verbosity)`
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))`
* `decomposition_alg::Union{<:SVDAdjoint,NamedTuple}`
* `projector_alg::Union{Symbol,NamedTuple}=:$(Defaults.projector_alg)`
"""
struct SimultaneousCTMRG{P <: ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function SimultaneousCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg = :SimultaneousCTMRG, kwargs...)
end

CTMRG_SYMBOLS[:SimultaneousCTMRG] = SimultaneousCTMRG

"""Expand every corner tensor for a simultaneous CTMRG iteration."""
function _expand_corners_simultaneously(network, env::CTMRGEnv)
    coordinates = eachcoordinate(network, 1:4)
    T_corners = Base.promote_op(
        TensorMap ∘ EnlargedCorner, typeof(network), typeof(env), eltype(coordinates)
    )
    enlarged_corners′ = similar(coordinates, T_corners)
    enlarged_corners::typeof(enlarged_corners′) =
        dtmap!!(enlarged_corners′, eachcoordinate(network, 1:4)) do idx
        return TensorMap(EnlargedCorner(network, env, idx))
    end
    return enlarged_corners
end

function ctmrg_iteration(
        network, env::CTMRGEnv, alg::SimultaneousCTMRG,
        cache::EmptyProjectorCache,
    )
    enlarged_corners = _expand_corners_simultaneously(network, env)
    projectors, info = simultaneous_projectors(enlarged_corners, env, alg.projector_alg)  # compute projectors on all coordinates
    env′ = renormalize_simultaneously(enlarged_corners, projectors, network, env)  # renormalize enlarged corners
    info = (;
        contraction_metrics = (; info.truncation_error),
        info.U, info.S, info.V,
    )
    return env′, info, cache
end

function initialize_projector_cache(
        network, env, alg::SimultaneousCTMRG{<:SubspaceIterationProjector}
    )
    return initialize_subspace_iteration_cache(network, env, alg.projector_alg)
end

function check_input(
        ::typeof(leading_boundary), network, env,
        alg::SimultaneousCTMRG{<:SubspaceIterationProjector},
    )
    return _check_subspace_iteration_input(network, env, alg.projector_alg)
end

function ctmrg_iteration(
        network, env::CTMRGEnv,
        alg::SimultaneousCTMRG{<:SubspaceIterationProjector},
        cache::SubspaceIterationCache,
    )
    if any(grid -> size(grid) != size(env), (cache.X, cache.Y, cache.U, cache.V))
        cache = initialize_projector_cache(network, env, alg)
    end
    enlarged_corners = _expand_corners_simultaneously(network, env)
    projectors, projector_info = simultaneous_projectors(
        enlarged_corners, env, alg.projector_alg, cache.X, cache.Y, cache.recycle
    )
    env′ = renormalize_simultaneously(enlarged_corners, projectors, network, env)
    subspace_error = _subspace_error(cache.U, cache.V, projector_info.U, projector_info.V)
    iteration = cache.iteration + 1
    recycle = iteration >= alg.projector_alg.min_subspace_iters &&
        subspace_error < alg.projector_alg.subspace_tol
    info = (;
        contraction_metrics = (;
            projector_info.truncation_error, subspace_error,
            recycled = projector_info.recycled,
        ),
        projector_info.U, projector_info.S, projector_info.V,
    )
    cache′ = SubspaceIterationCache(
        iteration, recycle,
        projector_info.X, projector_info.Y, projector_info.U, projector_info.V,
    )
    return env′, info, cache′
end

# Work-around to stop Zygote from choking on first execution (sometimes)
# Split up map returning projectors and info into separate arrays
function _split_proj_and_info(proj_and_info)
    P_left = map(x -> x[1][1], proj_and_info)
    P_right = map(x -> x[1][2], proj_and_info)
    truncation_error = maximum(x -> x[2].truncation_error, proj_and_info)
    U = map(x -> x[2].U, proj_and_info)
    S = map(x -> x[2].S, proj_and_info)
    V = map(x -> x[2].V, proj_and_info)
    info = (; truncation_error, U, S, V)
    return (P_left, P_right), info
end

"""Split SI projector results into projector, diagnostic, and rangefinder grids."""
function _split_subspace_proj_and_info(proj_and_info)
    projectors, info = _split_proj_and_info(proj_and_info)
    X = map(x -> x[2].X, proj_and_info)
    Y = map(x -> x[2].Y, proj_and_info)
    recycled = all(x -> x[2].recycled, proj_and_info)
    return projectors, (; info..., recycled, X, Y)
end

"""
    simultaneous_projectors(enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::ProjectorAlgorithm)
    simultaneous_projectors(coordinate, enlarged_corners::Array{E,3}, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:SimultaneousCTMRG` scheme either for all provided
enlarged corners or on a specific `coordinate`.
"""
function simultaneous_projectors(
        enlarged_corners::Array{E, 3}, env::CTMRGEnv, alg::ProjectorAlgorithm
    ) where {E}
    coordinates = eachcoordinate(env, 1:4)
    T_dst = Base.promote_op(
        simultaneous_projectors,
        NTuple{3, Int}, typeof(enlarged_corners), typeof(env), typeof(alg),
    )
    proj_and_info′ = similar(coordinates, T_dst)
    proj_and_info::typeof(proj_and_info′) =
        dtmap!!(proj_and_info′, coordinates) do coordinate
        return simultaneous_projectors(coordinate, enlarged_corners, env, alg)
    end
    return _split_proj_and_info(proj_and_info)
end
function simultaneous_projectors(
        coordinate, enlarged_corners::Array{E, 3}, env, alg::HalfInfiniteProjector
    ) where {E}
    ec, trunc = enlarged_corner_inputs(enlarged_corners, coordinate, env, alg, Val(2))
    alg′ = _set_decomposition_truncation(alg, trunc)
    return compute_projector(ec, alg′)
end
function simultaneous_projectors(
        coordinate::NTuple{3, Int}, enlarged_corners::Array{E, 3}, env,
        alg::FullInfiniteProjector,
    ) where {E}
    ec, trunc = enlarged_corner_inputs(enlarged_corners, coordinate, env, alg, Val(4))
    alg′ = _set_decomposition_truncation(alg, trunc)
    return compute_projector(ec, alg′)
end

# For SubspaceIterationProjector
function simultaneous_projectors(
        enlarged_corners::Array{E, 3}, env::CTMRGEnv,
        alg::SubspaceIterationProjector, X, Y, recycle::Bool,
    ) where {E}
    coordinates = eachcoordinate(env, 1:4)
    T_dst = Base.promote_op(
        simultaneous_projectors,
        NTuple{3, Int}, typeof(enlarged_corners), typeof(env), typeof(alg), Any, Any, Bool,
    )
    proj_and_info′ = similar(coordinates, T_dst)
    proj_and_info::typeof(proj_and_info′) =
        dtmap!!(proj_and_info′, coordinates) do coordinate
        return simultaneous_projectors(
            coordinate, enlarged_corners, env, alg,
            X[coordinate...], Y[coordinate...], recycle,
        )
    end
    return _split_subspace_proj_and_info(proj_and_info)
end

function simultaneous_projectors(
        coordinate::NTuple{3, Int}, enlarged_corners::Array{E, 3}, env,
        alg::SubspaceIterationProjector, X, Y, recycle::Bool,
    ) where {E}
    ec, trunc = enlarged_corner_inputs(enlarged_corners, coordinate, env, alg, Val(4))
    return compute_projector(ec, alg, (; X, Y, trunc, recycle))
end

"""
$(SIGNATURES)

Renormalize all enlarged corners and edges simultaneously.
"""
function renormalize_simultaneously(enlarged_corners, projectors, network, env)
    P_left, P_right = projectors
    coordinates = eachcoordinate(env, 1:4)
    T_CE = Tuple{cornertype(env), edgetype(env)}
    corners_edges′ = similar(coordinates, T_CE)
    corners_edges::typeof(corners_edges′) =
        dtmap!!(corners_edges′, coordinates) do (dir, r, c)
        if dir == NORTH
            corner = renormalize_northwest_corner(
                (r, c), enlarged_corners, P_left, P_right
            )
            edge = renormalize_north_edge((r, c), env, P_left, P_right, network)
        elseif dir == EAST
            corner = renormalize_northeast_corner(
                (r, c), enlarged_corners, P_left, P_right
            )
            edge = renormalize_east_edge((r, c), env, P_left, P_right, network)
        elseif dir == SOUTH
            corner = renormalize_southeast_corner(
                (r, c), enlarged_corners, P_left, P_right
            )
            edge = renormalize_south_edge((r, c), env, P_left, P_right, network)
        elseif dir == WEST
            corner = renormalize_southwest_corner(
                (r, c), enlarged_corners, P_left, P_right
            )
            edge = renormalize_west_edge((r, c), env, P_left, P_right, network)
        end
        return corner / norm(corner), edge / norm(edge)
    end

    return CTMRGEnv(map(first, corners_edges), map(last, corners_edges))
end

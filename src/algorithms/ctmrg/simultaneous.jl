"""
    SimultaneousCTMRG(; tol=$(Defaults.ctmrg_tol), maxiter=$(Defaults.ctmrg_maxiter),
                      miniter=$(Defaults.ctmrg_miniter), verbosity=$(Defaults.ctmrg_verbosity),
                      svd_alg=SVDAdjoint(), trscheme=truncation_scheme_symbols[Defaults.trscheme],
                      projector_alg=projector_symbols[Defaults.projector_alg])

CTMRG algorithm where all sides are grown and renormalized at the same time. In particular,
the projectors are applied to the corners from two sides simultaneously. The projectors are
computed using `projector_alg` from `svd_alg` SVDs where the truncation scheme is set via 
`trscheme`.
"""
struct SimultaneousCTMRG <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::ProjectorAlgorithm
end
function SimultaneousCTMRG(;
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    svd_alg=SVDAdjoint(),
    trscheme=truncation_scheme_symbols[Defaults.trscheme],
    projector_alg=projector_symbols[Defaults.projector_alg],
)
    return SimultaneousCTMRG(
        tol, maxiter, miniter, verbosity, projector_alg(; svd_alg, trscheme, verbosity)
    )
end

function ctmrg_iteration(network, env::CTMRGEnv, alg::SimultaneousCTMRG)
    enlarged_corners = dtmap(eachcoordinate(network, 1:4)) do idx
        return TensorMap(EnlargedCorner(network, env, idx), idx[1])
    end  # expand environment
    projectors, info = simultaneous_projectors(enlarged_corners, env, alg.projector_alg)  # compute projectors on all coordinates
    env′ = renormalize_simultaneously(enlarged_corners, projectors, network, env)  # renormalize enlarged corners
    return env′, info
end

# Work-around to stop Zygote from choking on first execution (sometimes)
# Split up map returning projectors and info into separate arrays
function _split_proj_and_info(proj_and_info)
    P_left = map(x -> x[1][1], proj_and_info)
    P_right = map(x -> x[1][2], proj_and_info)
    truncation_error = maximum(x -> x[2].truncation_error, proj_and_info)
    condition_number = maximum(x -> x[2].condition_number, proj_and_info)
    U = map(x -> x[2].U, proj_and_info)
    S = map(x -> x[2].S, proj_and_info)
    V = map(x -> x[2].V, proj_and_info)
    info = (; truncation_error, condition_number, U, S, V)
    return (P_left, P_right), info
end

"""
    simultaneous_projectors(enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::ProjectorAlgorithm)
    simultaneous_projectors(coordinate, enlarged_corners::Array{E,3}, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:simultaneous` scheme either for all provided
enlarged corners or on a specific `coordinate`.
"""
function simultaneous_projectors(
    enlarged_corners::Array{E,3}, env::CTMRGEnv, alg::ProjectorAlgorithm
) where {E}
    proj_and_info = dtmap(eachcoordinate(env, 1:4)) do coordinate
        coordinate′ = _next_coordinate(coordinate, size(env)[2:3]...)
        trscheme = truncation_scheme(alg, env.edges[coordinate[1], coordinate′[2:3]...])
        return simultaneous_projectors(
            coordinate, enlarged_corners, @set(alg.trscheme = trscheme)
        )
    end
    return _split_proj_and_info(proj_and_info)
end
function simultaneous_projectors(
    coordinate, enlarged_corners::Array{E,3}, alg::HalfInfiniteProjector
) where {E}
    coordinate′ = _next_coordinate(coordinate, size(enlarged_corners)[2:3]...)
    ec = (enlarged_corners[coordinate...], enlarged_corners[coordinate′...])
    return compute_projector(ec, coordinate, alg)
end
function simultaneous_projectors(
    coordinate, enlarged_corners::Array{E,3}, alg::FullInfiniteProjector
) where {E}
    rowsize, colsize = size(enlarged_corners)[2:3]
    coordinate2 = _next_coordinate(coordinate, rowsize, colsize)
    coordinate3 = _next_coordinate(coordinate2, rowsize, colsize)
    coordinate4 = _next_coordinate(coordinate3, rowsize, colsize)
    ec = (
        enlarged_corners[coordinate4...],
        enlarged_corners[coordinate...],
        enlarged_corners[coordinate2...],
        enlarged_corners[coordinate3...],
    )
    return compute_projector(ec, coordinate, alg)
end

"""
    renormalize_simultaneously(enlarged_corners, projectors, network, env)

Renormalize all enlarged corners and edges simultaneously.
"""
function renormalize_simultaneously(enlarged_corners, projectors, network, env)
    P_left, P_right = projectors
    coordinates = eachcoordinate(env, 1:4)
    corners_edges = dtmap(coordinates) do (dir, r, c)
        if dir == NORTH
            corner = renormalize_northwest_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_north_edge((r, c), env, P_left, P_right, network)
        elseif dir == EAST
            corner = renormalize_northeast_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_east_edge((r, c), env, P_left, P_right, network)
        elseif dir == SOUTH
            corner = renormalize_southeast_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_south_edge((r, c), env, P_left, P_right, network)
        elseif dir == WEST
            corner = renormalize_southwest_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_west_edge((r, c), env, P_left, P_right, network)
        end
        return corner / norm(corner), edge / norm(edge)
    end

    return CTMRGEnv(map(first, corners_edges), map(last, corners_edges))
end

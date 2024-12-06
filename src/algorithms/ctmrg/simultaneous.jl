"""
    simultaneous_ctmrg_iter(state, envs::CTMRGEnv, alg::CTMRG) -> envs′, info

Perform one simultaneous iteration of CTMRG, in which the environment is expanded and
renormalized in all directions at the same time.
"""
function simultaneous_ctmrg_iter(state, envs::CTMRGEnv, alg::CTMRG)
    enlarged_corners = dtmap(eachcoordinate(state, 1:4)) do idx
        return TensorMap(EnlargedCorner(state, envs, idx), idx[1])
    end  # expand environment
    projectors, info = simultaneous_projectors(enlarged_corners, envs, alg.projector_alg)  # compute projectors on all coordinates
    envs′ = renormalize_simultaneously(enlarged_corners, projectors, state, envs)  # renormalize enlarged corners
    return envs′, info
end

# Pre-allocate U, S, and V tensor as Zygote buffers to make it differentiable
function _prealloc_svd(edges::Array{E,N}, ::HalfInfiniteProjector) where {E,N}
    Sc = scalartype(E)
    U = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, space(e)), edges))
    V = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, domain(e), codomain(e)), edges))
    S = Zygote.Buffer(U.data, tensormaptype(spacetype(E), 1, 1, real(Sc)))  # Corner type but with real numbers
    return U, S, V
end
function _prealloc_svd(edges::Array{E,N}, ::FullInfiniteProjector) where {E,N}
    Sc = scalartype(E)
    Rspace(x) = spacetype(E)(dim(codomain(x)))
    U = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, Rspace(e), domain(e)), edges))
    V = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, domain(e), Rspace(e)), edges))
    S = Zygote.Buffer(U.data, tensormaptype(spacetype(E), 1, 1, real(Sc)))  # Corner type but with real numbers
    return U, S, V
end

function simultaneous_projectors(enlarged_corners, envs::CTMRGEnv, alg::ProjectorAlgs)
    U, S, V = _prealloc_svd(envs.edges, alg)
    ϵ = zero(real(scalartype(envs)))

    projectors = dtmap(eachcoordinate(envs, 1:4)) do coordinate
        proj, info = simultaneous_projector(enlarged_corners, coordinate, alg)
        U[coordinate...] = info.U
        S[coordinate...] = info.S
        V[coordinate...] = info.V
        ϵ = max(ϵ, info.err / norm(info.S))
        return proj
    end

    P_left = map(first, projectors)
    P_right = map(last, projectors)
    return (P_left, P_right), (; err=ϵ, U=copy(U), S=copy(S), V=copy(V))
end
function simultaneous_projector(
    enlarged_corners::Array{E,3}, coordinate, alg::HalfInfiniteProjector
) where {E}
    coordinate′ = _next_coordinate(coordinate, size(enlarged_corners)[2:3]...)
    ec = (enlarged_corners[coordinate...], enlarged_corners[coordinate′...])
    return compute_projector(ec, coordinate, alg)
end
function simultaneous_projector(
    enlarged_corners::Array{E,3}, coordinate, alg::FullInfiniteProjector
) where {E}
    rowsize, colsize = size(enlarged_corners)[2:3]
    coordinate2 = _next_coordinate(coordinate, rowsize, colsize)
    coordinate3 = _next_coordinate(coordinate2, rowsize, colsize)
    coordinate4 = _next_coordinate(coordinate3, rowsize, colsize)
    ec = (
        enlarged_corners[coordinate...],
        enlarged_corners[coordinate2...],
        enlarged_corners[coordinate3...],
        enlarged_corners[coordinate4...],
    )
    return compute_projector(ec, coordinate, alg)
end

function renormalize_simultaneously(enlarged_corners, projectors, state, envs)
    P_left, P_right = projectors
    coordinates = eachcoordinate(envs, 1:4)
    corners_edges = dtmap(coordinates) do (dir, r, c)
        if dir == NORTH
            corner = renormalize_northwest_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_north_edge((r, c), envs, P_left, P_right, state)
        elseif dir == EAST
            corner = renormalize_northeast_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_east_edge((r, c), envs, P_left, P_right, state)
        elseif dir == SOUTH
            corner = renormalize_southeast_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_south_edge((r, c), envs, P_left, P_right, state)
        elseif dir == WEST
            corner = renormalize_southwest_corner((r, c), enlarged_corners, P_left, P_right)
            edge = renormalize_west_edge((r, c), envs, P_left, P_right, state)
        end
        return corner / norm(corner), edge / norm(edge)
    end

    return CTMRGEnv(map(first, corners_edges), map(last, corners_edges))
end

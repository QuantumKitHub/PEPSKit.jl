"""
    SimultaneousCTMRG(; tol=Defaults.ctmrg_tol, maxiter=Defaults.ctmrg_maxiter,
                      miniter=Defaults.ctmrg_miniter, verbosity=0,
                      projector_alg=Defaults.projector_alg,
                      svd_alg=SVDAdjoint(), trscheme=FixedSpaceTruncation())

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
    verbosity=2,
    projector_alg=Defaults.projector_alg_type,
    svd_alg=Defaults.svd_alg,
    trscheme=Defaults.trscheme,
)
    return SimultaneousCTMRG(
        tol, maxiter, miniter, verbosity, projector_alg(; svd_alg, trscheme, verbosity)
    )
end

function ctmrg_iteration(state, envs::CTMRGEnv, alg::SimultaneousCTMRG)
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
    S = Zygote.Buffer(edges, tensormaptype(spacetype(E), 1, 1, real(Sc)))  # Corner type but with real numbers
    return U, S, V
end
function _prealloc_svd(edges::Array{E,N}, ::FullInfiniteProjector) where {E,N}
    Sc = scalartype(E)
    Rspace(x) = fuse(codomain(x))
    U = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, Rspace(e), domain(e)), edges))
    V = Zygote.Buffer(map(e -> TensorMap(zeros, Sc, domain(e), Rspace(e)), edges))
    S = Zygote.Buffer(edges, tensormaptype(spacetype(E), 1, 1, real(Sc)))  # Corner type but with real numbers
    return U, S, V
end

# Compute condition number σ_max / σ_min for diagonal singular value TensorMap
function _condition_number(S::AbstractTensorMap)
    S_diag = diag(S.data)
    return maximum(S_diag) / minimum(S_diag)
end
@non_differentiable _condition_number(S::AbstractTensorMap)

"""
    simultaneous_projectors(enlarged_corners::Array{E,3}, envs::CTMRGEnv, alg::ProjectorAlgorithm)
    simultaneous_projectors(coordinate, enlarged_corners::Array{E,3}, alg::ProjectorAlgorithm)

Compute CTMRG projectors in the `:simultaneous` scheme either for all provided
enlarged corners or on a specific `coordinate`.
"""
function simultaneous_projectors(
    enlarged_corners::Array{E,3}, envs::CTMRGEnv, alg::ProjectorAlgorithm
) where {E}
    U, S, V = _prealloc_svd(envs.edges, alg)
    ϵ = Zygote.Buffer(zeros(real(scalartype(envs)), size(envs)))

    projectors = dtmap(eachcoordinate(envs, 1:4)) do coordinate
        coordinate′ = _next_coordinate(coordinate, size(envs)[2:3]...)
        trscheme = truncation_scheme(alg, envs.edges[coordinate[1], coordinate′[2:3]...])
        proj, info = simultaneous_projectors(
            coordinate, enlarged_corners, @set(alg.trscheme = trscheme)
        )
        U[coordinate...] = info.U
        S[coordinate...] = info.S
        V[coordinate...] = info.V
        ϵ[coordinate...] = info.err / norm(info.S)
        return proj
    end

    P_left = map(first, projectors)
    P_right = map(last, projectors)
    S = copy(S)
    truncation_error = maximum(copy(ϵ))  # TODO: This makes Zygote error on first execution?
    condition_number = maximum(_condition_number, S)
    info = (; truncation_error, condition_number, U=copy(U), S, V=copy(V))
    return (P_left, P_right), info
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
    renormalize_simultaneously(enlarged_corners, projectors, state, envs)

Renormalize all enlarged corners and edges simultaneously.
"""
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

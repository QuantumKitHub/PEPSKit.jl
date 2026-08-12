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
    return CTMRGAlgorithm(; alg = :SequentialCTMRG, kwargs...)
end

CTMRG_SYMBOLS[:SequentialCTMRG] = SequentialCTMRG

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
    U, S, V = _initialize_decomposition_unit_cell(env)
    for _ in 1:4 # rotate
        for col in 1:size(network, 2) # left move column-wise
            env, info = ctmrg_leftmove(col, network, env, alg)
            ignore_derivatives() do
                truncation_error = max(truncation_error, info.truncation_error)
                U[WEST, :, col] .= info.U
                S[WEST, :, col] .= info.S
                V[WEST, :, col] .= info.V
            end
        end
        network = rotate_north(network, EAST)
        env = rotate_north(env, EAST)
        ignore_derivatives() do
            U, S, V = _rotl90_decomposition_unit_cell(U, S, V)
        end
    end
    return env, (; contraction_metrics = (; truncation_error), U, S, V)
end

# initialize empty unit cell array of SVD tensors from CTMRG environment
function _initialize_decomposition_unit_cell(env::CTMRGEnv)
    U = map(similar, env.edges)
    S = map(
        e -> DiagonalTensorMap(zeros(real(scalartype(env)), only(domain(e)), only(domain(e)))), env.edges
    )
    V = map(similar ∘ transpose, env.edges)
    return U, S, V
end

# adapted Base.rotl90(::CTMRGEnv) for SVD tensor arrays
function _rotl90_decomposition_unit_cell(U::Array{Ut}, S::Array{St}, V::Array{Vt}) where {Ut, St, Vt}
    U′ = Array{Ut, 3}(undef, 4, size(U, 3), size(U, 2))
    S′ = Array{St, 3}(undef, 4, size(S, 3), size(S, 2))
    V′ = Array{Vt, 3}(undef, 4, size(V, 3), size(V, 2))
    for dir in 1:4
        dir2 = _prev(dir, 4)
        U′[dir2, :, :] = rotl90(U[dir, :, :])
        S′[dir2, :, :] = rotl90(S[dir, :, :])
        V′[dir2, :, :] = rotl90(V[dir, :, :])
    end
    return U′, S′, V′
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
        proj, info = sequential_projectors((WEST, r, c), network, env, alg)
        return proj, info
    end
    return _split_proj_and_info(proj_and_info′)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::HalfInfiniteProjector
    )
    _, r, c = coordinate
    r′ = r - 1
    trunc = truncation_strategy(alg, edge(env, WEST, r′, c))
    alg´ = _set_decomposition_truncation(alg, trunc)
    Q1 = TensorMap(EnlargedCorner(network, env, (SOUTHWEST, r, c)))
    Q2 = TensorMap(EnlargedCorner(network, env, (NORTHWEST, r′, c)))
    return compute_projector((Q1, Q2), alg´)
end
function sequential_projectors(
        coordinate::NTuple{3, Int}, network, env::CTMRGEnv, alg::FullInfiniteProjector
    )
    rowsize, colsize = size(env)[2:3]
    coordinate_nw = _next_coordinate(coordinate, rowsize, colsize)
    coordinate_ne = _next_coordinate(coordinate_nw, rowsize, colsize)
    coordinate_se = _next_coordinate(coordinate_ne, rowsize, colsize)
    trunc = truncation_strategy(alg, env.edges[WEST, coordinate_nw[2:3]...])
    alg´ = _set_decomposition_truncation(alg, trunc)
    ec = (
        TensorMap(EnlargedCorner(network, env, coordinate_se)),
        TensorMap(EnlargedCorner(network, env, coordinate)),
        TensorMap(EnlargedCorner(network, env, coordinate_nw)),
        TensorMap(EnlargedCorner(network, env, coordinate_ne)),
    )
    return compute_projector(ec, alg´)
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

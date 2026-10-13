"""
    SplitCTMRG(; kwargs...)

Sequential split-CTMRG introduced in [Naumann et al.](@cite naumann_variationally_2025).
Uses [`SplitCTMRGEnv`](@ref), with independently truncated outer and interlayer environment bonds.
The common iteration, `trunc`, and `decomposition_alg` keywords follow [`SequentialCTMRG`](@ref).
`trunc` controls the green and teal projectors; `interlayer_trunc` controls the yellow and red projectors and defaults to `trunc`.
Uses [`SplitProjector`](@ref) to construct the split projector pairs.
The fixed-point gradient aligns the outer and interlayer gauges and holds both environment spaces fixed.
"""
struct SplitCTMRG{P <: SplitProjector, I <: TruncationStrategy} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
    interlayer_trunc::I
end
SplitCTMRG(; kwargs...) = CTMRGAlgorithm(; alg = :SplitCTMRG, kwargs...)::SplitCTMRG
CTMRG_SYMBOLS[:SplitCTMRG] = SplitCTMRG

function check_input(::typeof(leading_boundary), network, env::SplitCTMRGEnv, alg::CTMRGAlgorithm)
    alg isa SplitCTMRG || throw(ArgumentError("SplitCTMRGEnv requires SplitCTMRG."))
    eltype(network) <: PEPSSandwich || throw(ArgumentError("SplitCTMRG currently supports PEPS overlap networks only."))
    size(env)[2:3] == size(network) || throw(DimensionMismatch("Network and environment unit cells must match."))
    env.ket_first == (true, true, false, false) || throw(
        ArgumentError(
            "SplitCTMRG requires ket_first == (true, true, false, false); got $(env.ket_first). " *
                "For a rotated standard environment, apply the same rotl90, rotr90, or rot180 transformation " *
                "to both the environment and the PEPS state/network to restore this orientation before calling " *
                "`leading_boundary`, then rotate the returned environment back."
        )
    )
    alg.maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    return nothing
end
function check_input(::typeof(leading_boundary), network, env::CTMRGEnv, alg::SplitCTMRG)
    throw(ArgumentError("SplitCTMRG requires a SplitCTMRGEnv; convert the initial environment first."))
end

function convergence_tensors(env::SplitCTMRGEnv, alg)
    # The separate factors have an interlayer gauge freedom; compare normalized joined edges.
    edges = map(E -> E / norm(E), CTMRGEnv(env).edges)
    return env.corners, edges
end
network_value(network::InfiniteSquareNetwork, env::SplitCTMRGEnv) = network_value(network, CTMRGEnv(env))

"""Reflect and conjugate a PEPS overlap network to match the split-edge orientation."""
function _reflect_split_network(network::InfiniteSquareNetwork{<:PEPSSandwich}, diagonal::Bool)
    p = diagonal ? (4, 3, 2, 1) : (1, 4, 3, 2)
    tensors = dtmap(unitcell(network)) do A
        return map(t -> copy(permute(t', ((5,), p))), A)
    end
    tensors = diagonal ? permutedims(tensors) : reverse(tensors; dims = 2)
    return InfiniteSquareNetwork(tensors)
end

function ChainRulesCore.rrule(::typeof(_reflect_split_network), network::InfiniteSquareNetwork{<:PEPSSandwich}, diagonal::Bool)
    reflected = _reflect_split_network(network, diagonal)
    function reflection_pullback(Δ_)
        Δ = unthunk(Δ_)
        Δnetwork = Δ isa AbstractZero ? ZeroTangent() : _reflect_split_network(Δ, diagonal)
        return NoTangent(), Δnetwork, NoTangent()
    end
    return reflected, reflection_pullback
end

"""Orient a directional split move as a left move without refactorizing its edges."""
function _orient_split_move(network, env, dir::Int; inverse::Bool = false)
    if dir == WEST
        return network, env
    elseif dir == NORTH || dir == EAST
        diagonal = dir == NORTH
        return _reflect_split_network(network, diagonal), _reflect_split_env(env, diagonal)
    else
        rotate = inverse ? rotl90 : rotr90
        return rotate(network), rotate(env)
    end
end

function ctmrg_iteration(network::InfiniteSquareNetwork, env::SplitCTMRGEnv, alg::SplitCTMRG)
    truncation_error = zero(real(scalartype(network)))
    for dir in (WEST, NORTH, EAST, SOUTH)
        oriented_network, oriented_env = _orient_split_move(network, env, dir)
        for col in axes(oriented_network, 2)
            oriented_env, info = ctmrg_leftmove(col, oriented_network, oriented_env, alg)
            truncation_error = max(truncation_error, info.truncation_error)
        end
        _, env = _orient_split_move(oriented_network, oriented_env, dir; inverse = true)
    end
    return env, (; contraction_metrics = (; truncation_error))
end

function ctmrg_leftmove(col::Int, network, env::SplitCTMRGEnv, alg::SplitCTMRG)
    # Infer the contraction graphs for their concrete tensor types, independently of the sweep.
    projectors, info = Base.inferencebarrier(sequential_projectors)(col, network, env, alg.projector_alg, alg.interlayer_trunc)
    env′ = Base.inferencebarrier(renormalize_sequentially)(col, projectors, network, env)::typeof(env)
    return env′, (; truncation_error = real(scalartype(network))(info.truncation_error))
end

function renormalize_sequentially(col::Int, projectors, network, env::SplitCTMRGEnv)
    (; green, teal, yellow, red) = projectors
    updates = dtmap(axes(env, 2)) do r
        nextrow = _next(r, size(env, 2))
        NW = renormalize_northwest_corner(
            corner(env, NORTHWEST, r, col - 1), edge(env, NORTH, r, col),
            green[2][nextrow], teal[2][nextrow],
        )
        SW = renormalize_southwest_corner(
            corner(env, SOUTHWEST, r, col - 1), edge(env, SOUTH, r, col),
            green[1][r], teal[1][r],
        )
        K = renormalize_west_edge(
            edge(env, :ket, WEST, r, col - 1), green[1][r], teal[1][r],
            yellow[2][r], red[2][r], ket(network[r, col]),
        )
        B = renormalize_west_edge(
            edge(env, :bra, WEST, r, col - 1), green[2][nextrow], teal[2][nextrow],
            yellow[1][r], red[1][r], bra(network[r, col]), Val(:bra),
        )
        return (
            _normalize_split_tensor(NW, corner(env, NORTHWEST, r, col)),
            _normalize_split_tensor(SW, corner(env, SOUTHWEST, r, col)),
            _normalize_split_tensor(K, edge(env, :ket, WEST, r, col)),
            _normalize_split_tensor(B, edge(env, :bra, WEST, r, col)),
        )
    end
    corners = map(CartesianIndices(env.corners)) do I
        dir, r, c = Tuple(I)
        c == col && dir == NORTHWEST && return updates[r][1]
        c == col && dir == SOUTHWEST && return updates[r][2]
        return env.corners[I]
    end
    edges_ket = map(CartesianIndices(env.edges_ket)) do I
        dir, r, c = Tuple(I)
        return c == col && dir == WEST ? updates[r][3] : env.edges_ket[I]
    end
    edges_bra = map(CartesianIndices(env.edges_bra)) do I
        dir, r, c = Tuple(I)
        return c == col && dir == WEST ? updates[r][4] : env.edges_bra[I]
    end
    return SplitCTMRGEnv(corners, edges_ket, edges_bra, env.ket_first)
end

"""Normalize an updated tensor and align its global phase with the previous factor."""
function _normalize_split_tensor(t, previous)
    t = t / norm(t)
    space(t) == space(previous) || return t
    overlap = dot(t, previous)
    phase = iszero(overlap) ? one(overlap) : overlap / abs(overlap)
    return t * phase
end

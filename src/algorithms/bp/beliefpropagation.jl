"""
    struct BeliefPropagation

Algorithm for computing the belief propagation fixed point messages.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct BeliefPropagation
    "Stopping criterion for the BP iterations in relative trace norm difference"
    tol::Float64 = 1.0e-6

    "Minimal number of BP iterations"
    miniter::Int = 2

    "Maximal number of BP iterations"
    maxiter::Int = 50

    "Toggle for projecting messages onto the hermitian subspace immediately after update through BP equation"
    project_hermitian::Bool = true

    "When true, preserve bipartite structure of BPEnv inherited from input network"
    bipartite::Bool = false

    "Output verbosity level"
    verbosity::Int = 2
end

"""
    leading_boundary(env₀::BPEnv, network, alg::BeliefPropagation)

Contract `network` in the BP approximation and return the corresponding messages.
"""
function leading_boundary(env₀::BPEnv, network::InfiniteSquareNetwork, alg::BeliefPropagation)
    return LoggingExtras.withlevel(; alg.verbosity) do
        env = deepcopy(env₀)
        log = MPSKit.IterLog("BP")
        ϵ = Inf
        @infov 1 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            env′ = bp_iteration(network, env, alg)
            ϵ = oftype(ϵ, tr_distance(env, env′))
            env = env′

            if ϵ <= alg.tol && iter >= alg.miniter
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end

        return env, ϵ
    end
end
function leading_boundary(env₀::BPEnv, state::InfiniteState, alg::BeliefPropagation)
    if alg.bipartite
        @assert _state_bipartite_check(state)
    end
    return leading_boundary(env₀, InfiniteSquareNetwork(state), alg)
end

"""
One iteration to update the BP environment.
"""
function bp_iteration(network::InfiniteSquareNetwork, env::BPEnv, alg::BeliefPropagation)
    if alg.bipartite
        @assert size(network, 1) == size(network, 2) == 2
        # update BP env around 1st column of state
        # [N/S, 1:2, 1], [E/W, 1:2, 2]
        messages = map(Iterators.product(1:4, 1:2)) do (d, r)
            c = (d == NORTH || d == SOUTH) ? 1 : 2
            I = CartesianIndex(d, r, c)
            M = update_message(I, network, env)
            normalize!(M)
            alg.project_hermitian && (M = project_hermitian!!(M))
            return M
        end
        messages = map(Iterators.product(1:4, 1:2, 1:2)) do (d, r, c)
            r′ = _next(r, 2)
            if d == NORTH || d == SOUTH
                return (c == 1) ? messages[d, r] : copy(messages[d, r′])
            else
                return (c == 2) ? messages[d, r] : copy(messages[d, r′])
            end
        end
        return BPEnv(messages)
    else
        messages = map(eachindex(env)) do I
            M = update_message(I, network, env)
            normalize!(M)
            alg.project_hermitian && (M = project_hermitian!!(M))
            return M
        end
        return BPEnv(messages)
    end
end

"""
Update the BP message in `env.messages[I]`.
"""
function update_message(I::CartesianIndex{3}, network::InfiniteSquareNetwork, env::BPEnv)
    dir, row, col = Tuple(I)
    (1 <= dir <= 4) || throw(ArgumentError("Invalid direction $dir"))

    A = network[row, col]
    dir == SOUTH || (M_north = env[NORTH, _prev(row, end), col])
    dir == WEST || (M_east = env[EAST, row, _next(col, end)])
    dir == NORTH || (M_south = env[SOUTH, _next(row, end), col])
    dir == EAST || (M_west = env[WEST, row, _prev(col, end)])

    return if dir == NORTH
        contract_north_message(A, M_west, M_north, M_east)
    elseif dir == EAST
        contract_east_message(A, M_north, M_east, M_south)
    elseif dir == SOUTH
        contract_south_message(A, M_east, M_south, M_west)
    else # dir == WEST
        contract_west_message(A, M_south, M_west, M_north)
    end
end

function tr_distance(A::BPEnv, B::BPEnv)
    return sum(zip(A.messages, B.messages)) do (a, b)
        return trnorm(add(a, b, -inv(tr(b)), inv(tr(a))))
    end / length(A.messages)
end

function trnorm(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals(M), p, zero(real(scalartype(M))))
end
function trnorm!(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals!(M), p, zero(real(scalartype(M))))
end

project_hermitian!!(t) = add(t, t', 1 / 2, 1 / 2)

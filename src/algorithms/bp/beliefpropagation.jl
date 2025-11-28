@kwdef struct BeliefPropagation
    maxiter::Int = 50
    tol::Float64 = 1.0e-6
    verbosity::Int = 2
end

"""
Find the fixed point solution of the BP equations.
"""
function bp_fixedpoint(env::BPEnv, network::InfiniteSquareNetwork, alg::BeliefPropagation)
    log = MPSKit.IterLog("BP")
    ϵ = Inf

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 1 loginit!(log, ϵ)
        iter = 0
        while true
            iter += 1
            env′ = bp_iteration(network, env)
            ϵ = oftype(ϵ, tr_distance(env, env′))
            env = env′

            if ϵ < alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                return env, ϵ
            end
            if iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
                return env, ϵ
            end

            @infov 3 logiter!(log, iter, ϵ)
        end
    end
end

"""
One iteration to update the BP environment.
"""
function bp_iteration(network::InfiniteSquareNetwork, env::BPEnv)
    messages = similar(env.messages)
    for I in CartesianIndices(messages)
        messages[I] = normalize!(update_message(I, network, env))
    end
    return BPEnv(messages)
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
    new_message = if dir == NORTH
        contract_north_message(A, M_west, M_north, M_east)
    elseif dir == EAST
        contract_east_message(A, M_north, M_east, M_south)
    elseif dir == SOUTH
        contract_south_message(A, M_east, M_south, M_west)
    else # dir == WEST
        contract_west_message(A, M_south, M_west, M_north)
    end
    @assert space(new_message) == space(env[I])
    # TODO: enforce hermiticity (avoid accumulation of numerical errors)
    if env[I] ≈ env[I]'
        # ensure a single iteration preserves hermiticity
        @assert new_message ≈ new_message'
        new_message = (new_message + new_message') / 2
    end
    return new_message
end

function tr_distance(A::BPEnv, B::BPEnv)
    return sum(zip(A.messages, B.messages)) do (a, b)
        return trnorm(add(a, b, -inv(tr(b)), inv(tr(a))))
    end
end

function trnorm(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals(M), p, zero(real(scalartype(M))))
end
function trnorm!(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals!(M), p, zero(real(scalartype(M))))
end

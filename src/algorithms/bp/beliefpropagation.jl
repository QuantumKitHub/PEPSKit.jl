@kwdef struct BeliefPropagation
    maxiter::Int = 10
    tol::Float64 = 1e-6
    verbosity::Int = 2
end

function bp_fixedpoint(env::BPEnv, network::InfiniteSquareNetwork, alg::BeliefPropagation)
    log = MPSKit.IterLog("BP")
    ϵ = Inf

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 1 loginit!(log, ϵ)
        iter = 0
        while true
            iter += 1
            env′ = bp_iteration(network, env, alg)
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

function bp_iteration(network::InfiniteSquareNetwork, env::BPEnv, alg::BeliefPropagation)
    messages = similar(env.messages)
    for I in eachindex(IndexCartesian(), messages)
        dir, row, col = Tuple(I)
        if dir == NORTH
            row += 1
        elseif dir == EAST
            col += 1
        elseif dir == SOUTH
            row -= 1
        elseif dir == WEST
            col -= 1
        end
        messages[dir, mod1(row, end), mod1(col, end)] = normalize!(
            update_message(I, network, env)
        )
    end
    return BPEnv(messages)
end

function update_message(I::CartesianIndex{3}, network::InfiniteSquareNetwork, env::BPEnv)
    dir, row, col = Tuple(I)

    A = network[row, col]
    dir == SOUTH || (M_north = env.messages[NORTH, _prev(row, end), col])
    dir == WEST || (M_east = env.messages[EAST, row, _next(col, end)])
    dir == NORTH || (M_south = env.messages[SOUTH, _next(row, end), col])
    dir == EAST || (M_west = env.messages[WEST, row, _prev(col, end)])

    return if dir == NORTH
        contract_north_message(A, M_west, M_north, M_east)
    elseif dir == EAST
        contract_east_message(A, M_north, M_east, M_south)
    elseif dir == SOUTH
        contract_south_message(A, M_east, M_south, M_west)
    elseif dir == WEST
        contract_west_message(A, M_south, M_west, M_north)
    else
        throw(ArgumentError("Invalid direction $dir"))
    end
end

function tr_distance(A::BPEnv, B::BPEnv)
    return sum(zip(A.messages, B.messages)) do (a, b)
        return trnorm(add(a, b, -inv(tr(b)), inv(tr(a))))
    end
end

function trnorm(M::AbstractTensorMap, p::Real=1)
    return TensorKit._norm(svdvals(M), p, zero(real(scalartype(M))))
end
function trnorm!(M::AbstractTensorMap, p::Real=1)
    return TensorKit._norm(svdvals!(M), p, zero(real(scalartype(M))))
end

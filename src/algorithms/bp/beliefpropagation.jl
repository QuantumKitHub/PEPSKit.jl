@kwdef struct BeliefPropagation
    maxiter::Int = 10
    tol::Float64 = 1.0e-6
    verbosity::Int = 2
end

function gauge_fix(psi::InfinitePEPS, alg::BeliefPropagation, env::BPEnv = BPEnv(psi))
    # Compute belief propagation fixed point solutions
    env, err = bp_fixedpoint(env, InfiniteSquareNetwork(psi), alg)

    # Bring PEPS to the Vidal gauge
    sqrtmsgs = map(env.messages) do M
        U, S, Vᴴ = tsvd!(M)
        sqrtM = U * sdiag_pow(S, 1 / 2) * Vᴴ
        isqrtM = U * sdiag_pow(S, -1 / 2) * Vᴴ
        return sqrtM, isqrtM
    end
    bond_svds = map(eachcoordinate(psi, 1:2)) do (dir, r, c)
        # TODO: would be more reasonable to define SOUTH as adjoint(NORTH)...
        MM = sqrtmsgs[dir, r, c][1] * transpose(sqrtmsgs[mod1(dir + 2, 4), r, c][1])
        U, S, Vᴴ = tsvd!(MM)
        return U, S, Vᴴ
    end
    vertices = map(eachcoordinate(psi)) do (r, c)
        isqrtM_north = sqrtmsgs[NORTH, _prev(r, end), c][2]
        isqrtM_south = sqrtmsgs[SOUTH, _next(r, end), c][2]
        isqrtM_east = sqrtmsgs[EAST, r, _next(c, end)][2]
        isqrtM_west = sqrtmsgs[WEST, r, _prev(c, end)][2]

        U_north = bond_svds[NORTH, _prev(r, end), c][1]
        U_east = bond_svds[EAST, r, _next(c, end)][1]
        Vᴴ_south = bond_svds[NORTH, _next(r, end), c][3]
        Vᴴ_west = bond_svds[EAST, r, _prev(c, end)][3]

        @tensor contractcheck = true begin
            A[d; DN DE DS DW] ≔
                psi[r, c][d; DN1 DE1 DS1 DW1] *
                (isqrtM_north[DN1; DN2] * U_north[DN2; DN]) *
                (isqrtM_east[DE1; DE2] * U_east[DE2; DE]) *
                (isqrtM_south[DS1; DS2] * Vᴴ_south[DS; DS2]) *
                (isqrtM_west[DW1; DW2] * Vᴴ_west[DW; DW2])
        end
        return A
    end
    # TODO: decide on a convention here, possibly altering InfiniteWeightPEPS
    weight_mats = SUWeight(
        map(eachcoordinate(psi, 1:2)) do (dir, r, c)
            if dir == 1 # horizontal direction
                return bond_svds[EAST, r, _next(c, end)][2]
            else # vertical direction
                return bond_svds[NORTH, _prev(r, end), c][2]
            end
        end,
    )
    return InfiniteWeightPEPS(vertices, weight_mats)
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

function trnorm(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals(M), p, zero(real(scalartype(M))))
end
function trnorm!(M::AbstractTensorMap, p::Real = 1)
    return TensorKit._norm(svdvals!(M), p, zero(real(scalartype(M))))
end

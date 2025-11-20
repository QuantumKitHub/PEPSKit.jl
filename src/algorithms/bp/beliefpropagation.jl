@kwdef struct BeliefPropagation
    maxiter::Int = 50
    tol::Float64 = 1.0e-6
    verbosity::Int = 2
end

function gauge_fix(psi::InfinitePEPS, alg::BeliefPropagation, env::BPEnv = BPEnv(psi))
    # Compute belief propagation fixed point solutions
    env, err = bp_fixedpoint(env, InfiniteSquareNetwork(psi), alg)
    psi′, wts = _gauge_fix_bp(psi, env)
    return psi′, wts, env
end

"""
Use BP environment `env` to fix gauge of InfinitePEPS `psi`.
"""
function _gauge_fix_bp(psi::InfinitePEPS, env::BPEnv)
    # Bring PEPS to the Vidal gauge
    sqrtmsgs = map(env.messages) do M
        # U = V for positive semi-definite message M
        U, S, Vᴴ = svd_compact!(M)
        sqrtM = U * sdiag_pow(S, 1 / 2) * Vᴴ
        isqrtM = U * sdiag_pow(S, -1 / 2) * Vᴴ
        return sqrtM, isqrtM
    end
    bond_svds = map(eachcoordinate(psi, 1:2)) do (dir, r, c)
        # TODO: would be more reasonable to define SOUTH as adjoint(NORTH)...
        # TODO: figure out twists for fermion
        #= 
        - dir = 1: x-bond (r,c) → (r,c+1)
            m[(r,c) → (r,c+1)] = env[4, r, c]
            m[(r,c+1) → (r,c)] = env[2, r, c+1]
        - dir = 2: y-bond (r,c) → (r-1,c)
            m[(r,c) → (r-1,c)] = env[3, r, c]
            m[(r-1,c) → (r,c)] = env[1, r-1, c]
        =#
        MM = if dir == 1
            transpose(sqrtmsgs[WEST, r, c][1]) * sqrtmsgs[EAST, r, _next(c, end)][1]
        else
            transpose(sqrtmsgs[SOUTH, r, c][1]) * sqrtmsgs[NORTH, _prev(r, end), c][1]
        end
        # TODO: preserve bond arrow direction (e.g. with flip_svd?)
        U, S, Vᴴ = svd_compact!(MM)
        return U, S, Vᴴ
    end
    ## bond weights Λ
    wts = SUWeight(
        map(eachcoordinate(psi, 1:2)) do (dir, r, c)
            return bond_svds[dir, r, c][2]
        end
    )
    ## gauge-fixed state
    psi′ = map(eachcoordinate(psi)) do (r, c)
        isqrtM_north = transpose(sqrtmsgs[SOUTH, r, c][2])
        isqrtM_south = transpose(sqrtmsgs[NORTH, r, c][2])
        isqrtM_east = transpose(sqrtmsgs[WEST, r, c][2])
        isqrtM_west = transpose(sqrtmsgs[EAST, r, c][2])

        U_north = bond_svds[2, r, c][1]
        U_east = bond_svds[1, r, c][1]
        Vᴴ_south = bond_svds[2, _next(r, end), c][3]
        Vᴴ_west = bond_svds[1, r, _prev(c, end)][3]
        # Vertex Γ tensors in Vidal gauge
        @tensor contractcheck = true begin
            Γ[d; DN DE DS DW] ≔
                psi[r, c][d; DN1 DE1 DS1 DW1] *
                (isqrtM_north[DN1; DN2] * U_north[DN2; DN]) *
                (isqrtM_east[DE1; DE2] * U_east[DE2; DE]) *
                (isqrtM_south[DS1; DS2] * Vᴴ_south[DS; DS2]) *
                (isqrtM_west[DW1; DW2] * Vᴴ_west[DW; DW2])
        end
        return absorb_weight(Γ, wts, r, c, Tuple(1:4))
    end
    return InfinitePEPS(psi′), wts
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

"""
One iteration to update the BP environment.
"""
function bp_iteration(network::InfiniteSquareNetwork, env::BPEnv, alg::BeliefPropagation)
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

"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SimpleUpdate <: TimeEvolution
    "Truncation strategy for bonds updated by Trotter gates"
    trunc::TruncationStrategy
    "When true (or false), the Trotter gate is `exp(-H dt)` (or `exp(-iH dt)`)"
    imaginary_time::Bool = true
    "When true, force the usage of MPO simple update for nearest neighbor Hamiltonians."
    force_mpo::Bool = false
    "When true, assume bipartite unit cell structure"
    bipartite::Bool = false
    "(Only applicable to InfinitePEPO) 
    When true, the PEPO is regarded as a purified PEPS, and updated as
    `|ρ(t + dt)⟩ = exp(-H dt/2) |ρ(t)⟩`.
    When false, the PEPO is updated as 
    `ρ(t + dt) = exp(-H dt/2) ρ(t) exp(-H dt/2)`."
    purified::Bool = true
end

"""
Internal state of simple update algorithm
"""
struct SUState{S <: InfiniteState, E <: SUWeight, N <: Number}
    "number of performed iterations"
    iter::Int
    "evolved time"
    t::N
    "PEPS/PEPO"
    psi::S
    "SUWeight environment"
    env::E
end

"""
    TimeEvolver(
        psi0::Union{InfinitePEPS, InfinitePEPO}, H::LocalOperator, dt::Number,
        nstep::Int, alg::SimpleUpdate, env0::SUWeight; t0::Number = 0.0,
        symmetrize_gates::Bool = false
    )

Initialize a `TimeEvolver` with Hamiltonian `H` and simple update `alg`, 
starting from the initial state `psi0` and `SUWeight` environment `env0`.

- The initial time is specified by `t0`.
- Use `symmetrize_gates = true` for second-order Trotter decomposition.
"""
function TimeEvolver(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::SimpleUpdate, env0::SUWeight; t0::Number = 0.0,
        symmetrize_gates::Bool = false
    )
    # sanity checks
    _timeevol_sanity_check(psi0, physicalspace(H), alg)
    dt′ = _get_dt(psi0, dt, alg.imaginary_time)
    # create Trotter gates
    gate = trotterize(H, dt′; symmetrize_gates, force_mpo = alg.force_mpo)
    state = SUState(0, t0, psi0, env0)
    # TODO: check gates for bipartite case
    return TimeEvolver(alg, dt, nstep, gate, state)
end

"""
Optimized simple update of nearest neighbor bonds utilizing
reduced bond tensors without decomposing the gate into a 2-site MPO.

When `purified = true`, `gate` acts on the codomain physical legs of `state`.
Otherwise, `gate` acts on both the codomain and the domain physical legs of `state`.
"""
function _su_nnbond!(
        state::InfiniteState, gate::Union{NNGate, Nothing}, env::SUWeight,
        dir::Int, row::Int, col::Int, trunc::TruncationStrategy;
        purified::Bool = true
    )
    Nr, Nc, = size(state)
    @assert dir == 1 || dir == 2
    # position of bond tensors
    siteA = CartesianIndex(row, col)
    siteB = CartesianIndex((dir == 1) ? (row, _next(col, Nc)) : (_prev(row, Nr), col))
    A, B = state.A[siteA], state.A[siteB]
    # absorb environment weights
    openaxsA = (dir == 1) ? (NORTH, SOUTH, WEST) : (EAST, SOUTH, WEST)
    openaxsB = (dir == 1) ? (NORTH, SOUTH, EAST) : (NORTH, EAST, WEST)
    A = absorb_weight(A, env, siteA[1], siteA[2], openaxsA; inv = false)
    B = absorb_weight(B, env, siteB[1], siteB[2], openaxsB; inv = false)
    normalize!(A, Inf)
    normalize!(B, Inf)
    # apply gate
    ϵ, s = 0.0, nothing
    if dir == 2
        A, B = rotr90(A), rotr90(B)
    end
    gate_axs = purified ? (1:1) : (1:2)
    for gate_ax in gate_axs
        X, a, b, Y = _qr_bond(A, B; gate_ax)
        a, s, b, ϵ′ = _apply_gate(a, b, gate, trunc)
        ϵ = max(ϵ, ϵ′)
        A, B = _qr_bond_undo(X, a, b, Y)
    end
    if dir == 2
        A, B = rotl90(A), rotl90(B)
    end
    # remove environment weights
    A = absorb_weight(A, env, siteA[1], siteA[2], openaxsA; inv = true)
    B = absorb_weight(B, env, siteB[1], siteB[2], openaxsB; inv = true)
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    # update tensor dict and weight on current bond
    state.A[siteA], state.A[siteB] = A, B
    env.data[dir, row, col] = s
    return ϵ
end

"""
One iteration of simple update
"""
function su_iter(
        state::InfiniteState, gates::TrotterGates,
        alg::SimpleUpdate, env::SUWeight
    )
    Nr, Nc, = size(state)
    state2, env2, ϵ = deepcopy(state), deepcopy(env), 0.0
    purified = alg.purified
    for (sites, gs) in gates.data
        if length(sites) == 1
            # 1-site gate
            # TODO: special treatment for bipartite state
            site = sites[1]
            r, c = mod1(site[1], Nr), mod1(site[2], Nc)
            state2.A[r, c] = _apply_sitegate(state2.A[r, c], gs; purified)
        elseif length(sites) == 2 && (isa(gs, NNGate) || gs === nothing)
            # 2-site gate not decomposed to MPO
            site1, site2 = sites
            r, c = mod1(site1[1], Nr), mod1(site1[2], Nc)
            (alg.bipartite && r > 1) && continue
            d = if site1 - site2 == CartesianIndex(0, -1)
                1 # x-bonds (leftwards)
            elseif site1 - site2 == CartesianIndex(1, 0)
                2 # y-bonds (downwards)
            else
                error("Non-standard direction of NN bonds for 2-site Trotter gates.")
            end
            trunc = truncation_strategy(alg.trunc, d, r, c)
            ϵ′ = _su_nnbond!(state2, gs, env2, d, r, c, trunc; purified)
            ϵ = max(ϵ, ϵ′)
            (!alg.bipartite) && continue
            if d == 1
                rp1, cp1 = _next(r, Nr), _next(c, Nc)
                state2.A[rp1, cp1] = deepcopy(state2.A[r, c])
                state2.A[rp1, c] = deepcopy(state2.A[r, cp1])
                env2.data[1, rp1, cp1] = deepcopy(env2.data[1, r, c])
            else
                rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
                state2.A[rm1, cm1] = deepcopy(state2.A[r, c])
                state2.A[r, cm1] = deepcopy(state2.A[rm1, c])
                env2.data[2, rm1, cm1] = deepcopy(env2.data[2, r, c])
            end
        else
            # N-site MPO gate (N ≥ 2)
            alg.bipartite && error("Multi-site MPO gates are not compatible with bipartite states.")
            truncs = _get_cluster_trunc(alg.trunc, sites, size(state)[1:2])
            ϵ′ = _su_cluster!(state2, gs, env2, sites, truncs; purified)
            ϵ = max(ϵ, ϵ′)
        end
    end
    return state2, env2, ϵ
end

function Base.iterate(it::TimeEvolver{<:SimpleUpdate}, state = it.state)
    iter, t = state.iter, state.t
    (iter == it.nstep) && return nothing
    psi, env, ϵ = su_iter(state.psi, it.gate, it.alg, state.env)
    # update internal state
    iter += 1
    t += it.dt
    it.state = SUState(iter, t, psi, env)
    info = (; t, ϵ)
    return (psi, env, info), it.state
end

"""
    timestep(
        it::TimeEvolver{<:SimpleUpdate}, psi::InfiniteState, env::SUWeight
    ) -> (psi, env, info)

Given the `TimeEvolver` iterator `it`, perform one step of time evolution
on the input state `psi` and its environment `env`.
"""
function MPSKit.timestep(
        it::TimeEvolver{<:SimpleUpdate}, psi::InfiniteState, env::SUWeight
    )
    _timeevol_sanity_check(psi, physicalspace(it.state.psi), it.alg)
    state = SUState(it.state.iter, it.state.t, psi, env)
    result = iterate(it, state)
    if result === nothing
        @warn "TimeEvolver `it` has already reached the end."
        return nothing
    else
        return first(result)
    end
end

"""
    time_evolve(
        it::TimeEvolver{<:SimpleUpdate}; 
        tol::Float64 = 0.0, check_interval::Int = 500
    ) -> (psi, env, info)

Perform time evolution to the end of `TimeEvolver` iterator `it`,
or until convergence of `SUWeight` set by a positive `tol`.

- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only).
    For other usages it should not be changed.
- `check_interval` sets the number of iterations between outputs of information.
"""
function MPSKit.time_evolve(
        it::TimeEvolver{<:SimpleUpdate};
        tol::Float64 = 0.0, check_interval::Int = 500
    )
    time_start = time()
    check_convergence = (tol > 0)
    @info "--- Time evolution (simple update), dt = $(it.dt) ---"
    if check_convergence
        @assert (it.state.psi isa InfinitePEPS) && it.alg.imaginary_time "Only imaginary time evolution of InfinitePEPS allows convergence checking."
    end
    env0, time0 = it.state.env, time()
    for (psi, env, info) in it
        iter = it.state.iter
        diff = compare_weights(env0, env)
        stop = (iter == it.nstep) || (diff < tol)
        showinfo = (check_interval > 0) &&
            ((iter % check_interval == 0) || (iter == 1) || stop)
        time1 = time()
        if showinfo
            @info "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
            @info @sprintf("SU iter %-7d: |Δλ| = %.3e. Time = %.3f s/it", iter, diff, time1 - time0)
        end
        if check_convergence
            if (iter == it.nstep) && (diff >= tol)
                @warn "SU: bond weights have not converged."
            end
            if diff < tol
                @info "SU: bond weights have converged."
            end
        end
        if stop
            time_end = time()
            @info @sprintf("Time evolution finished in %.2f s", time_end - time_start)
            return psi, env, info
        else
            env0 = env
        end
        time0 = time()
    end
    return
end

"""
    time_evolve(
        psi0::Union{InfinitePEPS, InfinitePEPO}, H::LocalOperator, dt::Number, nstep::Int,
        alg::SimpleUpdate, env0::SUWeight; symmetrize_gates::Bool = false,
        tol::Float64 = 0.0, t0::Number = 0.0, check_interval::Int = 500
    ) -> (psi, env, info)

Perform time evolution on the initial iPEPS or iPEPO `psi0` and
initial environment `env0` with Hamiltonian `H`, using `SimpleUpdate`
algorithm `alg`, time step `dt` for `nstep` number of steps. 

- Set `symmetrize_gates = true` for second-order Trotter decomposition.
- Set `tol > 0` to enable convergence check (for imaginary time evolution of iPEPS only).
    For other usages it should not be changed.
- Use `t0` to specify the initial time of the evolution.
- `check_interval` sets the interval to output information. Output during the evolution can be turned off by setting `check_interval <= 0`.
- `info` is a NamedTuple containing information of the evolution, 
    including the time `info.t` evolved since `psi0`.
"""
function MPSKit.time_evolve(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::SimpleUpdate, env0::SUWeight; symmetrize_gates::Bool = false,
        tol::Float64 = 0.0, t0::Number = 0.0, check_interval::Int = 500
    )
    it = TimeEvolver(psi0, H, dt, nstep, alg, env0; t0, symmetrize_gates)
    return time_evolve(it; tol, check_interval)
end

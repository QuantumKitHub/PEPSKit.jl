"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SimpleUpdate{T <: TruncationStrategy} <: TimeEvolution
    "Truncation strategy for bonds updated by Trotter gates"
    trunc::T
    "When true (or false), the Trotter gate is `exp(-H dt)` (or `exp(-iH dt)`)"
    imaginary_time::Bool = true
    "When true, force decomposition of nearest neighbor gates to MPOs."
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
    circuit = trotterize(H, dt′; symmetrize_gates, force_mpo = alg.force_mpo)
    state = SUState(0, t0, psi0, env0)
    # convert FixedSpaceTruncation to site-dependent `truncspace`s
    if alg.trunc isa FixedSpaceTruncation
        trunc = _get_fixedspacetrunc(psi0)
        @reset alg.trunc = trunc
    end
    # TODO: bipartite check for alg.trunc after equality is defined for all kinds of truncation strategies
    # TODO: check gates for bipartite case
    return TimeEvolver(alg, dt, nstep, circuit, state)
end


"""
Simple update optimized for nearest neighbor gates
utilizing reduced bond tensors with the physical leg.
"""
function _su_iter!(
        state::InfiniteState, gate::NNGate, env::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::SimpleUpdate
    )
    @assert length(sites) == 2
    trunc = only(_get_cluster_trunc(alg.trunc, sites))
    Ms, open_vaxs, = _get_cluster(state, sites)
    _absorb_weight!(Ms, sites, open_vaxs, env)
    # rotate
    (dir, r, c) = _nn_bonddir(sites...)
    A, B = _bond_rotation.(Ms, dir, EAST)
    # apply gate
    ϵ = 0.0
    local s
    for gate_ax in 1:2
        a, X = bond_tensor_first(A; gate_ax, positive = true)
        b, Y = bond_tensor_last(B; gate_ax, positive = true)
        a, s, b, ϵ′ = _apply_gate(a, b, gate, trunc)
        ϵ = max(ϵ, ϵ′)
        A = undo_bond_tensor_first(a, X; gate_ax)
        B = undo_bond_tensor_last(b, Y; gate_ax)
        alg.purified && break # only apply gate to 1st physical leg
    end
    # rotate back
    A = _bond_rotation(A, EAST, dir)
    B = _bond_rotation(B, EAST, dir)
    (dir in (WEST, SOUTH)) && (s = transpose(s))
    # remove environment weights
    siteA, siteB = sites
    A = absorb_weight(A, env, siteA[1], siteA[2], open_vaxs[1]; inv = true)
    B = absorb_weight(B, env, siteB[1], siteB[2], open_vaxs[2]; inv = true)
    # update tensor dict and weight on current bond
    state[siteA] = normalize!(A, Inf)
    state[siteB] = normalize!(B, Inf)
    d = dir in (EAST, WEST) ? 1 : 2
    env[d, r, c] = normalize!(s, Inf)
    return ϵ
end

"""
One iteration of simple update
"""
function su_iter(
        state::InfiniteState, circuit::LocalCircuit,
        alg::SimpleUpdate, env::SUWeight
    )
    state2, env2, ϵ = deepcopy(state), deepcopy(env), 0.0
    for (sites, gate) in circuit.gates
        if length(sites) == 1
            # 1-site gate
            # TODO: special treatment for bipartite state
            site = only(sites)
            state2[site] = _apply_sitegate(state2[site], gate; alg.purified)
        elseif length(sites) == 2
            (dir, r, c) = _nn_bonddir(sites...)
            d = dir in (EAST, WEST) ? 1 : 2
            alg.bipartite && iseven(r) && continue
            ϵ′ = _su_iter!(state2, gate, env2, sites, alg)
            ϵ = max(ϵ, ϵ′)
            (!alg.bipartite) && continue
            if d == 1
                state2[r + 1, c + 1] = copy(state2[r, c])
                state2[r + 1, c] = copy(state2[r, c + 1])
                env2[1, r + 1, c + 1] = copy(env2[1, r, c])
            else
                state2[r - 1, c - 1] = copy(state2[r, c])
                state2[r, c - 1] = copy(state2[r - 1, c])
                env2[2, r - 1, c - 1] = copy(env2[2, r, c])
            end
        else
            # N-site MPO gate (N ≥ 2)
            alg.bipartite && error("Multi-site MPO gates are not compatible with bipartite states.")
            ϵ′ = _su_iter!(state2, gate, env2, sites, alg)
            ϵ = max(ϵ, ϵ′)
        end
    end
    return state2, env2, ϵ
end

function Base.iterate(it::TimeEvolver{<:SimpleUpdate}, state = it.state)
    iter, t = state.iter, state.t
    (iter == it.nstep) && return nothing
    psi, env, ϵ = su_iter(state.psi, it.circuit, it.alg, state.env)
    # update internal state
    iter += 1
    t += it.dt
    it.state = SUState(iter, t, psi, env)
    info = (; t, ϵ)
    return (psi, env, info), it.state
end

"""
    timestep(
        it::TimeEvolver{<:SimpleUpdate}, psi::InfiniteState, env::SUWeight;
        iter::Int = it.state.iter, t::Number = it.state.t
    ) -> (psi, env, info)

Given the `TimeEvolver` iterator `it`, perform one step of time evolution
on the input state `psi` and its environment `env`.

- Using `iter` and `t` to reset the current iteration number and evolved time
    respectively of the TimeEvolver `it`.
"""
function MPSKit.timestep(
        it::TimeEvolver{<:SimpleUpdate}, psi::InfiniteState, env::SUWeight;
        iter::Int = it.state.iter, t::Number = it.state.t
    )
    _timeevol_sanity_check(psi, physicalspace(it.state.psi), it.alg)
    state = SUState(iter, t, psi, env)
    result = iterate(it, state)
    if result === nothing
        @warn "TimeEvolver `it` has already reached the end."
        return nothing
    else
        return first(result)
    end
end

"""
    time_evolve(it; verbosity = 2, check_interval = 500) -> (psi, env, info)
    time_evolve(it, H; tol = 1.0e-8, verbosity = 2, check_interval = 500) -> (psi, env, info)

Perform time evolution to the end of `TimeEvolver` iterator `it`,
or until convergence of `SUWeight` set by a positive `tol`.

- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only).
    For other usages it should not be changed.
- `verbosity` sets the verbosity level to output information.
    - 0: output no information except warnings.
    - 1: indicate the start and the end of the evolution.
    - 2: (default) output detailed progress of the evolution.
- `check_interval` sets the number of iterations to output evolution progress.
"""
function MPSKit.time_evolve(
        it::TimeEvolver{<:SimpleUpdate};
        verbosity::Int = 2, check_interval::Int = 500
    )
    @assert check_interval >= 0
    return LoggingExtras.withlevel(; verbosity) do
        time_start = time()
        @infov 1 "--- Time evolution (simple update), dt = $(it.dt) ---"
        env0, time0 = it.state.env, time()
        for (psi, env, info) in it
            iter = it.state.iter
            diff = compare_weights(env0, env)
            stop = (iter == it.nstep)
            showinfo = (check_interval > 0) &&
                ((iter % check_interval == 0) || (iter == 1) || stop)
            time1 = time()
            if showinfo
                @infov 2 "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
                @infov 2 @sprintf("SU iter %-7d: |Δλ| = %.3e. Time = %.3f s/it", iter, diff, time1 - time0)
            end
            if stop
                time_end = time()
                @infov 1 @sprintf("Time evolution finished in %.2f s", time_end - time_start)
                return psi, env, info
            else
                env0 = env
            end
            time0 = time()
        end
    end
end

function MPSKit.time_evolve(
        it::TimeEvolver{<:SimpleUpdate, G, S}, H::LocalOperator;
        tol::Float64 = 1.0e-8, verbosity::Int = 2, check_interval::Int = 500
    ) where {G, S <: SUState{<:InfinitePEPS}}
    @assert check_interval >= 0
    return LoggingExtras.withlevel(; verbosity) do
        time_start = time()
        @infov 1 "--- Time evolution (simple update), dt = $(it.dt) ---"
        @assert it.alg.imaginary_time "Only imaginary time evolution of InfinitePEPS allows convergence checking."
        env0, time0 = it.state.env, time()
        for (psi, env, info) in it
            iter = it.state.iter
            diff = compare_weights(env0, env)
            stop = (iter == it.nstep) || (diff < tol)
            showinfo = (check_interval > 0) &&
                ((iter % check_interval == 0) || (iter == 1) || stop)
            time1 = time()
            if showinfo
                # TODO: convert to BPEnv instead
                ctmenv = CTMRGEnv(env)
                energy = real(expectation_value(psi, H, ctmenv)) / prod(size(psi))
                @infov 2 "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
                @infov 2 @sprintf(
                    "SU iter %-7d: E ≈ %.5f, |Δλ| = %.3e. Time = %.3f s/it",
                    iter, energy, diff, time1 - time0
                )
            end
            if (iter == it.nstep) && (diff >= tol)
                @warn "SU: bond weights have not converged."
            end
            if diff < tol
                @infov 2 "SU: bond weights have converged."
            end
            if stop
                time_end = time()
                @infov 1 @sprintf("Time evolution finished in %.2f s", time_end - time_start)
                return psi, env, info
            else
                env0 = env
            end
            time0 = time()
        end
    end
end

"""
    time_evolve(
        psi0::Union{InfinitePEPS, InfinitePEPO}, H::LocalOperator,
        dt::Number, nstep::Int, alg::SimpleUpdate, env0::SUWeight;
        symmetrize_gates::Bool = false, tol::Float64 = 0.0,
        t0::Number = 0.0, verbosity::Int = 2, check_interval::Int = 500,
    ) -> (psi, env, info)

Perform time evolution on the initial iPEPS or iPEPO `psi0` and
initial environment `env0` with Hamiltonian `H`, using `SimpleUpdate`
algorithm `alg`, time step `dt` for `nstep` number of steps.

- Set `symmetrize_gates = true` for second-order Trotter decomposition.
- Set `tol > 0` to enable convergence check (for imaginary time evolution of iPEPS only).
- Use `t0` to specify the initial time of the evolution.
- `verbosity` sets the verbosity level to output information.
- `check_interval` sets the interval to output evolution progress.
- `info` is a NamedTuple containing information of the evolution,
    including the time `info.t` evolved since `psi0`.
"""
function MPSKit.time_evolve(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::SimpleUpdate, env0::SUWeight; symmetrize_gates::Bool = false,
        tol::Float64 = 0.0, t0::Number = 0.0,
        verbosity::Int = 2, check_interval::Int = 500,
    )
    it = TimeEvolver(psi0, H, dt, nstep, alg, env0; t0, symmetrize_gates)
    return if tol == 0
        time_evolve(it; verbosity, check_interval)
    else
        time_evolve(it, H; tol, verbosity, check_interval)
    end
end

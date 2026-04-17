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
    gate = trotterize(H, dt′; symmetrize_gates, force_mpo = alg.force_mpo)
    state = SUState(0, t0, psi0, env0)
    # TODO: check gates for bipartite case
    return TimeEvolver(alg, dt, nstep, gate, state)
end

function _bond_rotation(x, bonddir::Int, rev::Bool; inv::Bool = false)
    return if bonddir == 1 # x-bond
        rev ? rot180(x) : x
    elseif bonddir == 2 # y-bond
        if rev
            inv ? rotr90(x) : rotl90(x)
        else
            inv ? rotl90(x) : rotr90(x)
        end
    else
        error("`bonddir` must be 1 (for x-bonds) or 2 (for y-bonds).")
    end
end

"""
Obtain the left (first) cluster tensor from `state` at `site`,
where `in_ax` is the virtual axis connecting to the next tensor.
The tensor is not permuted; the returned `invperm` is the identity.
"""
function _get_left(
        state::InfiniteState, site::CartesianIndex{2}, in_ax::Int,
        env::SUWeight
    )
    Nr, Nc = size(state)
    open_vaxs = _filtered_oneto(in_ax, Val(4))
    s = mod1(site[1], Nr), mod1(site[2], Nc)
    t = absorb_weight(state[s...], env, s[1], s[2], open_vaxs)
    Nax = 4 + numout(eltype(state))
    invperm = (ntuple(identity, Nax - 1), (Nax,))
    return t, open_vaxs, invperm
end

"""
Obtain the right (last) cluster tensor from `state` at `site`,
where `out_ax` is the virtual axis connecting to the previous tensor.
The tensor is not permuted; the returned `invperm` is the identity.
"""
function _get_right(
        state::InfiniteState, site::CartesianIndex{2}, out_ax::Int,
        env::SUWeight
    )
    Nr, Nc = size(state)
    open_vaxs = _filtered_oneto(out_ax, Val(4))
    s = mod1(site[1], Nr), mod1(site[2], Nc)
    t = absorb_weight(state[s...], env, s[1], s[2], open_vaxs)
    Nax = 4 + numout(eltype(state))
    invperm = (ntuple(identity, Nax - 1), (Nax,))
    return t, open_vaxs, invperm
end

"""
Simple update optimized for nearest neighbor gates
utilizing reduced bond tensors with the physical leg.
"""
function _su_iter_gate!(
        state::InfiniteState, gate::NNGate, env::SUWeight,
        siteA::CartesianIndex{2}, siteB::CartesianIndex{2}, alg::SimpleUpdate
    )
    Nr, Nc = size(state)
    trunc = only(_get_cluster_trunc(alg.trunc, [siteA, siteB], (Nr, Nc)))
    in_ax = _nn_vec_direction(siteB - siteA)
    out_ax = mod1(in_ax + 2, 4)
    A0, open_vaxs_A, = _get_left(state, siteA, in_ax, env)
    B0, open_vaxs_B, = _get_right(state, siteB, out_ax, env)
    # rotate
    bond, rev = _nn_bondrev(siteA, siteB, (Nr, Nc))
    dir = first(bond)
    A = _bond_rotation(A0, dir, rev; inv = false)
    B = _bond_rotation(B0, dir, rev; inv = false)
    # apply gate
    ϵ = 0.0
    local s
    gate_axs = alg.purified ? (1:1) : (1:2)
    for gate_ax in gate_axs  # TODO try to use type stable helper function
        X, a, b, Y = _qr_bond(A, B; gate_ax, positive = true)
        a, s, b, ϵ′ = _apply_gate(a, b, gate, trunc)
        ϵ = max(ϵ, ϵ′)
        A, B = _qr_bond_undo(X, a, b, Y)
    end
    rev && (s = transpose(s))
    # rotate back & remove environment weights
    for (site, vertex, open_vaxs) in ((siteA, A, open_vaxs_A), (siteB, B, open_vaxs_B))
        s′ = (mod1(site[1], Nr), mod1(site[2], Nc))
        rotated = _bond_rotation(vertex, dir, rev; inv = true)
        state[s′...] = absorb_weight(rotated, env, s′..., open_vaxs; inv = true)
    end
    env[bond...] = normalize!(s, Inf)
    return ϵ
end

"""
One iteration of simple update
"""
function su_iter(
        state::InfiniteState, circuit::LocalCircuit,
        alg::SimpleUpdate, env::SUWeight
    )
    Nr, Nc, = size(state)
    state2, env2, ϵ = deepcopy(state), deepcopy(env), 0.0
    for (sites, gate) in circuit.gates
        if length(sites) == 1
            # 1-site gate
            # TODO: special treatment for bipartite state
            site = sites[1]
            r, c = mod1(site[1], Nr), mod1(site[2], Nc)
            state2[r, c] = _apply_sitegate(state2[r, c], gate; alg.purified)
        elseif length(sites) == 2
            (d, r, c), = _nn_bondrev(sites..., (Nr, Nc))
            alg.bipartite && r > 1 && continue
            ϵ′ = _su_iter_gate!(state2, gate, env2, sites[1], sites[2], alg)
            ϵ = max(ϵ, ϵ′)
            (!alg.bipartite) && continue
            if d == 1
                rp1, cp1 = _next(r, Nr), _next(c, Nc)
                state2[rp1, cp1] = copy(state2[r, c])
                state2[rp1, c] = copy(state2[r, cp1])
                env2[1, rp1, cp1] = copy(env2[1, r, c])
            else
                rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
                state2[rm1, cm1] = copy(state2[r, c])
                state2[r, cm1] = copy(state2[rm1, c])
                env2[2, rm1, cm1] = copy(env2[2, r, c])
            end
        else
            # N-site MPO gate (N ≥ 2)
            alg.bipartite && error("Multi-site MPO gates are not compatible with bipartite states.")
            ϵ′ = _su_iter_mpo!(state2, gate, env2, sites, alg)
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

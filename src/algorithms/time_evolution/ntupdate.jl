"""
$(TYPEDEF)

Algorithm struct for neighbourhood tensor update (NTU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
@kwdef struct NeighbourUpdate <: TimeEvolution
    "Bond truncation algorithm after applying time evolution gate"
    opt_alg::Union{ALSTruncation, FullEnvTruncation} =
        ALSTruncation(; trunc = truncerror(; atol = 1.0e-10))
    "When true (or false), the Trotter gate is `exp(-H dt)` (or `exp(-iH dt)`)"
    imaginary_time::Bool = true
    "Algorithm to construct NTU bond environment."
    bondenv_alg::NeighbourEnv = NNEnv()
    "When true, fix gauge of bond environment"
    fixgauge::Bool = true
    "When true, assume bipartite unit cell structure"
    bipartite::Bool = false
end

"""
Internal state of neighbourhood tensor update algorithm
"""
struct NTUState{S <: InfiniteState, N <: Number}
    "number of performed iterations"
    iter::Int
    "evolved time"
    t::N
    "PEPS/PEPO"
    psi::S
end

"""
    TimeEvolver(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int, 
        alg::NeighbourUpdate; t0::Number = 0.0, symmetrize_gates::Bool = false
    )

Initialize a TimeEvolver with Hamiltonian `H` and neighbourhood tensor update `alg`, 
starting from the initial state `psi0`.

- The initial time is specified by `t0`.
- Use `symmetrize_gates = true` for second-order Trotter decomposition.
"""
function TimeEvolver(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::NeighbourUpdate; t0::Number = 0.0, symmetrize_gates::Bool = false
    )
    _timeevol_sanity_check(psi0, physicalspace(H), alg)
    dt′ = _get_dt(psi0, dt, alg.imaginary_time)
    gate = trotterize(H, dt′; symmetrize_gates)
    state = NTUState(0, t0, psi0)
    return TimeEvolver(alg, dt, nstep, gate, state)
end

"""
Neighbourhood tensor update optimized for nearest neighbor gates
utilizing reduced bond tensors with the physical leg.
"""
function _ntu_iter(
        state::InfiniteState, gate::NNGate, wts::SUWeight,
        sites::Vector{CartesianIndex{2}}, alg::NeighbourUpdate
    )
    @assert length(sites) == 2
    return _bond_truncate(state, wts, Tuple(sites), alg; gate)
end

"""
One round of neighbourhood tensor update
"""
function ntu_iter(
        state::InfiniteState, circuit::LocalCircuit, alg::NeighbourUpdate
    )
    Nr, Nc, = size(state)
    state2, wts = deepcopy(state), SUWeight(state)
    info = (; fid = 1.0)
    for (sites, gate) in circuit.gates
        if length(sites) == 1
            # 1-site gate
            # TODO: special treatment for bipartite state
            site = sites[1]
            r, c = mod1(site[1], Nr), mod1(site[2], Nc)
            state2[r, c] = _apply_sitegate(state2[r, c], gate)
            info′ = (; fid = 1.0)
        elseif length(sites) == 2
            (d, r, c), = _nn_bondrev(sites..., (Nr, Nc))
            if alg.bipartite
                length(sites) > 2 && error("Multi-site MPO gates are not compatible with bipartite states.")
                r > 1 && continue
            end
            state2, wts, info′ = _ntu_iter(state2, gate, wts, sites, alg)
            (!alg.bipartite) && continue
            if d == 1
                rp1, cp1 = _next(r, Nr), _next(c, Nc)
                state2[rp1, cp1] = deepcopy(state2[r, c])
                state2[rp1, c] = deepcopy(state2[r, cp1])
                wts[1, rp1, cp1] = deepcopy(wts[1, r, c])
            else
                rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
                state2[rm1, cm1] = deepcopy(state2[r, c])
                state2[r, cm1] = deepcopy(state2[rm1, c])
                wts[2, rm1, cm1] = deepcopy(wts[2, r, c])
            end
        else
            # N-site MPO gate (N ≥ 2)
            alg.bipartite && error("Multi-site MPO gates are not compatible with bipartite states.")
            state2, wts, info′ = _ntu_iter(state2, gate, wts, sites, alg)
        end
        # record the worst fidelity
        (info′.fid < info.fid) && (info = info′)
    end
    return state2, wts, info
end

function Base.iterate(it::TimeEvolver{<:NeighbourUpdate}, state = it.state)
    iter, t = state.iter, state.t
    (iter == it.nstep) && return nothing
    psi, wts, info = ntu_iter(state.psi, it.gate, it.alg)
    iter, t = iter + 1, t + it.dt
    # update internal state
    it.state = NTUState(iter, t, psi)
    info = (; (; t, wts)..., info...)
    return (psi, info), it.state
end

"""
    timestep(
        it::TimeEvolver{<:NeighbourUpdate}, psi::InfiniteState;
        iter::Int = it.state.iter, t::Float64 = it.state.t
    ) -> (psi, info)

Given the TimeEvolver iterator `it`, perform one step of NTU time evolution
on the input state `psi`.

- Using `iter` and `t` to reset the current iteration number and evolved time
    respectively of the TimeEvolver `it`.
"""
function MPSKit.timestep(
        it::TimeEvolver{<:NeighbourUpdate}, psi::InfiniteState;
        iter::Int = it.state.iter, t::Float64 = it.state.t
    )
    _timeevol_sanity_check(psi, physicalspace(it.state.psi), it.alg)
    state = NTUState(iter, t, psi)
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
        it::TimeEvolver{<:NeighbourUpdate}; check_interval::Int = 500
    ) -> (psi, info)

Perform time evolution to the end of `TimeEvolver` iterator `it`.

- `check_interval` sets the number of iterations between outputs of information.
"""
function MPSKit.time_evolve(it::TimeEvolver{<:NeighbourUpdate}; check_interval::Int = 500)
    time_start = time0 = time()
    @info "--- Time evolution (neighbourhood tensor update), dt = $(it.dt) ---"
    info0 = nothing
    for (psi, info) in it
        iter = it.state.iter
        stop = (iter == it.nstep)
        showinfo = (iter == 1) || (iter % check_interval == 0) || stop
        time1 = time()
        if showinfo
            Δλ = (info0 === nothing) ? NaN : compare_weights(info.wts, info0.wts)
            @info @sprintf(
                "NTU iter %d: t = %.2e, |Δλ| = %.3e. Time: %.2f s",
                it.state.iter, it.state.t, Δλ, time1 - time0
            )
        end
        if stop
            time_end = time()
            @info @sprintf("Time evolution finished in %.2f s", time_end - time_start)
            return psi, info
        end
        info0, time0 = info, time()
    end
    return
end

"""
    time_evolve(
        psi0::Union{InfinitePEPS, InfinitePEPO}, H::LocalOperator, 
        dt::Number, nstep::Int, alg::NeighbourUpdate; 
        t0::Number = 0.0, symmetrize_gates::Bool = false,
        check_interval::Int = 10
    ) -> (psi, info)

Perform time evolution on the initial state `psi0` and initial environment `env0`
with Hamiltonian `H`, using `NeighbourUpdate` algorithm `alg`, time step `dt` for 
`nstep` number of steps. 

- Convergence check for ground state search is not supported.
- Set `symmetrize_gates = true` for second-order Trotter decomposition.
- Use `t0` to specify the initial time of `psi0`.
- `check_interval` sets the interval to output information (and check convergence). 
    Output during the evolution can be turned off by setting `check_interval <= 0`.
- `info` is a NamedTuple containing information of the evolution, 
    including the time `info.t` evolved since `psi0`.
"""
function MPSKit.time_evolve(
        psi0::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::NeighbourUpdate; symmetrize_gates::Bool = false,
        t0::Number = 0.0, check_interval::Int = 10
    )
    it = TimeEvolver(psi0, H, dt, nstep, alg; t0, symmetrize_gates)
    return time_evolve(it; check_interval)
end

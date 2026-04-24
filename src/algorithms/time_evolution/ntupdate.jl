"""
$(TYPEDEF)

Algorithm struct for neighbourhood tensor update (NTU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)

Reference: 
- Physical Review B 104, 094411 (2021)
- Physical Review B 106, 195105 (2022)
"""
@kwdef struct NeighbourUpdate{
        TR <: Union{ALSTruncation, FullEnvTruncation},
        BE <: NeighbourEnv,
    } <: TimeEvolution
    "Bond truncation algorithm after applying time evolution gate"
    opt_alg::TR = ALSTruncation(; trunc = truncerror(; atol = 1.0e-10))
    "When true (or false), the Trotter gate is `exp(-H dt)` (or `exp(-iH dt)`)"
    imaginary_time::Bool = true
    "Algorithm to construct NTU bond environment."
    bondenv_alg::BE = NNEnv()
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
    state2, wts = copy(state), SUWeight(state)
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
            alg.bipartite && r > 1 && continue
            state2, wts, info′ = _ntu_iter(state2, gate, wts, sites, alg)
            (!alg.bipartite) && continue
            if d == 1
                rp1, cp1 = _next(r, Nr), _next(c, Nc)
                state2[rp1, cp1] = copy(state2[r, c])
                state2[rp1, c] = copy(state2[r, cp1])
                wts[1, rp1, cp1] = copy(wts[1, r, c])
            else
                rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
                state2[rm1, cm1] = copy(state2[r, c])
                state2[r, cm1] = copy(state2[rm1, c])
                wts[2, rm1, cm1] = copy(wts[2, r, c])
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
        it::TimeEvolver{<:NeighbourUpdate},
        [H::LocalOperator, env::CTMRGEnv, ctm_alg::CTMRGAlgorithm];
        tol::Float64 = 1.0e-7, check_interval::Int = 10
    ) -> (psi, info)

Perform time evolution to the end of `NeighbourUpdate` TimeEvolver `it`,
or until convergence of energy set by a positive `tol`.

To enable convergence check (for imaginary time evolution of InfinitePEPS only),
provide the Hamiltonian `H`, CTMRG environment `env`, CTMRG algorithm `ctm_alg`
and setting `tol > 0`.

`check_interval` sets the number of iterations between energy checks
(for ground state search) and outputs of information.
"""
function MPSKit.time_evolve(it::TimeEvolver{<:NeighbourUpdate}; check_interval::Int = 50)
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

function MPSKit.time_evolve(
        it::TimeEvolver{<:NeighbourUpdate, G, S},
        H::LocalOperator, env::CTMRGEnv, ctm_alg::CTMRGAlgorithm;
        tol::Float64 = 1.0e-7, check_interval::Int = 10
    ) where {G, S <: NTUState{<:InfinitePEPS}}
    @info "--- Time evolution (neighbourhood tensor update), dt = $(it.dt) ---"
    time_start = time0 = time()
    psi0 = copy(it.state.psi)
    @assert it.alg.imaginary_time "Only imaginary time evolution of InfinitePEPS allows convergence checking."
    # initial energy
    env, = leading_boundary(env, psi0, ctm_alg)
    energy = real(expectation_value(psi0, H, env)) / prod(size(psi0))
    @info @sprintf("NTU iter 0: E = %.4e", energy)
    info0 = (; energy, env)
    # start evolving
    energy0, ΔE = energy, 0.0
    iter0, t0 = it.state.iter, it.state.t
    for (psi, info) in it
        iter = it.state.iter
        showinfo = (iter == 1) || (iter % check_interval == 0) || (iter == it.nstep)
        !showinfo && continue
        # bond weight change
        Δλ = hasproperty(info0, :wts) ? compare_weights(info.wts, info0.wts) : NaN
        # reconverge environment
        if all(space(t) == space(t0) for (t, t0) in zip(psi.A, psi0.A))
            # recreate `env` from bond weights if psi virtual space changed
            env = CTMRGEnv(info.wts)
        end
        env, = leading_boundary(env, psi, ctm_alg)
        # measure energy
        energy = real(expectation_value(psi, H, env)) / prod(size(psi))
        ΔE = energy - energy0
        info = @insert info.energy = energy
        info = @insert info.env = env
        # show information
        time1 = time()
        @info @sprintf(
            "NTU iter %-6d: E = %.5f, ΔE = %.3e, |Δλ| = %.3e. Time: %.2f s",
            it.state.iter, energy, ΔE, Δλ, time1 - time0
        )
        # determine whether to stop evolution
        stop = false
        if (ΔE <= 0 && abs(ΔE) < tol)
            stop = true
            @info "NTU: energy has converged."
        end
        if ΔE > 0
            stop = true
            @warn "NTU: energy has increased. Abort evolution and return results from last check."
            psi, info, energy = psi0, info0, energy0
            it.state = NTUState(iter0, t0, psi0)
        end
        if iter == it.nstep
            stop = true
            @info "NTU: reached maximum iteration."
        end
        if stop
            time_end = time()
            @info @sprintf("Time evolution finished in %.2f s", time_end - time_start)
            return psi, info
        else
            iter0, t0 = it.state.iter, it.state.t
            psi0, energy0, info0 = psi, energy, info
        end
        time0 = time()
    end
    return
end

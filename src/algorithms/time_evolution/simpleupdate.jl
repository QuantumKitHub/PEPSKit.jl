"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SimpleUpdate <: TimeEvolution
    # Truncation scheme after applying Trotter gates
    trunc::TruncationStrategy
    # Switch for imaginary or real time
    imaginary_time::Bool = true
    # controls interval to print information
    check_interval::Int = 500
    # force usage of 3-site simple update
    force_3site::Bool = false
    # ---- only applicable to ground state search ----
    # assume bipartite unit cell structure
    bipartite::Bool = false
    # ---- only applicable to PEPO evolution ----
    gate_bothsides::Bool = false # when false, purified approach is assumed
end

# internal state of simple update algorithm
struct SUState{S}
    # number of performed iterations
    iter::Int
    # evolved time
    t::Float64
    # measure of difference (of SUWeight, energy, etc.) from last iteration
    diff::Float64
    # PEPS/PEPO
    ψ::S
    # SUWeight environment
    env::SUWeight
end

"""
    TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int, 
        alg::SimpleUpdate, env₀::SUWeight; 
        tol::Float64 = 0.0, t₀::Float64 = 0.0
    )

Initialize a TimeEvolver with Hamiltonian `H` and simple update `alg`, 
starting from the initial state `ψ₀` and SUWeight environment `env₀`.

- The initial (real or imaginary) time is specified by `t₀`.
- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only).
    For other usages it should not be changed.
"""
function TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int,
        alg::SimpleUpdate, env₀::SUWeight;
        tol::Float64 = 0.0, t₀::Float64 = 0.0
    )
    # sanity checks
    _timeevol_sanity_check(ψ₀, physicalspace(H), tol, alg)
    # create Trotter gates
    nnonly = is_nearest_neighbour(H)
    use_3site = alg.force_3site || !nnonly
    if alg.bipartite
        @assert !use_3site "3-site simple update is incompatible with bipartite lattice."
    end
    dt′ = _get_dt(ψ₀, dt, alg.imaginary_time)
    gate = if use_3site
        [
            _get_gatempos_se(H, dt′),
            _get_gatempos_se(rotl90(H), dt′),
            _get_gatempos_se(rot180(H), dt′),
            _get_gatempos_se(rotr90(H), dt′),
        ]
    else
        get_expham(H, dt′)
    end
    state = SUState(0, t₀, NaN, ψ₀, env₀)
    return TimeEvolver(alg, dt, nstep, gate, tol, state)
end

#=
Simple update of the x-bond between `[r,c]` and `[r,c+1]`.

```
        |           |
    -- T[r,c] -- T[r,c+1] --
        |           |
```
=#
function _su_xbond!(
        state::InfiniteState, gate::AbstractTensorMap{T, S, 2, 2}, env::SUWeight,
        row::Int, col::Int, trunc::TruncationStrategy; gate_ax::Int = 1
    ) where {T <: Number, S <: ElementarySpace}
    Nr, Nc, = size(state)
    cp1 = _next(col, Nc)
    # absorb environment weights
    A, B = state.A[row, col], state.A[row, cp1]
    A = absorb_weight(A, env, row, col, (NORTH, SOUTH, WEST); inv = false)
    B = absorb_weight(B, env, row, cp1, (NORTH, SOUTH, EAST); inv = false)
    normalize!(A, Inf)
    normalize!(B, Inf)
    # apply gate
    X, a, b, Y = _qr_bond(A, B; gate_ax)
    a, s, b, ϵ = _apply_gate(a, b, gate, trunc)
    A, B = _qr_bond_undo(X, a, b, Y)
    # remove environment weights
    A = absorb_weight(A, env, row, col, (NORTH, SOUTH, WEST); inv = true)
    B = absorb_weight(B, env, row, cp1, (NORTH, SOUTH, EAST); inv = true)
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    # update tensor dict and weight on current bond
    state.A[row, col], state.A[row, cp1] = A, B
    env.data[1, row, col] = s
    return ϵ
end

#=
Simple update of the y-bond between `[r,c]` and `[r-1,c]`.
```
        |
    --T[r-1,c] --
        |
    -- T[r,c] ---
        |
```
=#
function _su_ybond!(
        state::InfiniteState, gate::AbstractTensorMap{T, S, 2, 2}, env::SUWeight,
        row::Int, col::Int, trunc::TruncationStrategy; gate_ax::Int = 1
    ) where {T <: Number, S <: ElementarySpace}
    Nr, Nc, = size(state)
    rm1 = _prev(row, Nr)
    # absorb environment weights
    A, B = state.A[row, col], state.A[rm1, col]
    A = absorb_weight(A, env, row, col, (EAST, SOUTH, WEST); inv = false)
    B = absorb_weight(B, env, rm1, col, (NORTH, EAST, WEST); inv = false)
    normalize!(A, Inf)
    normalize!(B, Inf)
    # apply gate
    X, a, b, Y = _qr_bond(rotr90(A), rotr90(B); gate_ax)
    a, s, b, ϵ = _apply_gate(a, b, gate, trunc)
    A, B = rotl90.(_qr_bond_undo(X, a, b, Y))
    # remove environment weights
    A = absorb_weight(A, env, row, col, (EAST, SOUTH, WEST); inv = true)
    B = absorb_weight(B, env, rm1, col, (NORTH, EAST, WEST); inv = true)
    # update tensor dict and weight on current bond
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    state.A[row, col], state.A[rm1, col] = A, B
    env.data[2, row, col] = s
    return ϵ
end

function su_iter(
        state::InfiniteState, gate::LocalOperator, alg::SimpleUpdate, env::SUWeight
    )
    Nr, Nc, = size(state)
    state2, env2, ϵ = deepcopy(state), deepcopy(env), 0.0
    gate_axs = alg.gate_bothsides ? (1:2) : (1:1)
    for r in 1:Nr, c in 1:Nc
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r, c + 1)))
        trunc = truncation_strategy(alg.trunc, 1, r, c)
        for gate_ax in gate_axs
            ϵ′ = _su_xbond!(state2, term, env2, r, c, trunc; gate_ax)
            ϵ = max(ϵ, ϵ′)
        end
        if alg.bipartite
            rp1, cp1 = _next(r, Nr), _next(c, Nc)
            state2.A[rp1, cp1] = deepcopy(state2.A[r, c])
            state2.A[rp1, c] = deepcopy(state2.A[r, cp1])
            env2.data[1, rp1, cp1] = deepcopy(env2.data[1, r, c])
        end
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r - 1, c)))
        trunc = truncation_strategy(alg.trunc, 2, r, c)
        for gate_ax in gate_axs
            ϵ′ = _su_ybond!(state2, term, env2, r, c, trunc; gate_ax)
            ϵ = max(ϵ, ϵ′)
        end
        if alg.bipartite
            rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
            state2.A[rm1, cm1] = deepcopy(state2.A[r, c])
            state2.A[r, cm1] = deepcopy(state2.A[rm1, c])
            env2.data[2, rm1, cm1] = deepcopy(env2.data[2, r, c])
        end
    end
    return state2, env2, ϵ
end

function Base.iterate(it::TimeEvolver{<:SimpleUpdate}, state = it.state)
    alg = it.alg
    iter, t, diff = state.iter, state.t, state.diff
    check_convergence = (it.tol > 0)
    if check_convergence
        if diff < it.tol
            @info "SU: bond weights have converged."
            return nothing
        elseif iter == it.nstep
            @warn "SU: bond weights have not converged."
            return nothing
        end
    else
        (iter == it.nstep) && return nothing
    end
    # perform one iteration and record time
    time0 = time()
    ψ, env, ϵ = su_iter(state.ψ, it.gate, it.alg, state.env)
    diff = compare_weights(env, state.env)
    elapsed_time = time() - time0
    # update internal state
    iter += 1
    t += it.dt
    it.state = SUState(iter, t, diff, ψ, env)
    # output information
    converged = check_convergence ? (diff < it.tol) : false
    showinfo = (iter % alg.check_interval == 0) || (iter == 1) || (iter == it.nstep) || converged
    if showinfo
        @info "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
        @info @sprintf(
            "SU iter %-7d: dt = %.0e, |Δλ| = %.3e. Time = %.3f s/it",
            iter, it.dt, diff, elapsed_time
        )
    end
    info = (; t, ϵ)
    return (ψ, env, info), it.state
end

"""
    timestep(
        it::TimeEvolver{<:SimpleUpdate}, ψ::InfiniteState, env::SUWeight;
        iter::Int = it.state.iter, t::Float64 = it.state.t
    ) -> (ψ, env, info)

Given the TimeEvolver iterator `it`, perform one step of time evolution
on the input state `ψ` and its environment `env`.

- Using `iter` and `t` to reset the current iteration number and evolved time
    respectively of the TimeEvolver `it`.
"""
function MPSKit.timestep(
        it::TimeEvolver{<:SimpleUpdate}, ψ::InfiniteState, env::SUWeight;
        iter::Int = it.state.iter, t::Float64 = it.state.t
    )
    Pspaces = physicalspace(it.state.ψ)
    _timeevol_sanity_check(ψ, Pspaces, it.tol, it.alg)
    state = SUState(iter, t, NaN, ψ, env)
    result = iterate(it, state)
    if result === nothing
        @warn "TimeEvolver `it` has already reached the end."
        return ψ, env, (; t, ϵ = NaN)
    else
        return first(result)
    end
end

"""
    time_evolve(it::TimeEvolver{<:SimpleUpdate})

Perform time evolution until the set number of iterations or convergence
directly using the specified TimeEvolver iterator.
"""
function MPSKit.time_evolve(it::TimeEvolver{<:SimpleUpdate})
    time_start = time()
    result = nothing
    for state in it
        result = state
    end
    time_end = time()
    @info @sprintf("Simple update finished. Total time elasped: %.2f s", time_end - time_start)
    return result
end

"""
    time_evolve(
        ψ₀::Union{InfinitePEPS, InfinitePEPO}, H::LocalOperator, 
        dt::Float64, nstep::Int, alg::SimpleUpdate, env₀::SUWeight;
        tol::Float64 = 0.0, t₀::Float64 = 0.0
    ) -> (ψ, env, info)

Perform time evolution on the initial state `ψ₀` and initial environment `env₀`
with Hamiltonian `H`, using SimpleUpdate algorithm `alg`, time step `dt` for 
`nstep` number of steps. 

- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only).
    For other usages it should not be changed.
- Using `t₀` to specify the initial (real or imaginary) time of `ψ₀`.
- `info` is a NamedTuple containing information of the evolution, 
    including the time evolved since `ψ₀`.
"""
function MPSKit.time_evolve(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int,
        alg::SimpleUpdate, env₀::SUWeight;
        tol::Float64 = 0.0, t₀::Float64 = 0.0
    )
    it = TimeEvolver(ψ₀, H, dt, nstep, alg, env₀; tol, t₀)
    return time_evolve(it)
end

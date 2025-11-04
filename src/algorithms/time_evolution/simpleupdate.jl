"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SimpleUpdate <: TimeEvolution
    # Initial state (InfinitePEPS or InfinitePEPO)
    ψ0::InfiniteState
    # Hamiltonian
    H::LocalOperator
    # Initial Bond weights
    env0::SUWeight
    # Trotter time step
    dt::Float64
    # number of iteration steps
    nstep::Int
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
    # converged change of weight between two steps
    tol::Float64 = 0.0
    # ---- only applicable to PEPO evolution ----
    gate_bothsides::Bool = false # when false, purified approach is assumed
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
    Nr, Nc = size(state)[1:2]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
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
    Nr, Nc = size(state)[1:2]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
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
    @assert size(gate.lattice) == size(state)[1:2]
    Nr, Nc = size(state)[1:2]
    alg.bipartite && (@assert Nr == Nc == 2)
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError("`state` unit cell size for simple update should be no smaller than (2, 2)."),
    )
    state2, env2 = deepcopy(state), deepcopy(env)
    gate_axs = alg.gate_bothsides ? (1:2) : (1:1)
    for r in 1:Nr, c in 1:Nc
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r, c + 1)))
        trunc = truncation_strategy(alg.trunc, 1, r, c)
        for gate_ax in gate_axs
            _su_xbond!(state2, term, env2, r, c, trunc; gate_ax)
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
            _su_ybond!(state2, term, env2, r, c, trunc; gate_ax)
        end
        if alg.bipartite
            rm1, cm1 = _prev(r, Nr), _prev(c, Nc)
            state2.A[rm1, cm1] = deepcopy(state2.A[r, c])
            state2.A[r, cm1] = deepcopy(state2.A[rm1, c])
            env2.data[2, rm1, cm1] = deepcopy(env2.data[2, r, c])
        end
    end
    wtdiff = compare_weights(env2, env)
    return state2, env2, (; wtdiff)
end

# Initial iteration
function Base.iterate(alg::SimpleUpdate)
    # sanity checks
    nnonly = is_nearest_neighbour(alg.H)
    use_3site = alg.force_3site || !nnonly
    @assert alg.tol >= 0
    @assert !(alg.bipartite && alg.ψ0 isa InfinitePEPO) "Evolution of PEPO with bipartite structure is not implemented."
    @assert !(alg.bipartite && use_3site) "3-site simple update is incompatible with bipartite lattice."
    if alg.ψ0 isa InfinitePEPS
        @assert !(alg.gate_bothsides) "alg.gate_bothsides = true is incompatible with PEPS evolution."
    end
    # construct Trotter 2-site gates or 3-site gate-MPOs
    dt′ = _process_timeevol_args(alg.ψ0, alg.dt, alg.imaginary_time)
    gate = if use_3site
        [
            _get_gatempos_se(alg.H, dt′),
            _get_gatempos_se(rotl90(alg.H), dt′),
            _get_gatempos_se(rot180(alg.H), dt′),
            _get_gatempos_se(rotr90(alg.H), dt′),
        ]
    else
        get_expham(alg.H, dt′)
    end
    time0 = time()
    ψ, env, info = su_iter(alg.ψ0, gate, alg, alg.env0)
    elapsed_time = time() - time0
    converged = false
    return (ψ, env, info), (ψ, env, info, gate, converged, 1, elapsed_time)
end

# subsequent iterations
function Base.iterate(alg::SimpleUpdate, state)
    ψ0, env0, info, gate, converged, it, elapsed_time = state
    check_convergence = (alg.tol > 0)
    if (it % alg.check_interval == 0) || (it == 1) || (it == alg.nstep) || converged
        @info "Space of x-weight at [1, 1] = $(space(env0[1, 1, 1], 1))"
        @info @sprintf(
            "SU iter %-7d: dt = %.0e, |Δλ| = %.3e. Time = %.3f s/it",
            it, alg.dt, info.wtdiff, elapsed_time
        )
        if (it == alg.nstep)
            check_convergence && (@warn "SU: bond weights have not converged.")
            return nothing
        elseif converged
            @info "SU: bond weights have converged."
            return nothing
        end
    end
    time0 = time()
    ψ, env, info = su_iter(ψ0, gate, alg, env0)
    elapsed_time = time() - time0
    converged = check_convergence ? (info.wtdiff < alg.tol) : false
    return (ψ, env, info), (ψ, env, info, gate, converged, it + 1, elapsed_time)
end

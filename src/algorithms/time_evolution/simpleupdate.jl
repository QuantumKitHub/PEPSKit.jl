"""
$(TYPEDEF)

Algorithm struct for simple update (SU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct SimpleUpdate
    trscheme::TruncationScheme
    check_interval::Int = 500
    # only applicable to ground state search
    bipartite::Bool = false
    tol::Float64 = 0.0
    # only applicable to PEPO evolution
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
        row::Int, col::Int, trscheme::TruncationScheme; gate_ax::Int = 1
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
    a, s, b, ϵ = _apply_gate(a, b, gate, trscheme)
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
        row::Int, col::Int, trscheme::TruncationScheme; gate_ax::Int = 1
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
    a, s, b, ϵ = _apply_gate(a, b, gate, trscheme)
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
        trscheme = truncation_scheme(alg.trscheme, 1, r, c)
        for gate_ax in gate_axs
            _su_xbond!(state2, term, env2, r, c, trscheme; gate_ax)
        end
        if alg.bipartite
            rp1, cp1 = _next(r, Nr), _next(c, Nc)
            state2.A[rp1, cp1] = deepcopy(state2.A[r, c])
            state2.A[rp1, c] = deepcopy(state2.A[r, cp1])
            env2.data[1, rp1, cp1] = deepcopy(env2.data[1, r, c])
        end
        term = get_gateterm(gate, (CartesianIndex(r, c), CartesianIndex(r - 1, c)))
        trscheme = truncation_scheme(alg.trscheme, 2, r, c)
        for gate_ax in gate_axs
            _su_ybond!(state2, term, env2, r, c, trscheme; gate_ax)
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

"""
Perform simple update with Hamiltonian `ham` containing up to nearest neighbor interaction terms. 
"""
function _simpleupdate2site(
        state::InfiniteState, H::LocalOperator,
        dt::Float64, nstep::Int, alg::SimpleUpdate, env::SUWeight;
        imaginary_time::Bool
    )
    time_start = time()
    dt′, check_convergence = _process_timeevol_args(state, dt, imaginary_time, alg.tol)
    gate = get_expham(H, dt′)
    info = nothing
    for step in 1:nstep
        time0 = time()
        state, env, info = su_iter(state, gate, alg, env)
        converge = (info.wtdiff < alg.tol)
        time1 = time()
        if (step % alg.check_interval == 0) || (step == nstep) || converge
            @info "Space of x-weight at [1, 1] = $(space(env[1, 1, 1], 1))"
            @info @sprintf(
                "SU iter %-7d: dt = %.0e, |Δλ| = %.3e. Time = %.3f s/it",
                step, dt, info.wtdiff, time1 - time0
            )
        end
        if check_convergence
            converge && break
            if step == nstep
                @warn "SU ground state search has not converged."
            end
        end
    end
    time_end = time()
    @info @sprintf("Simple update finished. Time elasped: %.2f s", time_end - time_start)
    return state, env, info
end

function MPSKit.time_evolve(
        state::InfiniteState, H::LocalOperator,
        dt::Float64, nstep::Int, alg::SimpleUpdate, env::SUWeight;
        imaginary_time::Bool = true, force_3site::Bool = false
    )
    # determine if Hamiltonian contains nearest neighbor terms only
    nnonly = is_nearest_neighbour(H)
    use_3site = force_3site || !nnonly
    @assert !(alg.bipartite && state isa InfinitePEPO) "Evolution of PEPO with bipartite structure is not implemented."
    @assert !(alg.bipartite && use_3site) "3-site simple update is incompatible with bipartite lattice."
    if state isa InfinitePEPS
        @assert !(alg.gate_bothsides) "alg.gate_bothsides = true is incompatible with PEPS evolution."
    end
    # TODO: check SiteDependentTruncation is compatible with bipartite structure
    if use_3site
        return _simpleupdate3site(state, H, dt, nstep, alg, env; imaginary_time)
    else
        return _simpleupdate2site(state, H, dt, nstep, alg, env; imaginary_time)
    end
end

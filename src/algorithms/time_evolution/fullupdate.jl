"""
$(TYPEDEF)

Algorithm struct for full update (FU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)

Reference: Physical Review B 92, 035142 (2015)
"""
@kwdef struct FullUpdate <: TimeEvolution
    # Fix gauge of bond environment
    fixgauge::Bool = true
    # Switch for imaginary or real time
    imaginary_time::Bool = true
    # Number of iterations without fully reconverging the environment
    reconverge_interval::Int = 5
    # Bond truncation algorithm after applying time evolution gate
    opt_alg::Union{ALSTruncation, FullEnvTruncation} = ALSTruncation(;
        trunc = truncerror(; atol = 1.0e-10)
    )
    # CTMRG algorithm to reconverge environment. Its `projector_alg` is also used for the fast update of the environment after each FU iteration
    ctm_alg::CTMRGAlgorithm = SequentialCTMRG(;
        tol = 1.0e-9, maxiter = 20, verbosity = 1,
        trunc = truncerror(; atol = 1.0e-10),
        projector_alg = :fullinfinite,
    )
end

# internal state of full update algorithm
struct FUState{S <: InfiniteState, E <: CTMRGEnv}
    # number of performed iterations
    iter::Int
    # evolved time
    t::Float64
    # PEPS/PEPO
    ψ::S
    # CTMRG environment
    env::E
    # whether the current environment is reconverged
    reconverged::Bool
end

"""
    TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int, 
        alg::FullUpdate, env₀::CTMRGEnv; t₀::Float64 = 0.0
    )

Initialize a TimeEvolver with Hamiltonian `H` and full update `alg`, 
starting from the initial state `ψ₀` and CTMRG environment `env₀`.

- The initial (real or imaginary) time is specified by `t₀`.
"""
function TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int,
        alg::FullUpdate, env₀::CTMRGEnv; t₀::Float64 = 0.0
    )
    _timeevol_sanity_check(ψ₀, physicalspace(H), alg)
    dt′ = _get_dt(ψ₀, dt, alg.imaginary_time)
    gate = get_expham(H, dt′ / 2)
    state = FUState(0, t₀, ψ₀, env₀, true)
    return TimeEvolver(alg, dt, nstep, gate, state)
end

"""
Full update for the bond between `[row, col]` and `[row, col+1]`.
"""
function _fu_xbond!(
        state::InfiniteState, gate::AbstractTensorMap{T, S, 2, 2}, env::CTMRGEnv,
        row::Int, col::Int, alg::FullUpdate
    ) where {T <: Number, S <: ElementarySpace}
    cp1 = _next(col, size(state, 2))
    A, B = state[row, col], state[row, cp1]
    X, a, b, Y = _qr_bond(A, B)
    # positive/negative-definite approximant: benv = ± Z Z†
    benv = bondenv_fu(row, col, X, Y, env)
    Z = positive_approx(benv)
    @debug "cond(benv) before gauge fix: $(LinearAlgebra.cond(Z' * Z))"
    # fix gauge
    if alg.fixgauge
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(Z' * Z))"
    end
    benv = Z' * Z
    # apply gate
    a, s, b, = _apply_gate(a, b, gate, truncerror(; atol = 1.0e-15))
    # optimize a, b
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    normalize!(a, Inf)
    normalize!(b, Inf)
    A, B = _qr_bond_undo(X, a, b, Y)
    normalize!(A, Inf)
    normalize!(B, Inf)
    normalize!(s, Inf)
    state.A[row, col], state.A[row, cp1] = A, B
    return s, info
end

"""
Update all horizontal bonds in the c-th column
(i.e. `(r,c) (r,c+1)` for all `r = 1, ..., Nr`).
To update rows, rotate the network clockwise by 90 degrees.
The iPEPS/iPEPO `state` is modified in place.
"""
function _fu_column!(
        state::InfiniteState, gate::LocalOperator,
        alg::FullUpdate, env::CTMRGEnv, col::Int
    )
    Nr, Nc, = size(state)
    fid = 1.0
    wts_col = Vector{PEPSWeight}(undef, Nr)
    for row in 1:Nr
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        wts_col[row], info = _fu_xbond!(state, term, env, row, col, alg)
        fid = min(fid, info.fid)
    end
    # update 2-layer CTMRGEnv
    network = if isa(state, InfinitePEPS)
        InfiniteSquareNetwork(state)
    else
        InfiniteSquareNetwork(InfinitePEPS(state))
    end
    colmove_alg = SequentialCTMRG()
    @reset colmove_alg.projector_alg = alg.ctm_alg.projector_alg
    env2, info = ctmrg_leftmove(col, network, env, colmove_alg)
    env2, info = ctmrg_rightmove(_next(col, Nc), network, env2, colmove_alg)
    for c in [col, _next(col, Nc)]
        env.corners[:, :, c] = env2.corners[:, :, c]
        env.edges[:, :, c] = env2.edges[:, :, c]
    end
    return wts_col, fid
end

"""
One round of fast full update on the input InfinitePEPS or InfinitePEPO `state`
and its 2-layer CTMRGEnv `env`, without fully reconverging `env`.
"""
function fu_iter(
        state::InfiniteState, gate::LocalOperator, alg::FullUpdate, env::CTMRGEnv
    )
    Nr, Nc, = size(state)
    fidmin = 1.0
    state2, env2 = deepcopy(state), deepcopy(env)
    wts = Array{PEPSWeight}(undef, 2, Nr, Nc)
    for i in 1:4
        N = size(state2, 2)
        for col in 1:N
            wts_col, fid_col = _fu_column!(state2, gate, alg, env2, col)
            fidmin = min(fidmin, fid_col)
            # assign the weights to the un-rotated `wts`
            if i == 1
                wts[1, :, col] = wts_col
            elseif i == 2
                wts[2, _next(col, N), :] = reverse(wts_col)
            elseif i == 3
                wts[1, :, mod1(N - col, N)] = reverse(wts_col)
            else
                wts[2, N + 1 - col, :] = wts_col
            end
        end
        gate, state2, env2 = rotl90(gate), rotl90(state2), rotl90(env2)
    end
    return state2, env2, SUWeight(collect(wt for wt in wts)), fidmin
end

function Base.iterate(it::TimeEvolver{<:FullUpdate}, state = it.state)
    iter, t = state.iter, state.t
    (iter == it.nstep) && return nothing
    ψ, env, wts, fid = fu_iter(state.ψ, it.gate, it.alg, state.env)
    iter, t = iter + 1, t + it.dt
    # reconverge environment for the last step and every `reconverge_interval` steps
    reconverged = (iter % it.alg.reconverge_interval == 0) || (iter == it.nstep)
    if reconverged
        network = isa(ψ, InfinitePEPS) ? ψ : InfinitePEPS(ψ)
        env, = leading_boundary(env, network, it.alg.ctm_alg)
    end
    # update internal state
    it.state = FUState(iter, t, ψ, env, reconverged)
    info = (; t, wts, fid)
    return (ψ, env, info), it.state
end

"""
    timestep(
        it::TimeEvolver{<:FullUpdate}, ψ::InfiniteState, env::CTMRGEnv;
        iter::Int = it.state.iter, t::Float64 = it.state.t
    ) -> (ψ, env, info)

Given the TimeEvolver iterator `it`, perform one step of time evolution
on the input state `ψ` and its environment `env`.

- Using `iter` and `t` to reset the current iteration number and evolved time
    respectively of the TimeEvolver `it`.
- Use `reconverge_env` to force reconverging the obtained environment.
"""
function MPSKit.timestep(
        it::TimeEvolver{<:FullUpdate}, ψ::InfiniteState, env::CTMRGEnv;
        iter::Int = it.state.iter, t::Float64 = it.state.t, reconverge_env::Bool = false
    )
    _timeevol_sanity_check(ψ, physicalspace(it.state.ψ), it.alg)
    state = FUState(iter, t, ψ, env, true)
    result = iterate(it, state)
    if result === nothing
        @warn "TimeEvolver `it` has already reached the end."
        return nothing
    else
        ψ, env, info = first(result)
        if reconverge_env && !(it.state.reconverged)
            network = isa(ψ, InfinitePEPS) ? ψ : InfinitePEPS(ψ)
            env, = leading_boundary(env, network, it.alg.ctm_alg)
            # update internal state of `it`
            state0 = it.state
            it.state = (@set state0.env = env)
        end
        return ψ, env, info
    end
end

function MPSKit.time_evolve(
        it::TimeEvolver{<:FullUpdate};
        tol::Float64 = 0.0, H::Union{Nothing, LocalOperator} = nothing
    )
    time_start = time()
    check_convergence = (tol > 0)
    if check_convergence
        @assert (it.state.ψ isa InfinitePEPS) && it.alg.imaginary_time "Only imaginary time evolution of InfinitePEPS allows convergence checking."
        @assert H isa LocalOperator "Hamiltonian should be provided for convergence checking in full update."
    end
    time0 = time()
    iter0, t0 = it.state.iter, it.state.t
    ψ0, env0, info0 = it.state.ψ, it.state.env, nothing
    energy0 = check_convergence ?
        expectation_value(ψ0, H, ψ0, env0) / prod(size(ψ0)) : NaN
    if check_convergence
        @info "FU: initial state energy = $(energy0)."
    end
    for (ψ, env, info) in it
        !(it.state.reconverged) && continue
        # do the following only when env has been reconverged
        iter = it.state.iter
        energy = check_convergence ?
            expectation_value(ψ, H, ψ, env) / prod(size(ψ)) : NaN
        diff = energy - energy0
        stop = (iter == it.nstep) || (diff < 0 && abs(diff) < tol) || (diff > 0)
        showinfo = (iter == 1) || it.state.reconverged || stop
        time1 = time()
        if showinfo
            corner = env.corners[1, 1, 1]
            corner_dim = dim.(space(corner, ax) for ax in 1:numind(corner))
            @info "Dimension of env.corner[1, 1, 1] = $(corner_dim)."
            Δλ = (info0 === nothing) ? NaN : compare_weights(info.wts, info0.wts)
            if check_convergence
                @info @sprintf(
                    "FU iter %-6d: E = %.5f, ΔE = %.3e, |Δλ| = %.3e. Time: %.2f s",
                    it.state.iter, energy, diff, Δλ, time1 - time0
                )
            else
                @info @sprintf(
                    "FU iter %d: t = %.2e, |Δλ| = %.3e. Time: %.2f s",
                    it.state.iter, it.state.t, Δλ, time1 - time0
                )
            end
        end
        if check_convergence
            if (diff < 0 && abs(diff) < tol)
                @info "FU: energy has converged."
                return ψ, env, info
            end
            if diff > 0
                @warn "FU: energy has increased from last check. Abort evolution and return results from last check."
                # also reset internal state of `it` to last check
                it.state = FUState(iter0, t0, ψ0, env0, true)
                return ψ0, env0, info0
            end
            if iter == it.nstep
                @info "FU: energy has not converged."
                return ψ, env, info
            end
        end
        if stop
            time_end = time()
            @info @sprintf("Full update finished. Total time elasped: %.2f s", time_end - time_start)
            return ψ, env, info
        else
            # update backup variables
            iter0, t0 = it.state.iter, it.state.t
            ψ0, env0, info0, energy0 = ψ, env, info, energy
        end
        time0 = time()
    end
    return
end

"""
Full update an infinite PEPS with nearest neighbor Hamiltonian.
"""
function MPSKit.time_evolve(
        ψ₀::InfiniteState, H::LocalOperator, dt::Float64, nstep::Int,
        alg::FullUpdate, env₀::CTMRGEnv;
        tol::Float64 = 0.0, t₀::Float64 = 0.0
    )
    it = TimeEvolver(ψ₀, H, dt, nstep, alg, env₀; t₀)
    return time_evolve(it; tol, H)
end

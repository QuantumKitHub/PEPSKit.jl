"""
$(TYPEDEF)

Algorithm struct for full update (FU) of InfinitePEPS or InfinitePEPO.

## Fields

$(TYPEDFIELDS)

Reference: Physical Review B 92, 035142 (2015)
"""
@kwdef struct FullUpdate <: TimeEvolution
    "When true, fix gauge of bond environment"
    fixgauge::Bool = true
    "When true (or false), the Trotter gate is `exp(-H dt)` (or `exp(-iH dt)`)"
    imaginary_time::Bool = true
    "Number of iterations without fully reconverging the environment"
    reconverge_interval::Int = 5
    "Do column and row moves after updating every single bond,
    instaed of waiting until an entire row/column of bonds are updated"
    finer_env_update::Bool = false
    "Bond truncation algorithm after applying time evolution gate"
    opt_alg::Union{ALSTruncation, FullEnvTruncation} =
        FullEnvTruncation(; trunc = truncerror(; atol = 1.0e-10))
    "CTMRG algorithm to reconverge environment.
    Its `projector_alg` is also used for the fast update of 
    the environment after each FU iteration"
    ctm_alg::SequentialCTMRG = SequentialCTMRG(;
        tol = 1.0e-9, maxiter = 20, verbosity = 1,
        trunc = truncerror(; atol = 1.0e-10),
        projector_alg = :fullinfinite,
    )
end

"""
Internal state of full update algorithm
"""
struct FUState{S <: InfiniteState, E <: CTMRGEnv, N <: Number}
    "number of performed iterations"
    iter::Int
    "evolved time"
    t::N
    "PEPS/PEPO"
    ψ::S
    "CTMRG environment"
    env::E
    "whether the current environment is reconverged"
    reconverged::Bool
end

"""
    TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Number, nstep::Int, 
        alg::FullUpdate, env₀::CTMRGEnv; t₀::Number = 0.0
    )

Initialize a TimeEvolver with Hamiltonian `H` and full update `alg`, 
starting from the initial state `ψ₀` and CTMRG environment `env₀`.

- The initial time is specified by `t₀`.
"""
function TimeEvolver(
        ψ₀::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::FullUpdate, env₀::CTMRGEnv; t₀::Number = 0.0
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
    # fix gauge of bond environment
    if alg.fixgauge
        Z, a, b, (Linv, Rinv) = fixgauge_benv(Z, a, b)
        X, Y = _fixgauge_benvXY(X, Y, Linv, Rinv)
        benv = Z' * Z
        @debug "cond(L) = $(LinearAlgebra.cond(Linv)); cond(R): $(LinearAlgebra.cond(Rinv))"
        @debug "cond(benv) after gauge fix: $(LinearAlgebra.cond(benv))"
    else
        benv = Z' * Z
    end
    # apply gate
    a, s, b, = _apply_gate(a, b, gate, truncerror(; atol = 1.0e-15))
    # optimize a, b; s is already normalized
    a, s, b, info = bond_truncate(a, b, benv, alg.opt_alg)
    A, B = _qr_bond_undo(X, a, b, Y)
    normalize!(A, Inf)
    normalize!(B, Inf)
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
    colmove_alg = SequentialCTMRG()
    @reset colmove_alg.projector_alg = alg.ctm_alg.projector_alg
    env2 = deepcopy(env)
    for row′ in 1:Nr
        # update row order: 1, Nr, 2, Nr-1, 3, Nr-2, ...
        row = isodd(row′) ? ((row′ + 1) ÷ 2) : (Nr - (row′ ÷ 2) + 1)
        term = get_gateterm(gate, (CartesianIndex(row, col), CartesianIndex(row, col + 1)))
        wts_col[row], info = _fu_xbond!(state, term, env, row, col, alg)
        fid = min(fid, info.fid)
        if alg.finer_env_update
            network = isa(state, InfinitePEPS) ? InfiniteSquareNetwork(state) :
                InfiniteSquareNetwork(InfinitePEPS(state))
            # match bonds between updated state tensors and the environment
            env2, info = ctmrg_leftmove(col, network, env2, colmove_alg)
            env2, info = ctmrg_rightmove(_next(col, Nc), network, env2, colmove_alg)
            # update environment around the bond to be updated next
            env2, info = ctmrg_upmove(row, network, env2, colmove_alg)
            env2, info = ctmrg_downmove(row, network, env2, colmove_alg)
        end
    end
    if !alg.finer_env_update
        network = isa(state, InfinitePEPS) ? InfiniteSquareNetwork(state) :
            InfiniteSquareNetwork(InfinitePEPS(state))
        env2, info = ctmrg_leftmove(col, network, env, colmove_alg)
        env2, info = ctmrg_rightmove(_next(col, Nc), network, env2, colmove_alg)
    end
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

"""
Imaginary time evolution of InfinitePEPS with convergence checking
"""
function _time_evolve_gs(
        it::TimeEvolver{<:FullUpdate}, tol::Float64, H::LocalOperator
    )
    time_start = time()
    @assert (it.state.ψ isa InfinitePEPS) && it.alg.imaginary_time "Only imaginary time evolution of InfinitePEPS allows convergence checking."
    time0 = time()
    # backup variables
    iter0, t0 = it.state.iter, it.state.t
    ψ0, env0, info0 = it.state.ψ, it.state.env, nothing
    energy0 = expectation_value(ψ0, H, ψ0, env0) / prod(size(ψ0))
    @info "FU: initial state energy = $(energy0)."
    for (ψ, env, info) in it
        iter = it.state.iter
        if iter == 1
            # reconverge for the 1st step
            network = isa(ψ, InfinitePEPS) ? ψ : InfinitePEPS(ψ)
            env, = leading_boundary(env, network, it.alg.ctm_alg)
            # update internal state of `it`
            # TODO: more elegant to use `Accessors.@set`
            it.state = FUState(iter, it.state.t, it.state.ψ, env, true)
        end
        !(it.state.reconverged) && continue
        # do the following only when env has been reconverged
        energy = expectation_value(ψ, H, ψ, env) / prod(size(ψ))
        diff = energy - energy0
        stop = (iter == it.nstep) || (diff < 0 && abs(diff) < tol) || (diff > 0)
        showinfo = (iter == 1) || it.state.reconverged || stop
        time1 = time()
        if showinfo
            corner = env.corners[1, 1, 1]
            corner_dim = dim.(space(corner, ax) for ax in 1:numind(corner))
            @info "Dimension of env.corner[1, 1, 1] = $(corner_dim)."
            Δλ = (info0 === nothing) ? NaN : compare_weights(info.wts, info0.wts)
            @info @sprintf(
                "FU iter %-6d: E = %.5f, ΔE = %.3e, |Δλ| = %.3e. Time: %.2f s",
                it.state.iter, energy, diff, Δλ, time1 - time0
            )
        end
        if (diff < 0 && abs(diff) < tol)
            @info "FU: energy has converged."
        end
        if diff > 0
            @warn "FU: energy has increased from last check. Abort evolution and return results from last check."
            ψ, env, info, energy = ψ0, env0, info0, energy0
            # also reset internal state of `it` to last check
            it.state = FUState(iter0, t0, ψ0, env0, true)
        end
        if iter == it.nstep
            @info "FU: energy has not converged."
        end
        if stop
            @assert it.state.reconverged
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
Time evolution without convergence checking
"""
function _time_evolve(it::TimeEvolver{<:FullUpdate})
    time_start = time0 = time()
    info0 = nothing
    for (ψ, env, info) in it
        iter = it.state.iter
        !(it.state.reconverged) && continue
        # do the following only when env has been reconverged
        stop = (iter == it.nstep)
        showinfo = (iter == 1) || it.state.reconverged || stop
        time1 = time()
        if showinfo
            corner = env.corners[1, 1, 1]
            corner_dim = dim.(space(corner, ax) for ax in 1:numind(corner))
            @info "Dimension of env.corner[1, 1, 1] = $(corner_dim)."
            Δλ = (info0 === nothing) ? NaN : compare_weights(info.wts, info0.wts)
            @info @sprintf(
                "FU iter %d: t = %.2e, |Δλ| = %.3e. Time: %.2f s",
                it.state.iter, it.state.t, Δλ, time1 - time0
            )
        end
        if stop
            @assert it.state.reconverged
            time_end = time()
            @info @sprintf("Full update finished. Total time elasped: %.2f s", time_end - time_start)
            return ψ, env, info
        else
            info0 = info
        end
        time0 = time()
    end
    return
end

"""
    time_evolve(
        it::TimeEvolver{<:FullUpdate};
        tol::Float64 = 0.0, H::Union{Nothing, LocalOperator} = nothing
    )

Perform time evolution to the end of FullUpdate TimeEvolver `it`,
or until convergence of energy set by a positive `tol`.

- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only). The Hamiltonian `H` should also be provided to measure the energy.
    For other usages they should not be changed.
"""
function MPSKit.time_evolve(
        it::TimeEvolver{<:FullUpdate};
        tol::Float64 = 0.0, H::Union{Nothing, LocalOperator} = nothing
    )
    if tol > 0
        return _time_evolve_gs(it, tol, H)
    else
        @assert tol == 0
        return _time_evolve(it)
    end
end

"""
    MPSKit.time_evolve(
        ψ₀::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::FullUpdate, env₀::CTMRGEnv;
        tol::Float64 = 0.0, t₀::Number = 0.0
    ) -> (ψ, env, info)

Perform time evolution on the initial state `ψ₀` and initial environment `env₀`
with Hamiltonian `H`, using FullUpdate algorithm `alg`, time step `dt` for 
`nstep` number of steps. 

- Setting `tol > 0` enables convergence check (for imaginary time evolution of InfinitePEPS only).
    For other usages it should not be changed.
- Use `t₀` to specify the initial time of `ψ₀`.
- `info` is a NamedTuple containing information of the evolution, 
    including the time `info.t` evolved since `ψ₀`.
"""
function MPSKit.time_evolve(
        ψ₀::InfiniteState, H::LocalOperator, dt::Number, nstep::Int,
        alg::FullUpdate, env₀::CTMRGEnv;
        tol::Float64 = 0.0, t₀::Number = 0.0
    )
    it = TimeEvolver(ψ₀, H, dt, nstep, alg, env₀; t₀)
    return time_evolve(it; tol, H)
end

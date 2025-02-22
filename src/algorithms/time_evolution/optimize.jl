"""
    ALSTruncation

Algorithm struct for the alternating least square optimization step in full update. 
`tol` is the maximum fidelity change between two optimization steps.
"""
@kwdef struct ALSTruncation
    trscheme::TensorKit.TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-15
    check_interval::Int = 0
end

function _als_message(
    iter::Int,
    cost::Float64,
    fid::Float64,
    Δcost::Float64,
    Δfid::Float64,
    time_elapsed::Float64,
)
    return @sprintf(
        "%5d, fid = %.8e, Δfid = %.8e, time = %.2e s\n", iter, fid, Δfid, time_elapsed
    ) * @sprintf("      cost = %.3e,   Δcost/cost0 = %.3e", cost, Δcost)
end

"""
    bond_optimize(a::AbstractTensorMap{T,S,2,1}, b::AbstractTensorMap{T,S,1,2}, benv::BondEnv{T,S}, alg) where {T<:Number,S<:ElementarySpace}

After time-evolving the reduced tensors `a` and `b` connected by a bond, 
truncate the bond dimension using the bond environment tensor `benv`.
```
    ┌-----------------------┐
    |   ┌----┐              |
    └---|    |-- a† == b† --┘
        |benv|   ↑     ↑
    ┌---|    |-- a === b ---┐
    |   └----┘              |
    └-----------------------┘
```
The truncation algorithm `alg` can be either `FullEnvTruncation` or `ALSTruncation`. 
The index order of `a` or `b` is
```
        2
        ↑       a[1 2; 3]
    1 -a/b- 3   b[1; 2 3]
```
"""
function bond_optimize(
    a::AbstractTensorMap{T,S,2,1},
    b::AbstractTensorMap{T,S,1,2},
    benv::BondEnv{T,S},
    alg::ALSTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    @assert codomain(benv) == domain(benv)
    time00 = time()
    verbose = (alg.check_interval > 0)
    a2b2 = _combine_ab(a, b)
    # initialize truncated aR, bL
    a, s, b = tsvd(a2b2; trunc=alg.trscheme)
    s /= norm(s, Inf)
    Vtrunc = space(s, 1)
    a, b = absorb_s(a, s, b)
    #= temporarily reorder axes of a and b
            3
            ↑
        1 -a/b- 2
    =#
    a, b = permute(a, ((1, 3, 2), ())), permute(b, ((1, 3, 2), ()))
    ab = _combine_ab(a, b)
    # cost function will be normalized by initial value
    cost00 = cost_function_als(benv, ab, a2b2)
    fid = fidelity(benv, ab, a2b2)
    cost0, fid0, Δfid = cost00, fid, 0.0
    verbose && @info "ALS init" * _als_message(0, cost0, fid, NaN, NaN, 0.0)
    for iter in 1:(alg.maxiter)
        time0 = time()
        #= 
        Fixing `b`, the cost function can be expressed in the R, S tensors as
        ```
            f(a†,a) = a† Ra a - a† Sa - Sa† a + const
        ```
        `f` is minimized when
            ∂f/∂ā = Ra a - Sa = 0
        =#
        Ra = _tensor_Ra(benv, b)
        Sa = _tensor_Sa(benv, b, a2b2)
        a, info_a = _solve_ab(Ra, Sa, a)
        # Fixing `a`, solve for `b` from `Rb b = Sb`
        Rb = _tensor_Rb(benv, a)
        Sb = _tensor_Sb(benv, a, a2b2)
        b, info_b = _solve_ab(Rb, Sb, b)
        ab = _combine_ab(a, b)
        cost = cost_function_als(benv, ab, a2b2)
        fid = fidelity(benv, ab, a2b2)
        Δcost = abs(cost - cost0) / cost00
        Δfid = abs(fid - fid0)
        cost0, fid0 = cost, fid
        time1 = time()
        converge = (Δfid < alg.tol)
        cancel = (iter == alg.maxiter)
        showinfo =
            cancel || (verbose && (converge || iter == 1 || iter % alg.check_interval == 0))
        if showinfo
            message = _als_message(
                iter,
                cost,
                fid,
                Δcost,
                Δfid,
                time1 - ((cancel || converge) ? time00 : time0),
            )
            if converge
                @info "ALS conv" * message
            elseif cancel
                @warn "ALS cancel" * message
            else
                @info "ALS iter" * message
            end
        end
        converge && break
    end
    a, s, b = tsvd(_combine_ab(a, b); trunc=truncspace(Vtrunc))
    # normalize singular value spectrum
    s /= norm(s, Inf)
    return a, s, b, (; fid, Δfid)
end

function bond_optimize(
    a::AbstractTensorMap{T,S,2,1},
    b::AbstractTensorMap{T,S,1,2},
    benv::BondEnv{T,S},
    alg::FullEnvTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    @assert codomain(benv) == domain(benv)
    #= initialize bond matrix using QR as `Ra Lb`

            ↑    ↑               ↑               ↑
        --- a == b ---   ==>   - Qa - Ra == Rb - Qb -
    =#
    Qa, Ra = leftorth(a)
    Qb, Rb = leftorth(b, ((2, 3), (1,)))
    isdual(codomain(Ra, 1)) && twist!(Ra, 1)
    isdual(codomain(Rb, 1)) && twist!(Rb, 1)
    @tensor b0[-1; -2] := Ra[-1 1] * Rb[-2 1]
    #= initialize bond environment around `Ra Lb`

        ┌--------------------------------------┐
        |   ┌----┐                             |
        └---|    |- 1 - Qa†- -1   -2 - Qb†- 2 -┘
            |    |      ↑              ↑
            |benv|      5              6
            |    |      ↑              ↑
        ┌---|    |- 3 - Qa - -3   -4 - Qb - 4 -┐
        |   └----┘                             |
        └--------------------------------------┘
    =#
    @tensor benv2[-1 -2; -3 -4] := (
        benv[1 2; 3 4] * conj(Qa[1 5 -1]) * conj(Qb[6 2 -2]) * Qa[3 5 -3] * Qb[6 4 -4]
    )
    # optimize bond matrix
    u, s, vh, info = fullenv_truncate(b0, benv2, alg)
    # truncate a, b tensors with u, s, vh
    @tensor a[-1 -2; -3] := Qa[-1 -2 3] * u[3 -3]
    @tensor b[-1; -2 -3] := vh[-1 1] * Qb[-2 -3 1]
    return a, s, b, info
end

"""
$(TYPEDEF)

Algorithm struct for the alternating least square (ALS) optimization of a bond. 

## Fields

$(TYPEDFIELDS)

## Constructors

    ALSTruncation(; kwargs...)

The truncation algorithm can be constructed from the following keyword arguments:

* `trunc::TruncationStrategy`: SVD truncation strategy when initilizing the truncated tensors connected by the bond.
* `maxiter::Int=50` : Maximal number of ALS iterations.
* `tol::Float64=1e-9` : ALS converges when the relative change in bond SVD spectrum between two iterations is smaller than `tol`.
* `check_interval::Int=0` : Set number of iterations to print information. Output is suppressed when `check_interval <= 0`. 
"""
@kwdef struct ALSTruncation{T <: TruncationStrategy}
    trunc::T
    maxiter::Int = 50
    tol::Float64 = 1.0e-9
    check_interval::Int = 0
end

function _als_message(
        iter::Int, cost::Float64, fid::Float64, Δcost::Float64,
        Δfid::Float64, Δs::Float64, time_elapsed::Float64,
    )
    return @sprintf(
        "%5d, fid = %.8e, Δfid = %.8e, time = %.4f s\n", iter, fid, Δfid, time_elapsed
    ) * @sprintf("      cost = %.3e, Δcost/cost0 = %.3e, |Δs| = %.4e.", cost, Δcost, Δs)
end

"""
Initialize truncated bond tensors for 2-site ALS
"""
function _als_init_truncate(
        ket2::AbstractTensorMap{T, S, 2, 2}, trunc::TruncationStrategy
    ) where {T, S}
    a, s0, b = svd_trunc!(permute(ket2, ((1, 3), (4, 2)); copy = true); trunc)
    a, b = absorb_s(a, s0, b)
    # put b in MPS axis order
    b = permute(b, ((1, 2), (3,)))
    xs = [a, b]
    return xs, s0
end

"""
    bond_truncate(a::AbstractTensorMap{T,S,2,1}, b::AbstractTensorMap{T,S,1,2}, benv::BondEnv{T,S}, alg) -> U, S, V, info

After time-evolving the reduced tensors `a` and `b` connected by a bond, 
truncate the bond dimension using the bond environment tensor `benv`.
```
    ┌-----------------------┐
    |   ┌----┐              |
    └---|    |-- a === b ---┘
        |benv|   ↓     ↓
    ┌---|    |-- a† == b† --┐
    |   └----┘              |
    └-----------------------┘
```
The truncation algorithm `alg` can be either `FullEnvTruncation` or `ALSTruncation`. 
The index order of `a` or `b` is
```
    1 -a/b- 3
        ↓       a[1 2; 3]
        2       b[1; 2 3]
```
"""
function bond_truncate(
        a::AbstractTensorMap{T, S, 2, 1},
        b::AbstractTensorMap{T, S, 1, 2},
        benv::BondEnv{T, S},
        alg::ALSTruncation,
    ) where {T <: Number, S <: ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    @assert codomain(benv) == domain(benv)
    need_flip = isdual(space(b, 1))
    time00 = time()
    verbose = (alg.check_interval > 0)

    # untruncated things
    ket2 = _combine_ket(a, b)
    benv_ket2 = _benv_ket(benv, ket2)
    b22 = _als_norm(ket2, benv_ket2)

    # initialize truncated bond tensors and bond weight
    xs, s0 = _als_init_truncate(ket2, alg.trunc)

    # initialize ALS cache
    Rs = [_als_tensor_R(benv, xs, i) for i in 1:2]
    Ss = [_als_tensor_S(benv_ket2, xs, i) for i in 1:2]

    # cost function will be normalized by initial value
    cost00, fid = cost_function_als(Rs[1], Ss[1], xs[1], b22)
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()
        for (i, (Rx, Sx, x)) in enumerate(zip(Rs, Ss, xs))
            # TODO: option to use pinv
            xs[i], info_x = _solve_als(Rx, Sx, x)
            @debug "Bond truncation info $(i):" info_x
            # update R, S for the next site
            i_next = _next(i, 2)
            Rs[i_next] = _als_tensor_R(benv, xs, i_next)
            Ss[i_next] = _als_tensor_S(benv_ket2, xs, i_next)
        end
        # cost function and local fidelity
        cost, fid = cost_function_als(Rs[1], Ss[1], xs[1], b22)
        # TODO: replace with truncated svdvals (without calculating u, vh)
        _, s, _ = svd_trunc!(_combine_ket_for_svd(xs...); trunc = alg.trunc)
        # fidelity, cost and normalized bond-s change
        s_nrm = norm(s0, Inf)
        Δs = _singular_value_distance(s, s0) / s_nrm
        Δcost = abs(cost - cost0) / cost00
        Δfid = abs(fid - fid0)
        cost0, fid0, s0 = cost, fid, s
        time1 = time()
        converge = (Δs < alg.tol)
        cancel = (iter == alg.maxiter)
        showinfo =
            cancel || (verbose && (converge || iter == 1 || iter % alg.check_interval == 0))
        if showinfo
            message = _als_message(
                iter, cost, fid, Δcost, Δfid, Δs,
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
    a, s, b = svd_trunc!(_combine_ket_for_svd(xs...); trunc = alg.trunc)
    a, b = absorb_s(a, s, b)
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    return a, s, b, (; fid, Δfid, Δs)
end

function bond_truncate(
        a::AbstractTensorMap{T, S, 2, 1},
        b::AbstractTensorMap{T, S, 1, 2},
        benv::BondEnv{T, S},
        alg::FullEnvTruncation,
    ) where {T <: Number, S <: ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    @assert codomain(benv) == domain(benv)
    need_flip = isdual(space(b, 1))
    #= initialize bond matrix using QR as `Ra Lb`

        --- a == b ---   ==>   - Qa ← Ra == Rb ← Qb -
            ↓    ↓               ↓               ↓
    =#
    Qa, Ra = left_orth(a; positive = true)
    Rb, Qb = right_orth(b; positive = true)
    @assert !isdual(space(Ra, 1)) && !isdual(space(Qb, 1))
    @tensor b0[-1; -2] := Ra[-1 1] * Rb[1 -2]
    #= initialize bond environment around `Ra Lb`

        ┌--------------------------------------┐
        |   ┌----┐                             |
        └---|    |- 3 - Qa - -3   -4 - Qb - 4 -┘
            |    |      ↓              ↓
            |benv|      5              6
            |    |      ↓              ↓
        ┌---|    |- 1 - Qa†- -1   -2 - Qb†- 2 -┐
        |   └----┘                             |
        └--------------------------------------┘
    =#
    @tensor benv2[-1 -2; -3 -4] := (
        benv[1 2; 3 4] * conj(Qa[1 5 -1]) * conj(Qb[-2 6 2]) * Qa[3 5 -3] * Qb[-4 6 4]
    )
    # optimize bond matrix
    u, s, vh, info = fullenv_truncate(b0, benv2, alg)
    u, vh = absorb_s(u, s, vh)
    # truncate a, b tensors with u, s, vh
    @tensor a[-1 -2; -3] := Qa[-1 -2 3] * u[3 -3]
    @tensor b[-1; -2 -3] := vh[-1 1] * Qb[1 -2 -3]
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    return a, s, b, info
end

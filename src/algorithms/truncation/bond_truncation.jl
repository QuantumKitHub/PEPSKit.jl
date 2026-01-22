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
@kwdef struct ALSTruncation
    trunc::TruncationStrategy
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
    bond_truncate(a::MPSBondTensor, b::MPSBondTensor,benv::BondEnv, alg)

After time-evolving the reduced tensors `a` and `b` connected by a bond, 
truncate the bond dimension using the bond environment tensor `benv`.
```
    ┌-----------------------┐
    |   ┌----┐              |
    └---|    |-- a === b ---┘
        |benv|
    ┌---|    |-- a† == b† --┐
    |   └----┘              |
    └-----------------------┘
```
 The truncation algorithm `alg` can be either `FullEnvTruncation` or `ALSTruncation`. 
"""
function bond_truncate(
        a::MPSBondTensor, b::MPSBondTensor,
        benv::BondEnv, alg::ALSTruncation
    )
    @assert codomain(benv) == domain(benv)
    need_flip = isdual(space(b, 1))
    time00 = time()
    verbose = (alg.check_interval > 0)
    a2b2 = _combine_ab(a, b)
    # initialize truncated a, b
    # TODO: initial truncation with effect of `benv`
    a, s0, b = svd_trunc(a2b2; trunc = alg.trunc)
    a, b = absorb_s(a, s0, b)
    ab = _combine_ab(a, b)
    # cost function will be normalized by initial value
    cost00 = cost_function_als(benv, ab, a2b2)
    fid = fidelity(benv, ab, a2b2)
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)
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
        @debug "Bond truncation info" info_a info_b
        ab = _combine_ab(a, b)
        cost = cost_function_als(benv, ab, a2b2)
        fid = fidelity(benv, ab, a2b2)
        # TODO: replace with truncated svdvals (without calculating u, vh)
        _, s, _ = svd_trunc!(ab; trunc = alg.trunc)
        # fidelity, cost and normalized bond-s change
        s_nrm = norm(s0, Inf)
        Δs = ((space(s) == space(s0)) ? _singular_value_distance((s, s0)) : NaN) / s_nrm
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
    a, s, b = svd_trunc!(_combine_ab(a, b); trunc = alg.trunc)
    a, b = absorb_s(a, s, b)
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    return a, s, b, (; fid, Δfid, Δs)
end

function bond_truncate(
        a::MPSBondTensor, b::MPSBondTensor,
        benv::BondEnv, alg::FullEnvTruncation
    )
    @assert codomain(benv) == domain(benv)
    need_flip = isdual(space(b, 1))
    # optimize bond matrix
    ab0 = _combine_ab(a, b)
    u, s, vh, info = fullenv_truncate(ab0, benv, alg)
    a, b = absorb_s(u, s, vh)
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    return a, s, b, info
end

"""
Initialize truncated bond tensors for 3-site ALS
"""
function _als3_init_truncate(
        Ms::Vector{T}, trunc::TruncationStrategy
    ) where {T <: GenericMPSTensor}
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    xs = copy.(Ms)
    _flip_virtuals!(xs, flips)
    wts0, _, Pas, Pbs = _cluster_truncate!(xs, fill(trunc, 2))
    return xs, (Pbs[1], Pas[2]), wts0, flips
end

"""
Truncate bonds in the 3-site cluster `Ms = [a, m, b]` using
the environment (norm tensor) `benv` surrounding `Ms` as
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a---m---b---┤
    |   ↓   ↓   ↓   |
    ├---ā---m̄---b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
`m` is the tensor at the middle site, while `a`, `b` are
reduced bond tensor from the first and the last site.

Input tensors `Ms = [a, m, b]` have MPS axis order along
the southeast 3-site cluster.

Reference: Phys. Rev. B 97, 174408 (2018)
"""
function se3site_truncate(
        Ms::Vector{T}, benv::BondEnv3site, alg::ALSTruncation
    ) where {T <: GenericMPSTensor}
    # dual check
    @assert length(Ms) == 3
    time00 = time()
    verbose = (alg.check_interval > 0)

    # untruncated things
    ket2 = _combine_ket(Ms...)
    benv_ket2 = _benv_ket(benv, ket2)
    b22 = real(_als_norm(ket2, benv_ket2))

    # initialize truncated bond tensors
    xs, _, wts0, flips = _als3_init_truncate(Ms, alg.trunc)

    # initialize ALS cache
    Rs = [_als_tensor_R(benv, xs, i) for i in 1:3]
    Ss = [_als_tensor_S(benv_ket2, xs, i) for i in 1:3]

    # initial cost and fidelity
    cost00, fid = cost_function_als(Rs[1], Ss[1], xs[1], b22)
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS3 init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()
        for (i, (Rx, Sx, x)) in enumerate(zip(Rs, Ss, xs))
            xs[i] = _solve_als_pinv(Rx, Sx)
            # @debug "Bond truncation info $(i):" info_x
            # update R, S for the next site
            i_next = _next(i, 3)
            Rs[i_next] = _als_tensor_R(benv, xs, i_next)
            Ss[i_next] = _als_tensor_S(benv_ket2, xs, i_next)
        end
        # compare cost, fidelity, bond weights
        cost, fid = cost_function_als(Rs[1], Ss[1], xs[1], b22)
        wts = _get_allprojs(xs, fill(notrunc(), 2))[3]
        Δcost = abs(cost - cost0) / cost00
        Δfid = abs(fid - fid0)
        Δs = mean(
            _singular_value_distance(s, s0) for (s, s0) in zip(wts, wts0)
        ) / norm(wts0[1], Inf)
        cost0, fid0, wts0 = cost, fid, wts
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
                @info "ALS3 conv" * message
            elseif cancel
                @warn "ALS3 cancel" * message
            else
                @info "ALS3 iter" * message
            end
        end
        converge && break
    end
    # convert to Vidal gauge at the end
    wts, = _cluster_truncate!(xs, fill(notrunc(), 2))
    # restore virtual arrows
    _flip_virtuals!(xs, flips)
    return xs, wts, (; fid, Δfid, Δs)
end

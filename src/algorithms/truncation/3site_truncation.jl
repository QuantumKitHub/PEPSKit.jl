@kwdef struct ALS3SiteTruncation{T}
    trunc::T
    maxiter::Int = 20
    inneriter::Int = 3
    fidtol::Float64 = 1.0e-11
    check_interval::Int = 0
end

function _als3_message(
        iter::Int, cost::Float64, fid::Float64, Δcost::Float64,
        Δfid::Float64, time_elapsed::Float64,
    )
    return @sprintf(
        "%5d, fid = %.8e, Δfid = %.8e, time = %.4f s\n", iter, fid, Δfid, time_elapsed
    ) * @sprintf("      cost = %.3e, Δcost/cost0 = %.3e.", cost, Δcost)
end

"""
Initialize truncated bond tensors for 3-site ALS
"""
function _als3_init_truncate(
        Ms::Vector{<:GenericMPSTensor}, trunc::TruncationStrategy
    )
    return _als3_init_truncate(Ms, fill(trunc, 2))
end
function _als3_init_truncate(
        Ms::Vector{<:GenericMPSTensor}, truncs::Vector{<:TruncationStrategy}
    )
    flips = [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)]
    xs = _flip_virtuals!(copy.(Ms), flips)
    wts0, _, Pas, Pbs = _cluster_truncate!(xs, truncs)
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
        Ms::Vector{T}, benv::BondEnv3site, alg::ALS3SiteTruncation
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
    xs, _, _, flips = _als3_init_truncate(Ms, alg.trunc)

    # initial cost and fidelity
    R1 = _als3s_tensor_R1(benv, xs[2], xs[3])
    S1 = _als3s_tensor_S1(benv_ket2, xs[2], xs[3])
    cost00, fid = cost_function_als(R1, S1, xs[1], b22)
    cost0, fid0, Δcost, Δfid = cost00, fid, NaN, NaN
    verbose && @info "ALS3 init" * _als3_message(0, cost0, fid, Δcost, Δfid, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()
        # optimize m
        R2 = _als3s_tensor_R2(benv, xs[1], xs[3])
        S2 = _als3s_tensor_S2(benv_ket2, xs[1], xs[3])
        xs[2] = _solve_als_pinv(R2, S2)
        # optimize a, b more frequently
        for _ in 1:alg.inneriter
            R3 = _als3s_tensor_R3(benv, xs[1], xs[2])
            S3 = _als3s_tensor_S3(benv_ket2, xs[1], xs[2])
            xs[3] = _solve_als_pinv(R3, S3)
            R1 = _als3s_tensor_R1(benv, xs[2], xs[3])
            S1 = _als3s_tensor_S1(benv_ket2, xs[2], xs[3])
            xs[1] = _solve_als_pinv(R1, S1)
        end
        # compare cost, fidelity, bond weights
        cost, fid = cost_function_als(R1, S1, xs[1], b22)
        Δcost = abs(cost - cost0) / cost00
        Δfid = abs(fid - fid0)
        cost0, fid0 = cost, fid
        time1 = time()
        converge = (Δfid < alg.fidtol)
        cancel = (iter == alg.maxiter)
        showinfo =
            cancel || (verbose && (converge || iter == 1 || iter % alg.check_interval == 0))
        if showinfo
            message = _als3_message(
                iter, cost, fid, Δcost, Δfid,
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
    for (i, (wt, fl)) in enumerate(zip(wts, flips))
        fl && (wts[i] = _fliptwist_s(wt))
    end
    return xs, wts, (; fid, Δfid)
end

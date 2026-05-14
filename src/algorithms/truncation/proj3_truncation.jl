@kwdef struct ALSProjTruncation{T <: TruncationStrategy}
    trunc::T
    maxiter::Int = 50
    inneriter::Int = 8
    tol::Float64 = 1.0e-9
    check_interval::Int = 0
end

function se3site_truncate(
        Ms::Vector{T}, benv::BondEnv3site, alg::ALSProjTruncation
    ) where {T <: GenericMPSTensor}
    # dual check
    @assert length(Ms) == 3
    time00 = time()
    verbose = (alg.check_interval > 0)
    @assert alg.inneriter > 0

    # untruncated things
    ket2 = _combine_ket(Ms...)
    benv_ket2 = _benv_ket(benv, ket2)
    b22 = real(_als_norm(ket2, benv_ket2))

    # initialize truncated bond tensors
    Ms = copy.(Ms)
    _flip_virtuals!(Ms, [isdual(space(M, 1)) for M in Iterators.drop(Ms, 1)])
    xs, (q, u), wts0, flips = _als3_init_truncate(Ms, alg.trunc)

    # initial cost and fidelity
    cost00, fid = cost_function_als(
        _als_tensor_R(benv, xs, 1),
        _als_tensor_S(benv_ket2, xs, 1), xs[1], b22
    )
    cost0, fid0, Δcost, Δfid, Δs = cost00, fid, NaN, NaN, NaN
    verbose && @info "ALS3 init" * _als_message(0, cost0, fid, Δcost, Δfid, Δs, 0.0)

    for iter in 1:(alg.maxiter)
        time0 = time()
        local benv_ab, benv_ket_ab

        # optimize on first bond
        benv_b = _benv_b(benv, xs[3])
        benv_ket_b = _benv_ket_b(benv_ket2, xs[3])
        mu = _apply_proj(Ms[2], u)
        for _ in 1:alg.inneriter
            # optimize xs[1]
            R = _als_tensor_Rp(benv_b, xs[2])
            S = _als_tensor_Sp(benv_ket_b, xs[2])
            xs[1] = _solve_als_pinv(R, S)
            # optimize q and xs[2]
            benv_ab = _benv_a(benv_b, xs[1])
            benv_ket_ab = _benv_ket_ba(benv_ket_b, xs[1])
            R = _als_tensor_Rq(benv_ab, mu)
            S = _als_tensor_Sq(benv_ket_ab, mu)
            q = _solveproj_als_pinv(R, S)
            xs[2] = _apply_proj(q, mu)
        end

        # optimize on second bond
        benv_a = _benv_a(benv, xs[1])
        benv_ket_a = _benv_ket_a(benv_ket2, xs[1])
        qm = _apply_proj(q, Ms[2])
        for _ in 1:alg.inneriter
            # optimize u and xs[2]
            benv_ab = _benv_b(benv_a, xs[3])
            benv_ket_ab = _benv_ket_ab(benv_ket_a, xs[3])
            R = _als_tensor_Ru(benv_ab, qm)
            S = _als_tensor_Su(benv_ket_ab, qm)
            u = _solveproj_als_pinv(R, S)
            xs[2] = _apply_proj(qm, u)
            # optimize xs[3]
            R = _als_tensor_Rv(benv_a, xs[2])
            S = _als_tensor_Sv(benv_ket_a, xs[2])
            xs[3] = _solve_als_pinv(R, S)
        end

        # compare cost, fidelity, bond weights
        cost, fid = cost_function_als(
            _als_tensor_R(benv, xs, 1),
            _als_tensor_S(benv_ket2, xs, 1), xs[1], b22
        )
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

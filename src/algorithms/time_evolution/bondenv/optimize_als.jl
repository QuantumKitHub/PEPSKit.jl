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

"""
Solve the equations
```
    Ra a = Sa, Rb b = Sb
```
"""
function _solve_ab(
    tR::AbstractTensorMap{T,S,2,2}, tS::AbstractTensor{T,S,3}, ab0::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    f(x) = (@tensor tS2[:] := tR[-1 -2 1 2] * x[1 2 -3])
    ab, info = linsolve(f, tS, permute(ab0, (1, 3, 2)), 0, 1)
    return permute(ab, (1, 3, 2)), info
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

function bond_optimize(
    benv::BondEnv{T,S},
    a::AbstractTensor{T,S,3},
    b::AbstractTensor{T,S,3},
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
    a, s, b = tsvd(a2b2, ((1, 2), (3, 4)); trunc=alg.trscheme)
    s /= norm(s, Inf)
    Vtrunc = space(s, 1)
    a, b = absorb_s(a, s, b)
    a, b = permute(a, (1, 2, 3)), permute(b, (1, 2, 3))
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
            verbose && (converge || cancel || iter == 1 || iter % alg.check_interval == 0)
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
    ab = _combine_ab(a, b)
    a, s, b = tsvd(ab, ((1, 2), (3, 4)); trunc=truncspace(Vtrunc), alg=TensorKit.SVD())
    # normalize singular value spectrum
    s /= norm(s, Inf)
    return a, s, b, (; fid, Δfid)
end

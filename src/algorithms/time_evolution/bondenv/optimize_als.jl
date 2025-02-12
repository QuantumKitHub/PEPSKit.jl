"""
    ALSTruncation

Algorithm struct for the alternating least square optimization step in full update. 
`tol` is the maximum `|fid_{n+1} - fid_{n}|` 
(fidelity change between two optimization steps)
"""
@kwdef struct ALSTruncation
    trscheme::TensorKit.TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-15
    check_int::Int = 0
end

"""
Construct the tensor
```
    |-----------benv-----------|
    |- DX1     Db1 - b† - DY1 -|
    |                ↑         |
    |                db        |
    |                ↑         |
    |- DX0     Db0 - b -- DY0 -|
    |--------------------------|
```
"""
function tensor_Ra(
    benv::BondEnv{T,S}, b::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Ra[DX1, Db1; DX0, Db0] := (
        benv[DX1, DY1, DX0, DY0] * b[Db0, db, DY0] * conj(b[Db1, db, DY1])
    )
end

"""
Construct the tensor
```
    |----------benv-----------|
    |- DX1   Db1 -- b† - DY1 -|
    |               ↑         |
    |         da    db        |
    |         ↑     ↑         |
    |- DX0 -- a2    b2 - DY0 -|
    |-------------------------|
```
"""
function tensor_Sa(
    benv::BondEnv{T,S}, b::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sa[DX1, Db1, da] := (
        benv[DX1, DY1, DX0, DY0] * conj(b[Db1, db, DY1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Construct the tensor
```
    |-----------benv-----------|
    |- DX1 - a† - Da1     DY1 -|
    |        ↑                 |
    |        da                |
    |        ↑                 |
    |- DX0 - a -- Da0     DY0 -|
    |--------------------------|
```
"""
function tensor_Rb(
    benv::BondEnv{T,S}, a::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Rb[Da1, DY1; Da0, DY0] := (
        benv[DX1, DY1, DX0, DY0] * a[DX0, da, Da0] * conj(a[DX1, da, Da1])
    )
end

"""
Construct the tensor
```
    |----------benv-----------|
    |- DX1 -- a† - Da1   DY1 -|
    |         ↑               |
    |         da   db         |
    |         ↑     ↑         |
    |- DX0 -- a2   b2 -- DY0 -|
    |-------------------------|
```
"""
function tensor_Sb(
    benv::BondEnv{T,S}, a::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sb[Da1, DY1, db] := (
        benv[DX1, DY1, DX0, DY0] * conj(a[DX1, da, Da1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Calculate the inner product <a1,b1|a2,b2>
```
    |---------benv----------|
    |- DX1 - (a1 b1)†- DY1 -|
    |        ↑    ↑         |
    |        da   db        |
    |        ↑    ↑         |
    |- DX0 - (a2 b2) - DY0 -|
    |-----------------------|
```
"""
function inner_prod(
    benv::BondEnv{T,S}, a1b1::AbstractTensor{T,S,4}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor benv[DX1, DY1, DX0, DY0] *
        conj(a1b1[DX1, da, db, DY1]) *
        a2b2[DX0, da, db, DY0]
end

"""
Calculate the fidelity between two evolution steps
```
        |⟨a1,b1|a2,b2⟩|^2
    --------------------------
    ⟨a1,b1|a1,b1⟩⟨a2,b2|a2,b2⟩
```
"""
function fidelity(
    benv::BondEnv{T,S}, a1b1::AbstractTensor{T,S,4}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    b12 = inner_prod(benv, a1b1, a2b2)
    b11 = inner_prod(benv, a1b1, a1b1)
    b22 = inner_prod(benv, a2b2, a2b2)
    return abs2(b12) / abs(b11 * b22)
end

"""
Contract the axis between `a` and `b` tensors
```
            da      db
            ↑       ↑
    -- DX - a - D - b - DY --
```
"""
function _combine_ab(
    a::AbstractTensor{T,S,3}, b::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @tensor ab[DX, da, db, DY] := a[DX, da, D] * b[D, db, DY]
end

"""
Calculate the cost function
```
    f(a,b)  = ‖ |a1,b1⟩ - |a2,b2⟩ ‖^2
    = ⟨a1,b1|a1,b1⟩ - 2 Re⟨a1,b1|a2,b2⟩ + ⟨a2,b2|a2,b2⟩
```
"""
function cost_function_als(
    benv::BondEnv{T,S}, aR1bL1::AbstractTensor{T,S,4}, aR2bL2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    t1 = inner_prod(benv, aR1bL1, aR1bL1)
    t2 = inner_prod(benv, aR2bL2, aR2bL2)
    t3 = inner_prod(benv, aR1bL1, aR2bL2)
    return real(t1) + real(t2) - 2 * real(t3)
end

"""
Solving the equations
```
    Ra a = Sa, Rb b = Sb
```
"""
function solve_ab(
    tR::AbstractTensorMap{T,S,2,2}, tS::AbstractTensor{T,S,3}, ab0::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    f(x) = ncon((tR, x), ([-1 -2 1 2], [1 2 -3]))
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
    time00 = time()
    verbose = (alg.check_int > 0)
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
    # only optimize when fidelity differ from 1 by larger than 1e-12
    (abs(fid0 - 1) > 1e-12) && for iter in 1:(alg.maxiter)
        time0 = time()
        #= 
        Fixing `b`, the cost function can be expressed in the R, S tensors as
        ```
            f(a†,a) = a† Ra a - a† Sa - Sa† a + const
        ```
        `f` is minimized when
            ∂f/∂ā = Ra a - Sa = 0
        =#
        Ra = tensor_Ra(benv, b)
        Sa = tensor_Sa(benv, b, a2b2)
        a, info_a = solve_ab(Ra, Sa, a)
        # Fixing `a`, solve for `b` from `Rb b = Sb`
        Rb = tensor_Rb(benv, a)
        Sb = tensor_Sb(benv, a, a2b2)
        b, info_b = solve_ab(Rb, Sb, b)
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
            verbose && (converge || cancel || iter == 1 || iter % alg.check_int == 0)
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

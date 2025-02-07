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
    |------------env-----------|
    |- DX1     Db1 - b† - DY1 -|
    |                ↑         |
    |                db        |
    |                ↑         |
    |- DX0     Db0 - b -- DY0 -|
    |--------------------------|
```
"""
function tensor_Ra(
    env::BondEnv{T,S}, b::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Ra[DX1, Db1; DX0, Db0] := (
        env[DX1, DY1, DX0, DY0] * b[Db0, db, DY0] * conj(b[Db1, db, DY1])
    )
end

"""
Construct the tensor
```
    |-----------env-----------|
    |- DX1   Db1 -- b† - DY1 -|
    |               ↑         |
    |         da    db        |
    |         ↑     ↑         |
    |- DX0 -- a2    b2 - DY0 -|
    |-------------------------|
```
"""
function tensor_Sa(
    env::BondEnv{T,S}, b::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sa[DX1, Db1, da] := (
        env[DX1, DY1, DX0, DY0] * conj(b[Db1, db, DY1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Construct the tensor
```
    |------------env-----------|
    |- DX1 - a† - Da1     DY1 -|
    |        ↑                 |
    |        da                |
    |        ↑                 |
    |- DX0 - a -- Da0     DY0 -|
    |--------------------------|
```
"""
function tensor_Rb(
    env::BondEnv{T,S}, a::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Rb[Da1, DY1; Da0, DY0] := (
        env[DX1, DY1, DX0, DY0] * a[DX0, da, Da0] * conj(a[DX1, da, Da1])
    )
end

"""
Construct the tensor
```
    |-----------env-----------|
    |- DX1 -- a† - Da1   DY1 -|
    |         ↑               |
    |         da   db         |
    |         ↑     ↑         |
    |- DX0 -- a2   b2 -- DY0 -|
    |-------------------------|
```
"""
function tensor_Sb(
    env::BondEnv{T,S}, a::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sb[Da1, DY1, db] := (
        env[DX1, DY1, DX0, DY0] * conj(a[DX1, da, Da1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Calculate the inner product <a1,b1|a2,b2>
```
    |----------env----------|
    |- DX1 - (a1 b1)†- DY1 -|
    |        ↑    ↑         |
    |        da   db        |
    |        ↑    ↑         |
    |- DX0 - (a2 b2) - DY0 -|
    |-----------------------|
```
"""
function inner_prod(
    env::BondEnv{T,S}, a1b1::AbstractTensor{T,S,4}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor env[DX1, DY1, DX0, DY0] *
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
    env::BondEnv{T,S}, a1b1::AbstractTensor{T,S,4}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    b12 = inner_prod(env, a1b1, a2b2)
    b11 = inner_prod(env, a1b1, a1b1)
    b22 = inner_prod(env, a2b2, a2b2)
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
function cost_func(
    env::BondEnv{T,S}, aR1bL1::AbstractTensor{T,S,4}, aR2bL2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    t1 = inner_prod(env, aR1bL1, aR1bL1)
    t2 = inner_prod(env, aR2bL2, aR2bL2)
    t3 = inner_prod(env, aR1bL1, aR2bL2)
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

function bond_optimize(
    env::BondEnv{T,S},
    a::AbstractTensor{T,S,3},
    b::AbstractTensor{T,S,3},
    alg::ALSTruncation,
) where {T<:Number,S<:ElementarySpace}
    # dual check of physical index
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    verbose = (alg.check_int > 0)
    if verbose
        @info @sprintf(
            "%-4s%12s%12s%12s%12s %10s\n",
            "ALS iter",
            "Cost",
            "Fidelity",
            "ϵ_cost",
            "Δfid",
            "Time/s"
        )
    end
    time0 = time()
    a2b2 = _combine_ab(a, b)
    # initialize truncated aR, bL
    a, s, b = tsvd(a2b2, ((1, 2), (3, 4)); trunc=alg.trscheme)
    s /= norm(s, Inf)
    Vtrunc = space(s, 1)
    a, b = absorb_s(a, s, b)
    a, b = permute(a, (1, 2, 3)), permute(b, (1, 2, 3))
    ab = _combine_ab(a, b)
    # cost function is normalized by initial value
    cost00 = cost_func(env, ab, a2b2)
    fid00 = fidelity(env, ab, a2b2)
    cost0, fid0, fid, diff_fid = cost00, fid00, 0.0, 0.0
    # no need to further optimize
    if abs(cost0) < 5e-15
        time1 = time()
        if verbose
            @info @sprintf(
                "%-4d%12.3e%12.3e%12.3e%12.3e %10.3e\n",
                0,
                cost0,
                fid0,
                NaN,
                NaN,
                time1 - time0
            )
        end
    else
        for count in 1:(alg.maxiter)
            time0 = time()
            #= 
            Fixing `b`, the cost function can be expressed in the R, S tensors as
            ```
                f(a†,a) = a† Ra a - a† Sa - Sa† a + const
            ```
            `f` is minimized when
                ∂f/∂ā = Ra a - Sa = 0
            =#
            Ra = tensor_Ra(env, b)
            Sa = tensor_Sa(env, b, a2b2)
            a, info_a = solve_ab(Ra, Sa, a)
            # Fixing `a`, solve for `b` from `Rb b = Sb`
            Rb = tensor_Rb(env, a)
            Sb = tensor_Sb(env, a, a2b2)
            b, info_b = solve_ab(Rb, Sb, b)
            ab = _combine_ab(a, b)
            cost = cost_func(env, ab, a2b2)
            fid = fidelity(env, ab, a2b2)
            diff_cost = abs(cost - cost0) / cost00
            diff_fid = abs(fid - fid0)
            time1 = time()
            if verbose && (count == 1 || count % alg.check_int == 0)
                @info @sprintf(
                    "%-4d%12.3e%12.3e%12.3e%12.3e %10.3e\n",
                    count,
                    cost,
                    fid,
                    diff_cost,
                    diff_fid,
                    time1 - time0
                )
            end
            cost0, fid0 = cost, fid
            if diff_fid < alg.tol
                break
            end
            aR0, bL0 = deepcopy(a), deepcopy(b)
            if count == alg.maxiter
                @warn "Warning: max iter $(alg.maxiter) reached for ALS optimization\n"
            end
        end
    end
    ab = _combine_ab(a, b)
    a, s, b = tsvd(ab, ((1, 2), (3, 4)); trunc=truncspace(Vtrunc))
    # normalize singular value spectrum
    s /= norm(s, Inf)
    return a, s, b, (; fid, diff_fid)
end

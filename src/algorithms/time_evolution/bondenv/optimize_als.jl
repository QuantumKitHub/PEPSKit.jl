"""
Algorithm struct for the alternating least square optimization step in full update. 
`tol` is the maximum `|fid_{n+1} - fid_{n}| / fid_0` 
(normalized local fidelity change between two optimization steps)
"""
@kwdef struct ALSTruncation
    trscheme::TensorKit.TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-15
    verbose::Bool = false
    check_int::Int = 1
end

"""
Construct the tensor
```
    |------------env------------|
    |→ DX1     Db1 → bL† ← DY1 ←|
    |                ↑          |
    |                db         |
    |                ↑          |
    |← DX0     Db0 ← bL -→ DY0 →|
    |---------------------------|
```
"""
function tensor_Ra(env::BondEnv{S}, bL::AbstractTensor{S,3}) where {S<:ElementarySpace}
    return @autoopt @tensor Ra[DX1, Db1, DX0, Db0] := (
        env[DX1, DY1, DX0, DY0] * bL[Db0, db, DY0] * conj(bL[Db1, db, DY1])
    )
end

"""
Construct the tensor
```
    |-----------env-----------|
    |→ DX1   Db1 → bL† ← DY1 ←|
    |               ↑         |
    |         da    db        |
    |         ↑     ↑         |
    |← DX0 ←- aR2 bL2 -→ DY0 →|
    |-------------------------|
```
"""
function tensor_Sa(
    env::BondEnv{S}, bL::AbstractTensor{S,3}, aR2bL2::AbstractTensor{S,4}
) where {S<:ElementarySpace}
    return @autoopt @tensor Sa[DX1, Db1, da] := (
        env[DX1, DY1, DX0, DY0] * conj(bL[Db1, db, DY1]) * aR2bL2[DX0, da, db, DY0]
    )
end

"""
Construct the tensor
```
    |------------env------------|
    |→ DX1 → aR† → Da1     DY1 ←|
    |        ↑                  |
    |        da                 |
    |        ↑                  |
    |← DX0 ← aR ←- Da0     DY0 →|
    |---------------------------|
```
"""
function tensor_Rb(env::BondEnv{S}, aR::AbstractTensor{S,3}) where {S<:ElementarySpace}
    return @autoopt @tensor Rb[Da1, DY1, Da0, DY0] := (
        env[DX1, DY1, DX0, DY0] * aR[DX0, da, Da0] * conj(aR[DX1, da, Da1])
    )
end

"""
Construct the tensor
```
    |-----------env-----------|
    |→ DX1 → aR† → Da1   DY1 ←|
    |         ↑               |
    |         da   db         |
    |         ↑     ↑         |
    |← DX0 ←- aR2 bL2 -→ DY0 →|
    |-------------------------|
```
"""
function tensor_Sb(
    env::BondEnv{S}, aR::AbstractTensor{S,3}, aR2bL2::AbstractTensor{S,4}
) where {S<:ElementarySpace}
    return @autoopt @tensor Sb[Da1, DY1, db] := (
        env[DX1, DY1, DX0, DY0] * conj(aR[DX1, da, Da1]) * aR2bL2[DX0, da, db, DY0]
    )
end

"""
Calculate the inner product <a1,b1|a2,b2>
```
    |----------env----------|
    |→ DX1 → aR1bL1† ← DY1 ←|
    |        ↑    ↑         |
    |        da   db        |
    |        ↑    ↑         |
    |← DX0 ← aR2bL2 → DY0 -→|
    |-----------------------|
```
"""
function inner_prod(
    env::BondEnv{S}, aR1bL1::AbstractTensor{S,4}, aR2bL2::AbstractTensor{S,4}
) where {S<:ElementarySpace}
    return @autoopt @tensor env[DX1, DY1, DX0, DY0] *
        conj(aR1bL1[DX1, da, db, DY1]) *
        aR2bL2[DX0, da, db, DY0]
end

"""
Calculate the fidelity using aR, bL
between two evolution steps
```
            |<aR1 bL1 | aR2 bL2>|^2
    ----------------------------------------
    <aR1 bL1 | aR1 bL1> <aR2 bL2 | aR2 bL2>
```
"""
function fidelity(
    env::BondEnv{S}, aR1bL1::AbstractTensor{S,4}, aR2bL2::AbstractTensor{S,4}
) where {S<:ElementarySpace}
    b12 = inner_prod(env, aR1bL1, aR2bL2)
    b11 = inner_prod(env, aR1bL1, aR1bL1)
    b22 = inner_prod(env, aR2bL2, aR2bL2)
    return abs2(b12) / abs(b11 * b22)
end

"""
Contract the axis between `aR` and `bL` tensors
"""
function _combine_aRbL(
    aR::AbstractTensor{S,3}, bL::AbstractTensor{S,3}
) where {S<:ElementarySpace}
    #= 
            da      db
            ↑       ↑
    ← DX ← aR ← D ← bL → DY →
    =#
    return @tensor aRbL[DX, da, db, DY] := aR[DX, da, D] * bL[D, db, DY]
end

"""
Calculate the cost function
```
    f(a,b)  = | |Psi(a1,b1)> - |Psi(a2,b2)> |^2
    = <Psi(a1,b1)|Psi(a1,b1)> + <Psi(a2,b2)|Psi(a2,b2)>
        - 2 Re<Psi(a1,b1)|Psi(a2,b2)>
```
"""
function cost_func(
    env::BondEnv{S}, aR1bL1::AbstractTensor{S,4}, aR2bL2::AbstractTensor{S,4}
) where {S<:ElementarySpace}
    t1 = inner_prod(env, aR1bL1, aR1bL1)
    t2 = inner_prod(env, aR2bL2, aR2bL2)
    t3 = inner_prod(env, aR1bL1, aR2bL2)
    return real(t1) + real(t2) - 2 * real(t3)
end

"""
Solving the equations
```
    Ra aR = Sa, Rb bL = Sb
```
"""
function solve_ab(
    tR::AbstractTensor{S,4}, tS::AbstractTensorMap{S,3}, ab0::AbstractTensor{S,3}
) where {S<:ElementarySpace}
    f(x) = ncon((tR, x), ([-1, -2, 1, 2], [1, 2, -3]))
    ab, info = linsolve(f, tS, permute(ab0, (1, 3, 2)), 0, 1)
    return permute(ab, (1, 3, 2)), info
end

"""
Minimize the cost function
```
    fix bL:
    d(aR,aR†) = aR† Ra aR - aR† Sa - Sa† aR + T
    minimized by Ra aR = Sa

    fix aR:
    d(bL,bL†) = bL† Rb bL - bL† Sb - Sb† bL + T
    minimized by Rb bL = Sb
```
`aR0`, `bL0` are initial values of `aR`, `bL`
"""
function bond_optimize(
    env::BondEnv{S}, a::AbstractTensor{S,3}, b::AbstractTensor{S,3}, alg::ALSTruncation
) where {S<:ElementarySpace}
    # dual check
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    @assert [isdual(space(a, ax)) for ax in 1:2] == [0, 0]
    @assert [isdual(space(b, ax)) for ax in 2:3] == [0, 0]
    if alg.verbose
        @info "Alternating least square optimization --------"
        @info @sprintf(
            "%-4s%12s%12s%12s%12s %10s\n",
            "Step",
            "Cost",
            "Fidelity",
            "ϵ_cost",
            "ϵ_fid",
            "Time/s"
        )
    end
    time0 = time()
    aR2bL2 = _combine_aRbL(a, b)
    # initialize truncated aR, bL
    aR, s, bL = tsvd(aR2bL2, ((1, 2), (3, 4)); trunc=alg.trscheme)
    # normalize
    s /= norm(s, Inf)
    Vtrunc = space(s, 1)
    aR, bL = absorb_s(aR, s, bL)
    aR, bL = permute(aR, (1, 2, 3)), permute(bL, (1, 2, 3))
    aRbL = _combine_aRbL(aR, bL)
    cost00 = cost_func(env, aRbL, aR2bL2)
    fid00 = fidelity(env, aRbL, aR2bL2)
    cost0, fid0, diff_fid = cost00, fid00, 0.0
    # no need to further optimize
    if abs(cost0) < 5e-15
        time1 = time()
        if alg.verbose
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
            Ra = tensor_Ra(env, bL)
            Sa = tensor_Sa(env, bL, aR2bL2)
            aR, info_a = solve_ab(Ra, Sa, aR)
            Rb = tensor_Rb(env, aR)
            Sb = tensor_Sb(env, aR, aR2bL2)
            bL, info_b = solve_ab(Rb, Sb, bL)
            aRbL = _combine_aRbL(aR, bL)
            cost = cost_func(env, aRbL, aR2bL2)
            fid = fidelity(env, aRbL, aR2bL2)
            diff_cost = abs(cost - cost0) / cost00
            diff_fid = abs(fid - fid0)
            time1 = time()
            if alg.verbose && (count == 1 || count % alg.check_int == 0)
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
            aR0, bL0 = deepcopy(aR), deepcopy(bL)
            if count == alg.maxiter
                @warn "Warning: max iter $(alg.maxiter) reached for ALS optimization\n"
            end
        end
    end
    aRbL = _combine_aRbL(aR, bL)
    aR, s, bL = tsvd(aRbL, ((1, 2), (3, 4)); trunc=truncspace(Vtrunc))
    # normalize singular value spectrum
    s /= norm(s, Inf)
    return aR, s, bL, (; fid0, diff_fid)
end

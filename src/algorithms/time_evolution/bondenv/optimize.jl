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
function tensor_Ra(env::BondEnv, bL::AbstractTensorMap)
    @autoopt @tensor Ra[DX1, Db1, DX0, Db0] := (
        env[DX1, DY1, DX0, DY0] * bL[Db0, db, DY0] * conj(bL[Db1, db, DY1])
    )
    return Ra
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
function tensor_Sa(env::BondEnv, bL::AbstractTensorMap, aR2bL2::AbstractTensorMap)
    @autoopt @tensor Sa[DX1, Db1, da] := (
        env[DX1, DY1, DX0, DY0] * conj(bL[Db1, db, DY1]) * aR2bL2[DX0, da, db, DY0]
    )
    return Sa
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
function tensor_Rb(env::BondEnv, aR::AbstractTensorMap)
    @autoopt @tensor Rb[Da1, DY1, Da0, DY0] := (
        env[DX1, DY1, DX0, DY0] * aR[DX0, da, Da0] * conj(aR[DX1, da, Da1])
    )
    return Rb
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
function tensor_Sb(env::BondEnv, aR::AbstractTensorMap, aR2bL2::AbstractTensorMap)
    @autoopt @tensor Sb[Da1, DY1, db] := (
        env[DX1, DY1, DX0, DY0] * conj(aR[DX1, da, Da1]) * aR2bL2[DX0, da, db, DY0]
    )
    return Sb
end

"""
Calculate the norm <Psi(a1,b1)|Psi(a2,b2)>
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
    env::BondEnv, aR1bL1::AbstractTensorMap, aR2bL2::AbstractTensorMap
)
    @autoopt @tensor t[:] := (
        env[DX1, DY1, DX0, DY0] * conj(aR1bL1[DX1, da, db, DY1]) * aR2bL2[DX0, da, db, DY0]
    )
    return first(blocks(t))[2][1]
end

"""
Contract the axis between `aR` and `bL` tensors
"""
function _combine_aRbL(aR::AbstractTensorMap, bL::AbstractTensorMap)
    #= 
            da      db
            ↑       ↑
    ← DX ← aR ← D ← bL → DY →
    =#
    @tensor aRbL[DX, da, db, DY] := aR[DX, da, D] * bL[D, db, DY]
    return aRbL
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
    env::BondEnv, aR1bL1::AbstractTensorMap, aR2bL2::AbstractTensorMap
)
    t1 = inner_prod(env, aR1bL1, aR1bL1)
    t2 = inner_prod(env, aR2bL2, aR2bL2)
    t3 = inner_prod(env, aR1bL1, aR2bL2)
    return real(t1) + real(t2) - 2 * real(t3)
end

"""
Calculate the approximate local inner product
`<aR1 bL1|aR2 bL2>`
```
    |→ aR1bL1† ←|
    |   ↑   ↑   |
    DW  da  db  DE
    |   ↑   ↑   |
    |← aR2 bL2 →|
```
"""
function inner_prod_local(aR1bL1::AbstractTensorMap, aR2bL2::AbstractTensorMap)
    @autoopt @tensor t[:] := (conj(aR1bL1[DW, da, db, DE]) * aR2bL2[DW, da, db, DE])
    return first(blocks(t))[2][1]
end

"""
Calculate the fidelity using aR, bL
between two evolution steps
```
                |<aR1 bL1 | aR2 bL2>|
    ---------------------------------------------
    sqrt(<aR1 bL1 | aR1 bL1> <aR2 bL2 | aR2 bL2>)
```
"""
function local_fidelity(aR1bL1::AbstractTensorMap, aR2bL2::AbstractTensorMap)
    b12 = inner_prod_local(aR1bL1, aR2bL2)
    b11 = inner_prod_local(aR1bL1, aR1bL1)
    b22 = inner_prod_local(aR2bL2, aR2bL2)
    return abs(b12) / sqrt(abs(b11 * b22))
end

"""
Solving the equations
```
    Ra aR = Sa, Rb bL = Sb
```
"""
function solve_ab(R::AbstractTensorMap, S::AbstractTensorMap, ab0::AbstractTensorMap)
    f(x) = ncon((R, x), ([-1, -2, 1, 2], [1, 2, -3]))
    ab, info = linsolve(f, S, permute(ab0, (1, 3, 2)), 0, 1)
    return permute(ab, (1, 3, 2)), info
end

"""
Algorithm struct for the alternating least square optimization step in full update. 
`tol` is the maximum `|fid_{n+1} - fid_{n}| / fid_0` 
(normalized local fidelity change between two optimization steps)
"""
@kwdef struct ALSOptimize
    maxiter::Int = 50
    tol::Float64 = 1e-15
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
function fu_optimize(
    aR0::AbstractTensorMap,
    bL0::AbstractTensorMap,
    aR2bL2::AbstractTensorMap,
    env::BondEnv,
    alg::ALSOptimize;
    check_int::Int=1,
)
    @debug "---- Iterative optimization ----\n"
    @debug @sprintf("%-6s%12s%12s%12s %10s\n", "Step", "Cost", "ϵ_d", "ϵ_ab", "Time/s")
    aR, bL = deepcopy(aR0), deepcopy(bL0)
    time0 = time()
    aRbL = _combine_aRbL(aR, bL)
    cost00 = cost_func(env, aRbL, aR2bL2)
    fid00 = local_fidelity(aRbL, aR2bL2)
    cost0, fid0 = cost00, fid00
    # no need to further optimize
    if abs(cost0) < 5e-15
        time1 = time()
        @debug @sprintf(
            "%-6d%12.3e%12.3e%12.3e %10.3f\n", 0, cost0, NaN, NaN, time1 - time0
        )
        return aR, bL, cost0
    end
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
        fid = local_fidelity(aRbL, aR2bL2)
        diff_d = abs(cost - cost0) / cost00
        diff_ab = abs(fid - fid0) / fid00
        time1 = time()
        if (count == 1 || count % check_int == 0)
            @debug @sprintf(
                "%-6d%12.3e%12.3e%12.3e %10.3f\n",
                count,
                cost,
                diff_d,
                diff_ab,
                time1 - time0
            )
        end
        if diff_ab < alg.tol
            break
        end
        aR0, bL0 = deepcopy(aR), deepcopy(bL)
        cost0, fid0 = cost, fid
        if count == alg.maxiter
            @warn "Warning: max iter $(alg.maxiter) reached for ALS optimization\n"
        end
    end
    return aR, bL, cost0
end

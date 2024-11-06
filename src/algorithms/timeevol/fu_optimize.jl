"""
Construct the environment (norm) tensor
```
    left half                       right half
    C1 -χ4 - T1 ------- χ6 ------- T1 - χ8 - C2     r-1
    |        ‖                      ‖        |
    χ2      DNX                    DNY      χ10
    |        ‖                      ‖        |
    T4 =DWX= XX = DX =       = DY = YY =DEY= T2     r
    |        ‖                      ‖        |
    χ1      DSX                    DSY       χ9
    |        ‖                      ‖        |
    C4 -χ3 - T3 ------- χ5 ------- T3 - χ7 - C3     r+1
    c-1      c                      c+1     c+2
```
which can be more simply denoted as
```
    |------------|
    |→ DX1  DY1 ←|   axis order
    |← DX0  DX1 →|   (DX1, DY1, DX0, DY0)
    |------------|
```
The axes 1, 2 (or 3, 4) come from X†, Y† (or X, Y)
"""
function tensor_env(
    row::Int, col::Int, X::AbstractTensorMap, 
    Y::AbstractTensorMap, envs::CTMRGEnv
)
    Nr, Nc = size(envs.corners)[[2,3]]
    cm1 = _prev(col, Nc);
    cp1 = _next(col, Nc); cp2 = _next(cp1, Nc)
    rm1 = _prev(row, Nr); rp1 = _next(row, Nr)
    c1 = envs.corners[1, rm1, cm1]
    c2 = envs.corners[2, rm1, cp2]
    c3 = envs.corners[3, rp1, cp2]
    c4 = envs.corners[4, rp1, cm1]
    t1X, t1Y = envs.edges[1, rm1, col], envs.edges[1, rm1, cp1]
    t2 = envs.edges[2, row, cp2]
    t3X, t3Y = envs.edges[3, rp1, col], envs.edges[3, rp1, cp1]
    t4 = envs.edges[4, row, cm1]
    # left half
    @autoopt @tensor lhalf[DX1, DX0, χ5, χ6] := (
        c4[χ3, χ1] * t4[χ1, DWX0, DWX1, χ2] * c1[χ2, χ4] *
        t3X[χ5, DSX0, DSX1, χ3] * X[DNX0, DX0, DSX0, DWX0] *
        conj(X[DNX1, DX1, DSX1, DWX1]) * t1X[χ4, DNX0, DNX1, χ6]
    )
    # right half
    @autoopt @tensor rhalf[DY1, DY0, χ5, χ6] := (
        c3[χ9, χ7] * t2[χ10, DEY0, DEY1, χ9] * c2[χ8, χ10] *
        t3Y[χ7, DSY0, DSY1, χ5] * Y[DNY0, DEY0, DSY0, DY0] * 
        conj(Y[DNY1, DEY1, DSY1, DY1]) * t1Y[χ6, DNY0, DNY1, χ8]
    )
    # combine
    @autoopt @tensor env[DX1, DY1; DX0, DY0] := (
        lhalf[DX1, DX0, χ5, χ6] * rhalf[DY1, DY0, χ5, χ6]
    )
    return env
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
function tensor_Ra(env::AbstractTensorMap, bL::AbstractTensorMap)
    @autoopt @tensor Ra[DX1, Db1, DX0, Db0] := (
        env[DX1, DY1, DX0, DY0] *
        bL[Db0, db, DY0] * conj(bL[Db1, db, DY1])
    )
    return Ra
end


"""
Construct the tensor
```
    |--------------env--------------|
    |→ DX1         Db1 → bL† ← DY1 ←|
    |                    ↑          |
    |         da         db         |
    |         ↑          ↑          |
    |← DX0 ←- aR2 ←- D ← bL2 → DY0 →|
    |-------------------------------|
```
"""
function tensor_Sa(
    env::AbstractTensorMap, aR2::AbstractTensorMap, 
    bL::AbstractTensorMap, bL2::AbstractTensorMap
)
    @autoopt @tensor Sa[DX1, Db1, da] := (
        env[DX1, DY1, DX0, DY0] * conj(bL[Db1, db, DY1]) *
        bL2[D, db, DY0] * aR2[DX0, da, D]
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
function tensor_Rb(env::AbstractTensorMap, aR::AbstractTensorMap)
    @autoopt @tensor Rb[Da1, DY1, Da0, DY0] := (
        env[DX1, DY1, DX0, DY0] *
        aR[DX0, da, Da0] * conj(aR[DX1, da, Da1])
    )
    return Rb
end


"""
Construct the tensor
```
    |--------------env--------------|
    |→ DX1 → aR† → Da1         DY1 ←|
    |        ↑                      |
    |        da         db          |
    |        ↑          ↑           |
    |← DX0 ← aR2 ← D ←- bL2 -→ DY0 →|
    |-------------------------------|
```
"""
function tensor_Sb(
    env::AbstractTensorMap, aR::AbstractTensorMap, 
    aR2::AbstractTensorMap, bL2::AbstractTensorMap
)
    @autoopt @tensor Sb[Da1, DY1, db] := (
        env[DX1, DY1, DX0, DY0] * conj(aR[DX1, da, Da1]) *
        aR2[DX0, da, D] * bL2[D, db, DY0]
    )
    return Sb
end


"""
Calculate the norm <Psi(a1,b1)|Psi(a2,b2)>
```
    |--------------env--------------|
    |→ DX1 → aR1†→ D1 → bL1† ← DY1 ←|
    |        ↑           ↑          |
    |        da          db         |
    |        ↑           ↑          |
    |← DX0 ← aR2 ← D0 ← bL2 → DY0 -→|
    |-------------------------------|
```
"""
function inner_prod(
    env::AbstractTensorMap, 
    aR1::AbstractTensorMap, bL1::AbstractTensorMap,
    aR2::AbstractTensorMap, bL2::AbstractTensorMap
)
    @autoopt @tensor t[:] := (
        env[DX1, DY1, DX0, DY0] *
        conj(aR1[DX1, da, D1]) * conj(bL1[D1, db, DY1]) *
        aR2[DX0, da, D0] * bL2[D0, db, DY0]
    )
    return first(blocks(t))[2][1]
end

"""
Calculate the cost function
```
    f(a,b)  = | |Psi(a,b)> - |Psi(a2,b2)> |^2
    = <Psi(a,b)|Psi(a,b)> + <Psi(a2,b2)|Psi(a2,b2)>
        - 2 Re<Psi(a,b)|Psi(a2,b2)>
```
"""
function cost_func(
    env::AbstractTensorMap, 
    aR::AbstractTensorMap, bL::AbstractTensorMap,
    aR2::AbstractTensorMap, bL2::AbstractTensorMap
)
    t1 = inner_prod(env, aR, bL, aR, bL)
    t2 = inner_prod(env, aR2, bL2, aR2, bL2)
    t3 = inner_prod(env, aR, bL, aR2, bL2)
    return real(t1) + real(t2) - 2 * real(t3)
end


"""
Calculate the approximate local inner product
`<aR1 bL1|aR2 bL2>`
```
    |→ aR1† → D1 → bL1† ←|
    |   ↑          ↑     |
    DW  da         db    DE
    |   ↑          ↑     |
    |← aR2 ←- D0 ← bL2 -→|
```
"""
function inner_prod_local(
    aR1::AbstractTensorMap, bL1::AbstractTensorMap,
    aR2::AbstractTensorMap, bL2::AbstractTensorMap
)
    @autoopt @tensor t[:] := (
        conj(aR1[DW, da, D1]) * conj(bL1[D1, db, DE]) *
        aR2[DW, da, D0] * bL2[D0, db, DE]
    )
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
function local_fidelity(
    aR1::AbstractTensorMap, bL1::AbstractTensorMap, 
    aR2::AbstractTensorMap, bL2::AbstractTensorMap
)
    b12 = inner_prod_local(aR1, bL1, aR2, bL2)
    b11 = inner_prod_local(aR1, bL1, aR1, bL1)
    b22 = inner_prod_local(aR2, bL2, aR2, bL2)
    return abs(b12) / sqrt(abs(b11*b22))
end

"""
Solving the equations
```
    Ra aR = Sa, Rb bL = Sb
```
"""
function solve_ab(
    R::AbstractTensorMap, S::AbstractTensorMap, 
    ab0::AbstractTensorMap
)
    f(x) = ncon((R, x), ([-1,-2,1,2], [1,2,-3]))
    ab, info = linsolve(f, S, permute(ab0, (1,3,2)), 0, 1)
    return permute(ab, (1,3,2)), info
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
    aR0::AbstractTensorMap, bL0::AbstractTensorMap, 
    aR2::AbstractTensorMap, bL2::AbstractTensorMap, 
    env::AbstractTensorMap;
    maxiter::Int=50, maxdiff::Float64=1e-15, 
    check_int::Int=1, verbose::Bool=false
)
    if verbose
        println("---- Iterative optimization ----")
        @printf(
            "%-6s%12s%12s%12s %10s\n", 
            "Step", "Cost", "ϵ_d", "ϵ_ab", "Time/s"
        )
    end
    aR, bL = deepcopy(aR0), deepcopy(bL0)
    time0 = time()
    cost00 = cost_func(env, aR, bL, aR2, bL2)
    fid00 = local_fidelity(aR, bL, aR2, bL2)
    cost0, fid0 = cost00, fid00
    # no need to further optimize
    if abs(cost0) < 5e-15
        if verbose
            time1 = time()
            println(@sprintf(
                "%-6d%12.3e%12.3e%12.3e %10.3f\n", 
                0, cost0, NaN, NaN, time1 - time0
            ))
        end
        return aR, bL, cost0
    end
    for count in 1:maxiter
        time0 = time()
        Ra = tensor_Ra(env, bL)
        Sa = tensor_Sa(env, aR2, bL, bL2)
        aR, info_a = solve_ab(Ra, Sa, aR)
        Rb = tensor_Rb(env, aR)
        Sb = tensor_Sb(env, aR, aR2, bL2)
        bL, info_b = solve_ab(Rb, Sb, bL)
        cost = cost_func(env, aR, bL, aR2, bL2)
        fid = local_fidelity(aR, bL, aR2, bL2)
        diff_d = abs(cost - cost0) / cost00
        diff_ab = abs(fid - fid0) / fid00
        time1 = time()
        if verbose && (count == 1 || count % check_int == 0)
            @printf(
                "%-6d%12.3e%12.3e%12.3e %10.3f\n", 
                count, cost, diff_d, diff_ab, time1 - time0
            )
        end
        if diff_ab < maxdiff
            break
        end
        aR0, bL0 = deepcopy(aR), deepcopy(bL)
        cost0, fid0 = cost, fid
        if count == maxiter
            println("Warning: max iter $maxiter reached for optimization")
        end
    end
    return aR, bL, cost0
end


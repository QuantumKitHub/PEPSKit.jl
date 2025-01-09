"""
Construct the tensor
```
    |→ DX1  Db1 → b† ← DY1 ←|
    |                       |
    |----------env----------|
    |                       |
    |← DX0  Db0 ← b -→ DY0 →|
```
"""
function tensor_ga(env::AbstractTensorMap, b::AbstractTensorMap)
    @autoopt @tensor ga[DX1, Db1, DX0, Db0] :=
        env[DX1, DY1, DX0, DY0] * b[Db0, DY0] * conj(b[Db1, DY1])
    return ga
end

"""
Construct the tensor
```
    |→ DX1  Db1 → b† ← DY1 ←|
    |                       |
    |----------env----------|
    |                       |
    |← DX0 ←-- a2b2 -→ DY0 →|
```
"""
function tensor_Ja(env::AbstractTensorMap, b::AbstractTensorMap, a2b2::AbstractTensorMap)
    @autoopt @tensor Ja[DX1, Db1] :=
        env[DX1, DY1, DX0, DY0] * conj(b[Db1, DY1]) * a2b2[DX0, DY0]
    return Ja
end

"""
Construct the tensor
```
    |→ DX1 → a† → Da1  DY1 ←|
    |                       |
    |----------env----------|
    |                       |
    |← DX0 ← a ←- Da0  DY0 →|
```
"""
function tensor_gb(env::AbstractTensorMap, a::AbstractTensorMap)
    @autoopt @tensor gb[Da1, DY1, Da0, DY0] :=
        env[DX1, DY1, DX0, DY0] * a[DX0, Da0] * conj(a[DX1, Da1])
    return gb
end

"""
Construct the tensor
```
    |→ DX1 → a† → Da1  DY1 ←|
    |                       |
    |----------env----------|
    |                       |
    |← DX0 ←-- a2b2 -→ DY0 →|
```
"""
function tensor_Jb(env::AbstractTensorMap, a::AbstractTensorMap, a2b2::AbstractTensorMap)
    @autoopt @tensor Jb[Da1, DY1] :=
        env[DX1, DY1, DX0, DY0] * conj(a[DX1, Da1]) * a2b2[DX0, DY0]
    return Jb
end

"""
Calculate the norm <Psi(a1,b1)|Psi(a2,b2)>
```
    |→ DX1 -→ a1b1† ←- DY1 ←|
    |                       |
    |----------env----------|
    |                       |
    |← DX0 ←-- a2b2 -→ DY0 →|
```
"""
function inner_prod2(
    env::AbstractTensorMap, a1b1::AbstractTensorMap, a2b2::AbstractTensorMap
)
    @autoopt @tensor t[:] := env[DX1, DY1, DX0, DY0] * conj(a1b1[DX1, DY1]) * a2b2[DX0, DY0]
    return first(blocks(t))[2][1]
end

"""
Contract the axis between `a` and `b` tensors
"""
function _combine_ab(a::AbstractTensorMap, b::AbstractTensorMap)
    #= 
    ← DX ← a ← D ← b → DY →
    =#
    @tensor ab[DX, DY] := a[DX, D] * b[D, DY]
    return ab
end

"""
Calculate the cost function
```
    f(a,b)  = | |Psi(a1,b1)> - |Psi(a2,b2)> |^2
    = <Psi(a1,b1)|Psi(a1,b1)> + <Psi(a2,b2)|Psi(a2,b2)>
        - 2 Re<Psi(a1,b1)|Psi(a2,b2)>
```
"""
function cost_func2(
    env::AbstractTensorMap, a1b1::AbstractTensorMap, a2b2::AbstractTensorMap
)
    t1 = inner_prod2(env, a1b1, a1b1)
    t2 = inner_prod2(env, a2b2, a2b2)
    t3 = inner_prod2(env, a1b1, a2b2)
    return real(t1) + real(t2) - 2 * real(t3)
end

"""
Calculate the approximate local inner product
`<a1 b1|a2 b2>`
```
    |-→ a1b1† ←-|
    |           |
    DW         DE
    |           |
    |←-- a2b2 -→|
```
"""
function inner_prod_local2(a1b1::AbstractTensorMap, a2b2::AbstractTensorMap)
    @autoopt @tensor t[:] := (conj(a1b1[DW, DE]) * a2b2[DW, DE])
    return first(blocks(t))[2][1]
end

"""
Calculate the fidelity using a, b
between two evolution steps
```
            |<a1 b1|a2 b2>|
    ---------------------------------
    sqrt(<a1 b1|a1 b1> <a2 b2|a2 b2>)
```
"""
function local_fidelity2(a1b1::AbstractTensorMap, a2b2::AbstractTensorMap)
    b12 = inner_prod_local2(a1b1, a2b2)
    b11 = inner_prod_local2(a1b1, a1b1)
    b22 = inner_prod_local2(a2b2, a2b2)
    return abs(b12) / sqrt(abs(b11 * b22))
end

"""
Solving the equations
```
    gx x = Jx (x = a, b)
```
"""
function solve_ab2(gx::AbstractTensorMap, Jx::AbstractTensorMap, x0::AbstractTensorMap)
    f(x) = ncon((gx, x), ([-1, -2, 1, 2], [1, 2]))
    x1, info = linsolve(f, Jx, permute(x0, (1, 2)), 0, 1)
    return permute(x1, ((1,), (2,))), info
end

"""
Algorithm struct for the alternating least square optimization step in full update. 
`tol` is the maximum `|fid_{n+1} - fid_{n}| / fid_0` 
(normalized local fidelity change between two optimization steps)
"""
@kwdef struct ALSOptimize
    maxiter::Int = 50
    tol::Float64 = 1e-15
    verbose::Bool = false
end

"""
    fu_optimize(a0::AbstractTensorMap, b0::AbstractTensorMap, a2b2::AbstractTensorMap, env::AbstractTensorMap, alg::ALSOptimize; check_int::Int=1)

Minimize the cost function
```
    fix b:
    d(a,a†) = a† ga a - a† Ja - Ja† a + T
    minimized by ga a = Ja

    fix a:
    d(b,b†) = b† gb b - b† Jb - Jb† b + T
    minimized by gb b = Jb
```
`a0`, `b0` are initial values of `a`, `b`
"""
function als_optimize(
    a0::AbstractTensorMap,
    b0::AbstractTensorMap,
    a2b2::AbstractTensorMap,
    env::AbstractTensorMap,
    alg::ALSOptimize;
    check_int::Int=1,
)
    verbose = alg.verbose
    if verbose
        @info "---- Iterative optimization ----\n"
        @info @sprintf("%-6s%12s%12s%12s %10s\n", "Step", "Cost", "ϵ_d", "ϵ_ab", "Time/s")
        flush(stderr)
    end
    a, b = deepcopy(a0), deepcopy(b0)
    time0 = time()
    ab = _combine_ab(a, b)
    cost00 = cost_func2(env, ab, a2b2)
    fid00 = local_fidelity2(ab, a2b2)
    cost0, fid0 = cost00, fid00
    time1 = time()
    if verbose
        @info @sprintf("%-6d%12.3e%12.3e%12.3e %10.3f\n", 0, cost0, NaN, NaN, time1 - time0)
    end
    # no need to further optimize
    if abs(cost0) < 5e-15
        return a, b, cost0
    end
    for count in 1:(alg.maxiter)
        time0 = time()
        ga = tensor_ga(env, b)
        Ja = tensor_Ja(env, b, a2b2)
        a, info_a = solve_ab2(ga, Ja, a)
        gb = tensor_gb(env, a)
        Jb = tensor_Jb(env, a, a2b2)
        b, info_b = solve_ab2(gb, Jb, b)
        ab = _combine_ab(a, b)
        cost = cost_func2(env, ab, a2b2)
        fid = local_fidelity2(ab, a2b2)
        diff_d = abs(cost - cost0) / cost00
        diff_ab = abs(fid - fid0) / fid00
        time1 = time()
        if (count == 1 || count % check_int == 0)
            if verbose
                @info @sprintf(
                    "%-6d%12.3e%12.3e%12.3e %10.3f\n",
                    count,
                    cost,
                    diff_d,
                    diff_ab,
                    time1 - time0
                )
            end
        end
        if diff_ab < alg.tol
            break
        end
        a0, b0 = deepcopy(a), deepcopy(b)
        cost0, fid0 = cost, fid
        if count == alg.maxiter
            @warn "Warning: max iter $(alg.maxiter) reached for ALS optimization\n"
        end
    end
    return a, b, cost0
end

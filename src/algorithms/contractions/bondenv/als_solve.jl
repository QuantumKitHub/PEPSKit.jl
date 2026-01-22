#= 
In the following, the names `Ra`, `Sa` etc comes from 
the fast full update article Physical Review B 92, 035142 (2015)
=#

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0     Db0 - b -- DY0 -┘
        |benv|
    ┌---|    |- DX1     Db1 - b† - DY1 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Ra(benv::BondEnv, b::MPSBondTensor)
    return @autoopt @tensor Ra[DX1 Db1; DX0 Db0] := (
        benv[DX1 DY1; DX0 DY0] * b[Db0; DY0] * conj(b[Db1; DY1])
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 -- (a2 b2) -- DY0 --┘
        |benv|
    ┌---|    |- DX1   Db1 -- b† - DY1 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sa(benv::BondEnv, b::MPSBondTensor, a2b2::MPSBondTensor)
    return @autoopt @tensor Sa[DX1; Db1] := (
        benv[DX1 DY1; DX0 DY0] * conj(b[Db1; DY1]) * a2b2[DX0; DY0]
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 - a -- Da0     DY0 -┘
        |benv|
    ┌---|    |- DX1 - a† - Da1     DY1 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Rb(benv::BondEnv, a::MPSBondTensor)
    return @autoopt @tensor Rb[Da1 DY1; Da0 DY0] := (
        benv[DX1 DY1; DX0 DY0] * a[DX0; Da0] * conj(a[DX1; Da1])
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 -- (a2 b2) -- DY0 --┘
        |benv|
    ┌---|    |- DX1 -- a† - Da1   DY1 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sb(benv::BondEnv, a::MPSBondTensor, a2b2::MPSBondTensor)
    return @autoopt @tensor Sb[Da1; DY1] := (
        benv[DX1 DY1; DX0 DY0] * conj(a[DX1; Da1]) * a2b2[DX0; DY0]
    )
end

"""
$(SIGNATURES)

Contract the axis between `a` and `b` tensors
```
    -- DX - a - D - b - DY --
```
"""
function _combine_ab(a::MPSBondTensor, b::MPSBondTensor)
    return @tensor ab[DX; DY] := a[DX; D] * b[D; DY]
end

"""
$(SIGNATURES)

Calculate the cost function
```
    f(a,b)  = ‖ |a1,b1⟩ - |a2,b2⟩ ‖^2
    = ⟨a1,b1|a1,b1⟩ - 2 Re⟨a1,b1|a2,b2⟩ + ⟨a2,b2|a2,b2⟩
```
"""
function cost_function_als(
        benv::BondEnv, a1b1::MPSBondTensor, a2b2::MPSBondTensor
    )
    t1 = inner_prod(benv, a1b1, a1b1)
    t2 = inner_prod(benv, a2b2, a2b2)
    t3 = inner_prod(benv, a1b1, a2b2)
    return real(t1) + real(t2) - 2 * real(t3)
end

"""
$(SIGNATURES)

Solve the equations `Rx x = Sx` (x = a, b) with initial guess `x0`
```
    ┌---------------------------┐
    |   ┌----┐                  |
    └---|    |--- 1 -- x -- 2 --┘
        | Rx |
    ┌---|    |--- -1       -2 --┐
    |   └----┘                  |
    └---------------------------┘
```
"""
function _solve_ab(Rx::BondEnv, Sx::MPSBondTensor, x0::MPSBondTensor)
    f(x) = (@tensor Sx2[-1; -2] := Rx[-1 -2; 1 2] * x[1; 2])
    x1, info = linsolve(f, Sx, x0, 0, 1)
    return x1, info
end

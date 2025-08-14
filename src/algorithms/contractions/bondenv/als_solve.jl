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
        |    |                ↓
        |benv|                db
        |    |                ↓
    ┌---|    |- DX1     Db1 - b† - DY1 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Ra(
        benv::BondEnv{T, S}, b::AbstractTensorMap{T, S, 2, 1}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Ra[DX1 Db1; DX0 Db0] := (
        benv[DX1 DY1; DX0 DY0] * b[Db0 DY0; db] * conj(b[Db1 DY1; db])
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 -- (a2 b2) -- DY0 --┘
        |    |         ↓     ↓
        |benv|         da    db
        |    |               ↓
    ┌---|    |- DX1   Db1 -- b† - DY1 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sa(
        benv::BondEnv{T, S}, b::AbstractTensorMap{T, S, 2, 1}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Sa[DX1 Db1; da] := (
        benv[DX1 DY1; DX0 DY0] * conj(b[Db1 DY1; db]) * a2b2[DX0 DY0; da db]
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 - a -- Da0     DY0 -┘
        |    |        ↓
        |benv|        da
        |    |        ↓
    ┌---|    |- DX1 - a† - Da1     DY1 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Rb(
        benv::BondEnv{T, S}, a::AbstractTensorMap{T, S, 2, 1}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Rb[Da1 DY1; Da0 DY0] := (
        benv[DX1 DY1; DX0 DY0] * a[DX0 Da0; da] * conj(a[DX1 Da1; da])
    )
end

"""
$(SIGNATURES)

Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX0 -- (a2 b2) -- DY0 --┘
        |    |         ↓     ↓
        |benv|         da    db
        |    |         ↓
    ┌---|    |- DX1 -- a† - Da1   DY1 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sb(
        benv::BondEnv{T, S}, a::AbstractTensorMap{T, S, 2, 1}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Sb[Da1 DY1; db] := (
        benv[DX1 DY1; DX0 DY0] * conj(a[DX1 Da1; da]) * a2b2[DX0 DY0; da db]
    )
end

"""
$(SIGNATURES)

Calculate the inner product <a1,b1|a2,b2>
```
    ┌--------------------------------┐
    |   ┌----┐                       |
    └---|    |- DX0 - (a2 b2) - DY0 -┘
        |    |        ↓    ↓
        |benv|        da   db
        |    |        ↓    ↓
    ┌---|    |- DX1 - (a1 b1)†- DY1 -┐
    |   └----┘                       |
    └--------------------------------┘
```
"""
function inner_prod(
        benv::BondEnv{T, S}, a1b1::AbstractTensorMap{T, S, 2, 2}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor benv[DX1 DY1; DX0 DY0] *
        conj(a1b1[DX1 DY1; da db]) *
        a2b2[DX0 DY0; da db]
end

"""
$(SIGNATURES)

Calculate the fidelity between two evolution steps
```
        |⟨a1,b1|a2,b2⟩|^2
    --------------------------
    ⟨a1,b1|a1,b1⟩⟨a2,b2|a2,b2⟩
```
"""
function fidelity(
        benv::BondEnv{T, S}, a1b1::AbstractTensorMap{T, S, 2, 2}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    b12 = inner_prod(benv, a1b1, a2b2)
    b11 = inner_prod(benv, a1b1, a1b1)
    b22 = inner_prod(benv, a2b2, a2b2)
    return abs2(b12) / abs(b11 * b22)
end

"""
$(SIGNATURES)

Contract the axis between `a` and `b` tensors
```
    -- DX - a - D - b - DY --
            ↓       ↓
            da      db
```
"""
function _combine_ab(
        a::AbstractTensorMap{T, S, 2, 1}, b::AbstractTensorMap{T, S, 1, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensor ab[DX DY; da db] := a[DX da; D] * b[D; db DY]
end
function _combine_ab(
        a::AbstractTensorMap{T, S, 2, 1}, b::AbstractTensorMap{T, S, 2, 1}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensor ab[DX DY; da db] := a[DX D; da] * b[D DY; db]
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
        benv::BondEnv{T, S}, a1b1::AbstractTensorMap{T, S, 2, 2}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
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
        |    |         ↓
        | Rx |        -3
        |    |
    ┌---|    |--- -1       -2 --┐
    |   └----┘                  |
    └---------------------------┘
```
"""
function _solve_ab(
        Rx::AbstractTensorMap{T, S, 2, 2},
        Sx::AbstractTensorMap{T, S, 2, 1},
        x0::AbstractTensorMap{T, S, 2, 1},
    ) where {T <: Number, S <: ElementarySpace}
    f(x) = (@tensor Sx2[-1 -2; -3] := Rx[-1 -2; 1 2] * x[1 2; -3])
    x1, info = linsolve(f, Sx, x0, 0, 1)
    return x1, info
end

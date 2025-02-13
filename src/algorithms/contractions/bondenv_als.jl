"""
Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX1     Db1 - b† - DY1 -┘
        |    |                ↑
        |benv|                db
        |    |                ↑
    ┌---|    |- DX0     Db0 - b -- DY0 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Ra(
    benv::BondEnv{T,S}, b::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Ra[DX1, Db1; DX0, Db0] := (
        benv[DX1, DY1, DX0, DY0] * b[Db0, db, DY0] * conj(b[Db1, db, DY1])
    )
end

"""
Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX1   Db1 -- b† - DY1 --┘
        |    |               ↑
        |benv|         da    db
        |    |         ↑     ↑
    ┌---|    |- DX0 -- a2    b2 - DY0 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sa(
    benv::BondEnv{T,S}, b::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sa[DX1, Db1, da] := (
        benv[DX1, DY1, DX0, DY0] * conj(b[Db1, db, DY1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX1 - a† - Da1     DY1 -┘
        |    |        ↑
        |benv|        da
        |    |        ↑
    ┌---|    |- DX0 - a -- Da0     DY0 -┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Rb(
    benv::BondEnv{T,S}, a::AbstractTensor{T,S,3}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Rb[Da1, DY1; Da0, DY0] := (
        benv[DX1, DY1, DX0, DY0] * a[DX0, da, Da0] * conj(a[DX1, da, Da1])
    )
end

"""
Construct the tensor
```
    ┌-----------------------------------┐
    |   ┌----┐                          |
    └---|    |- DX1 -- a† - Da1   DY1 --┘
        |    |         ↑
        |benv|         da   db
        |    |         ↑     ↑
    ┌---|    |- DX0 -- a2   b2 -- DY0 --┐
    |   └----┘                          |
    └-----------------------------------┘
```
"""
function _tensor_Sb(
    benv::BondEnv{T,S}, a::AbstractTensor{T,S,3}, a2b2::AbstractTensor{T,S,4}
) where {T<:Number,S<:ElementarySpace}
    return @autoopt @tensor Sb[Da1, DY1, db] := (
        benv[DX1, DY1, DX0, DY0] * conj(a[DX1, da, Da1]) * a2b2[DX0, da, db, DY0]
    )
end

"""
Calculate the inner product <a1,b1|a2,b2>
```
    ┌--------------------------------┐
    |   ┌----┐                       |
    └---|    |- DX1 - (a1 b1)†- DY1 -┘
        |    |        ↑    ↑
        |benv|        da   db
        |    |        ↑    ↑
    ┌---|    |- DX0 - (a2 b2) - DY0 -┐
    |   └----┘                       |
    └--------------------------------┘
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

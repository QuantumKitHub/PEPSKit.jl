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
function _tensor_Ra(benv::BondEnv, b::MPSTensor)
    return @autoopt @tensor Ra[DX1 Db1; DX0 Db0] := (
        benv[DX1 DY1; DX0 DY0] * b[Db0 db; DY0] * conj(b[Db1 db; DY1])
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
        benv::BondEnv, b::MPSTensor, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Sa[DX1 da; Db1] := (
        benv[DX1 DY1; DX0 DY0] * conj(b[Db1 db; DY1]) * a2b2[DX0 DY0; da db]
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
function _tensor_Rb(benv::BondEnv, a::MPSTensor)
    return @autoopt @tensor Rb[Da1 DY1; Da0 DY0] := (
        benv[DX1 DY1; DX0 DY0] * a[DX0 da; Da0] * conj(a[DX1 da; Da1])
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
        benv::BondEnv, a::MPSTensor, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor Sb[Da1 db; DY1] := (
        benv[DX1 DY1; DX0 DY0] * conj(a[DX1 da; Da1]) * a2b2[DX0 DY0; da db]
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
        benv::BondEnv, a1b1::AbstractTensorMap{T, S, 2, 2}, a2b2::AbstractTensorMap{T, S, 2, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @autoopt @tensor benv[DX1 DY1; DX0 DY0] *
        conj(a1b1[DX1 DY1; da db]) * a2b2[DX0 DY0; da db]
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
        a::MPSTensor, b::AbstractTensorMap{T, S, 1, 2}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensor ab[DX DY; da db] := a[DX da; D] * b[D; db DY]
end
function _combine_ab(a::MPSTensor, b::MPSTensor)
    return @tensor ab[DX DY; da db] := a[DX da; D] * b[D db; DY]
end

"""
$(SIGNATURES)

Calculate the cost function
```
    f(a,b) = ‖ |ψ1⟩ - |ψ2⟩ ‖^2
    = ⟨ψ1|benv|ψ1⟩ - 2 Re⟨ψ1|benv|ψ2⟩ + ⟨ψ2|benv|ψ2⟩
```
and the fidelity
```
        |⟨ψ1|benv|ψ2⟩|²
    ------------------------
    ⟨ψ1|benv|ψ1⟩⟨ψ2|benv|ψ2⟩
```
"""
function cost_function_als(benv, ψ1, ψ2)
    b12 = inner_prod(benv, ψ1, ψ2)
    b11 = inner_prod(benv, ψ1, ψ1)
    b22 = inner_prod(benv, ψ2, ψ2)
    cost = real(b11) + real(b22) - 2 * real(b12)
    fid = abs2(b12) / abs(b11 * b22)
    return cost, fid
end

"""
$(SIGNATURES)

Solve the equations `Rx x = Sx` with initial guess `x0`.
"""
function _solve_als(
        Rx::AbstractTensorMap{T, S, N, N},
        Sx::GenericMPSTensor{S, N},
        x0::GenericMPSTensor{S, N}; kwargs...
    ) where {T, S, N}
    @assert N >= 2
    pR = (codomainind(Rx), domainind(Rx))
    pX = ((1, (3:(N + 1))...), (2,))
    pRX = ((1, N + 1, (2:(N - 1))...), (N,))
    f(x) = tensorcontract(Rx, pR, false, x, pX, false, pRX)
    x1, info = linsolve(f, Sx, x0, 0, 1; kwargs...)
    return x1, info
end

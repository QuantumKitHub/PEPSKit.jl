#= 
In the following, the names `Ra`, `Sa` etc comes from 
the fast full update article Physical Review B 92, 035142 (2015)
=#
"""
Construct the norm with bra bond tensors removed
```
    ┌benv-------┐
    ├---a---b---┤
    |   ↓   ↓   |
    ├--       --┤
    └-----------┘
```
"""
function _benv_ab(benv::BondEnv, ab::AbstractTensorMap{T, S, 2, 2}) where {T, S}
    return @tensor benv_ab[DX1 DY1; da db] := benv[DX1 DY1; DX0 DY0] * ab[DX0 DY0; da db]
end
function _benv_ab(benv::BondEnv, a, b)
    ab = _combine_ab(a, b)
    return _benv_ab(benv, ab)
end

"""
    _als_tensor_R(benv::BondEnv, xs::Vector{<:MPSTensor}, i::Int)

Construct the bond environment around a single reduced bond tensor
```
    i = 1           i = 2
    ┌benv-------┐   ┌benv-------┐
    ├--   --b---┤   ├---a--   --┤
    |       ↓   |   |   ↓       |
    ├--   --b̄---┤   ├---ā--   --┤
    └-----------┘   └-----------┘
```
"""
function _als_tensor_R(benv::BondEnv, xs::Vector{<:MPSTensor}, i::Int)
    return _als_tensor_R(benv, xs, Val(i))
end
function _als_tensor_R(benv::BondEnv, xs::Vector{<:MPSTensor}, ::Val{1})
    return @tensor Ra[DX1 D1; DX0 D0] :=
        benv[DX1 DY1; DX0 DY0] * xs[2][D0 db; DY0] * conj(xs[2][D1 db; DY1])
end
function _als_tensor_R(benv::BondEnv, xs::Vector{<:MPSTensor}, ::Val{2})
    return @tensor Rb[D1 DY1; D0 DY0] :=
        benv[DX1 DY1; DX0 DY0] * xs[1][DX0 da; D0] * conj(xs[1][DX1 da; D1])
end

"""
Calculate the norm
```
    ┌benv-------┐
    ├---a---b---┤
    |   ↓   ↓   |
    ├---ā---b̄---┤
    └-----------┘
```
using pre-calcuated partial contraction results.
"""
function _als2_norm(
        ab::AbstractTensorMap{T, S, 2, 2}, benv_ab::AbstractTensorMap{T, S, 2, 2}
    ) where {T, S}
    return @tensor benv_ab[DX1 DY1; da db] * conj(ab[DX1 DY1; da db])
end
function _als2_norm(a::MPSTensor, Ra::BondEnv)
    # applies to b, Rb as well
    # @tensor Rb[D1 DY1; D0 DY0] * b[D0 db; DY0] * conj(b[D1 db; DY1])
    return @tensor Ra[DX1 D1; DX0 D0] * a[DX0 da; D0] * conj(a[DX1 da; D1])
end

"""
    _als_tensor_S(
        benv_ab2::AbstractTensorMap{T, S, 2, 2},
        xs::Vector{<:MPSTensor}, i::Int
    ) where {T <: Number, S <: ElementarySpace}

Construct the overlap but with one of the bra bond tensor removed.
```
    i = 1           i = 2
    ┌benv-------┐   ┌benv-------┐
    ├---a₂==b₂--┤   ├---a₂==b₂--┤
    |   ↓   ↓   |   |   ↓   ↓   |
    ├--   --b̄---┤   ├---ā--   --┤
    └-----------┘   └-----------┘
```
The ket part is provided by the partial contraction `benv_ab2`.
"""
function _als_tensor_S(
        benv_ab2::AbstractTensorMap{T, S, 2, 2},
        xs::Vector{<:MPSTensor}, i::Int
    ) where {T <: Number, S <: ElementarySpace}
    return _als_tensor_S(benv_ab2, xs, Val(i))
end
function _als_tensor_S(
        benv_ab2::AbstractTensorMap{T, S, 2, 2},
        xs::Vector{<:MPSTensor}, ::Val{1}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensor contractcheck = true Sa[DX1 da; D1] :=
        benv_ab2[DX1 DY1; da db] * conj(xs[2][D1 db; DY1])
end
function _als_tensor_S(
        benv_ab2::AbstractTensorMap{T, S, 2, 2},
        xs::Vector{<:MPSTensor}, ::Val{2}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensor contractcheck = true Sb[D1 db; DY1] :=
        benv_ab2[DX1 DY1; da db] * conj(xs[1][DX1 da; D1])
end

"""
Calculate the inner product (overlap)
```
    ┌benv-------┐
    ├---a₂--b₂--┤
    |   ↓   ↓   |
    ├---ā---b̄---┤
    └-----------┘
```
using pre-calculated partial contraction results.
"""
function _als2_overlap(a::MPSTensor, Sa::MPSTensor)
    # applies to b, Sb as well
    # @tensor Sb[D1 db; DY1] * conj(b[D1 db; DY1])
    return @tensor Sa[DX1 da; D1] * conj(a[DX1 da; D1])
end

"""
$(SIGNATURES)

Calculate the inner product <a1,b1|a2,b2>
```
    ┌benv-------┐
    ├---a₂--b₂--┤
    |   ↓   ↓   |
    ├---ā₁--b̄₁--┤
    └-----------┘
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

Contract the axis between reduced bond tensors `a` and `b`
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

function _combine_ab_for_svd(a::MPSTensor, b::MPSTensor)
    return @tensor ab[DX da; db DY] := a[DX da; D] * b[D db; DY]
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

# applies to Rb, Sb, b as well
# b22 is the pre-calculated untruncated norm
function cost_function_als2(Ra::BondEnv, Sa::MPSTensor, a::MPSTensor, b22::Real)
    b11 = real(_als2_norm(a, Ra))
    b12 = _als2_overlap(a, Sa)
    cost = b11 + b22 - 2 * real(b12)
    fid = abs2(b12) / abs(b11 * b22)
    return cost, fid
end

"""
$(SIGNATURES)

Solve the equations `Rx x = Sx` with initial guess `x0`.

In ALS over `a`, `b`, if we fix `b`, the cost function can
be expressed in the `Ra`, `Sa` tensors as
```
    f(a†,a) = a† Ra a - a† Sa - Sa† a + const
```
Therefore `f` is minimized when
```
    ∂f/∂ā = Ra a - Sa = 0
```
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

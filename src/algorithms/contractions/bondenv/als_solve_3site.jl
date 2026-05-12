"""
Contract the virtual legs between
```
           ╲ ╱      
    ----a---m---b----
        ↓   ↓   ↓   
```
"""
function _combine_ket(
        a::MPSTensor, m::GenericMPSTensor{S, 4}, b::MPSTensor
    ) where {S}
    return @tensoropt ket[Dw0 De0 Ds0 Dn0; da dm db] :=
        a[Dw0 da; Dw′0] * m[Dw′0 dm De0 Ds0; Dn′0] * b[Dn′0 db; Dn0]
end

"""
Construct the norm with bra bond tensors removed
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a---m---b---┤
    |   ↓   ↓   ↓   |
    ├--           --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_ket(benv::BondEnv3site, ket::AbstractTensorMap{T, S, 4, 3}) where {T, S}
    return benv * twistdual(ket, 1:4)
end

"""
    _als_tensor_R(benv::BondEnv3site, xs::Vector{<:GenericMPSTensor}, i::Int)

Construct the bond environment around the `i`th tensor
in three-site ALS optimization.
```
    i = 1               i = 2               i = 3
    ┌benv-┬---┬-----┐   ┌benv-┬---┬-----┐   ┌benv-┬---┬-----┐
    |      ╲ ╱      |   |      ╲ ╱      |   |      ╲ ╱      |
    ├--   --m---b---┤   ├---a--   --b---┤   ├---a---m--   --┤
    |       ↓   ↓   |   |   ↓       ↓   |   |   ↓   ↓       |
    ├--   --m̄---b̄---┤   ├---ā--   --b̄---┤   ├---ā---m̄--   --┤
    |      ╱ ╲      |   |      ╱ ╲      |   |      ╱ ╲      |
    └-----┴---┴-----┘   └-----┴---┴-----┘   └-----┴---┴-----┘
```
"""
function _als_tensor_R(benv::BondEnv3site, xs::Vector{<:GenericMPSTensor}, i::Int)
    @assert 1 <= i <= 3
    return if i == 1
        _als3s_tensor_R1(benv, xs[2], xs[3])
    elseif i == 2
        _als3s_tensor_R2(benv, xs[1], xs[3])
    else
        _als3s_tensor_R3(benv, xs[1], xs[2])
    end
end

function _als3s_tensor_R1(benv::BondEnv3site, m::GenericMPSTensor{S, 4}, b::MPSTensor) where {S}
    return @tensoropt Ra[Dw1 Dw′1; Dw0 Dw′0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(m[Dw′1 dm De1 Ds1; Dn′1]) * conj(b[Dn′1 db; Dn1]) *
        m[Dw′0 dm De0 Ds0; Dn′0] * b[Dn′0 db; Dn0]
end
function _als3s_tensor_R2(benv::BondEnv3site, a::MPSTensor, b::MPSTensor)
    return @tensoropt Rm[Dw′1 De1 Ds1 Dn′1; Dw′0 De0 Ds0 Dn′0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(b[Dn′1 db; Dn1]) *
        a[Dw0 da; Dw′0] * b[Dn′0 db; Dn0]
end
function _als3s_tensor_R3(benv::BondEnv3site, a::MPSTensor, m::GenericMPSTensor{S, 4}) where {S}
    return @tensoropt Rb[Dn′1 Dn1; Dn′0 Dn0] :=
        benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(a[Dw1 da; Dw′1]) * conj(m[Dw′1 dm De1 Ds1; Dn′1]) *
        a[Dw0 da; Dw′0] * m[Dw′0 dm De0 Ds0; Dn′0]
end

"""
Calculate the 3-site norm
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a---m---b---┤
    |   ↓   ↓   ↓   |
    ├---ā---m̄---b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
using pre-calcuated partial contraction results.
"""
function _als_norm(
        ket::AbstractTensorMap{T, S, 4, 3}, benv_ket::AbstractTensorMap{T, S, 4, 3}
    ) where {T, S}
    return @tensor benv_ket[Dw1 De1 Ds1 Dn1; da dm db] *
        conj(ket[Dw1 De1 Ds1 Dn1; da dm db])
end

"""
    _als_tensor_S(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        xs::Vector{<:GenericMPSTensor}, i::Int
    ) where {T <: Number, S <: ElementarySpace}

Construct the overlap but with one of the bra bond tensor removed.
```
    i = 1               i = 2               i = 3
    ┌benv-┬---┬-----┐   ┌benv-┬---┬-----┐   ┌benv-┬---┬-----┐
    |      ╲ ╱      |   |      ╲ ╱      |   |      ╲ ╱      |
    ├---a₂--m₂--b₂--┤   ├---a₂--m₂--b₂--┤   ├---a₂--m₂--b₂--┤
    |   ↓   ↓   ↓   |   |   ↓   ↓   ↓   |   |   ↓   ↓   ↓   |
    ├--   --m̄---b̄---┤   ├---ā--   --b̄---┤   ├---ā---m̄--   --┤
    |      ╱ ╲      |   |      ╱ ╲      |   |      ╱ ╲      |
    └-----┴---┴-----┘   └-----┴---┴-----┘   └-----┴---┴-----┘
```
The ket part is provided by the partial contraction `benv_ket`.
"""
function _als_tensor_S(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        xs::Vector{<:GenericMPSTensor}, i::Int
    ) where {T <: Number, S <: ElementarySpace}
    @assert 1 <= i <= 3
    return if i == 1
        _als3s_tensor_S1(benv_ket, xs[2], xs[3])
    elseif i == 2
        _als3s_tensor_S2(benv_ket, xs[1], xs[3])
    else
        _als3s_tensor_S3(benv_ket, xs[1], xs[2])
    end
end

function _als3s_tensor_S1(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        m::GenericMPSTensor{S, 4}, b::MPSTensor
    ) where {T <: Number, S <: ElementarySpace}
    return @tensoropt Sa[Dw1 da; Dw′1] :=
        benv_ket[Dw1 De1 Ds1 Dn1; da dm db] *
        conj(m[Dw′1 dm De1 Ds1; Dn′1]) * conj(b[Dn′1 db; Dn1])
end
function _als3s_tensor_S2(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        a::MPSTensor, b::MPSTensor
    ) where {T <: Number, S <: ElementarySpace}
    return @tensoropt Sm[Dw′1 dm De1 Ds1; Dn′1] :=
        benv_ket[Dw1 De1 Ds1 Dn1; da dm db] *
        conj(a[Dw1 da; Dw′1]) * conj(b[Dn′1 db; Dn1])
end
function _als3s_tensor_S3(
        benv_ket::AbstractTensorMap{T, S, 4, 3},
        a::MPSTensor, m::GenericMPSTensor{S, 4}
    ) where {T <: Number, S <: ElementarySpace}
    return @tensoropt Sb[Dn′1 db; Dn1] :=
        benv_ket[Dw1 De1 Ds1 Dn1; da dm db] *
        conj(a[Dw1 da; Dw′1]) * conj(m[Dw′1 dm De1 Ds1; Dn′1])
end

"""
Calculate the 3-site ALS inner product ⟨a₁,m₁,b₁|a₂,m₂,b₂⟩
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a₂--m₂--b₂--┤
    |   ↓   ↓   ↓   |
    ├---ā₁--m̄₁--b̄₁--┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function inner_prod(
        benv::BondEnv3site, xs1::Vector{T}, xs2::Vector{T}
    ) where {T <: GenericMPSTensor}
    @assert length(xs1) == length(xs2) == 3
    return @tensoropt benv[Dw1 De1 Ds1 Dn1; Dw0 De0 Ds0 Dn0] *
        conj(xs1[1][Dw1 da; Dw′1]) *
        conj(xs1[2][Dw′1 dm De1 Ds1; Dn′1]) * conj(xs1[3][Dn′1 db; Dn1]) *
        xs2[1][Dw0 da; Dw′0] * xs2[2][Dw′0 dm De0 Ds0; Dn′0] * xs2[3][Dn′0 db; Dn0]
end

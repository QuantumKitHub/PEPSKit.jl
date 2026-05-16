"""
Calculate the reduced bond environment
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a--       --┤
    |   ↓           |
    ├---ā--       --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_a(benv::BondEnv3site, a::MPSTensor)
    return @tensoropt benv_a[Dpq1 De1 Ds1 Db1; Dpq0 De0 Ds0 Db0] :=
        benv[Da1 De1 Ds1 Db1; Da0 De0 Ds0 Db0] *
        a[Da0 da; Dpq0] * conj(a[Da1 da; Dpq1])
end

"""
Calculate the reduced bond environment
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├--       --b---┤
    |           ↓   |
    ├--       --b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_b(benv::BondEnv3site, b::MPSTensor)
    return @tensoropt benv_b[Da1 De1 Ds1 Duv1; Da0 De0 Ds0 Duv0] :=
        benv[Da1 De1 Ds1 Db1; Da0 De0 Ds0 Db0] *
        b[Duv0 db; Db0] * conj(b[Duv1 db; Db1])
end

function _als_tensor_Rp(
        benv_b::BondEnv3site, m::GenericMPSTensor{S, 4}
    ) where {S}
    return @tensoropt Rp[Da1 Dpq1; Da0 Dpq0] :=
        benv_b[Da1 De1 Ds1 Duv1; Da0 De0 Ds0 Duv0] *
        m[Dpq0 dm De0 Ds0; Duv0] * conj(m[Dpq1 dm De1 Ds1; Duv1])
end

function _als_tensor_Rq(
        benv_ab::BondEnv3site, mu::GenericMPSTensor{S, 4}
    ) where {S}
    return @tensoropt Rq[Dpq1 Dqm1; Dpq0 Dqm0] :=
        benv_ab[Dpq1 De1 Ds1 Duv1; Dpq0 De0 Ds0 Duv0] *
        mu[Dqm0 dm De0 Ds0; Duv0] * conj(mu[Dqm1 dm De1 Ds1; Duv1])
end

function _als_tensor_Ru(
        benv_ab::BondEnv3site, qm::GenericMPSTensor{S, 4}
    ) where {S}
    return @tensoropt Ru[Dmu1 Duv1; Dmu0 Duv0] :=
        benv_ab[Dpq1 De1 Ds1 Duv1; Dpq0 De0 Ds0 Duv0] *
        qm[Dpq0 dm De0 Ds0; Dmu0] * conj(qm[Dpq1 dm De1 Ds1; Dmu1])
end

function _als_tensor_Rv(
        benv_a::BondEnv3site, m::GenericMPSTensor{S, 4}
    ) where {S}
    return @tensoropt Rv[Duv1 Db1; Duv0 Db0] :=
        benv_a[Dpq1 De1 Ds1 Db1; Dpq0 De0 Ds0 Db0] *
        m[Dpq0 dm De0 Ds0; Duv0] * conj(m[Dpq1 dm De1 Ds1; Duv1])
end

"""
Calculate the network
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a₂==m₂==b₂--┤
    |   ↓           |
    ├---ā-        --┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_ket_a(
        benv_ket::AbstractTensorMap{T, S, 4, 3}, a::MPSTensor
    ) where {T, S}
    return @tensor benv_ket_a[Dpq1 De1 Ds1 Db1; dm db] :=
        conj(a[Da1 da; Dpq1]) * benv_ket[Da1 De1 Ds1 Db1; da dm db]
end

"""
Calculate the network
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a₂==m₂==b₂--┤
    |           ↓   |
    ├--        -b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_ket_b(
        benv_ket::AbstractTensorMap{T, S, 4, 3}, b::MPSTensor
    ) where {T, S}
    return @tensor benv_ket_b[Da1 De1 Ds1 Duv1; da dm] :=
        benv_ket[Da1 De1 Ds1 Db1; da dm db] * conj(b[Duv1 db; Db1])
end

"""
Calculate the network
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a₂==m₂==b₂--┤
    |   ↓       ↓   |
    ├---ā-     -b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_ket_ab(
        benv_ket_a::AbstractTensorMap{T, S, 4, 2}, b::MPSTensor
    ) where {T, S}
    return @tensor benv_ket_ab[Dpq1 De1 Ds1 Duv1; dm] :=
        benv_ket_a[Dpq1 De1 Ds1 Db1; dm db] * conj(b[Duv1 db; Db1])
end

"""
Calculate the network
```
    ┌benv-┬---┬-----┐
    |      ╲ ╱      |
    ├---a₂==m₂==b₂--┤
    |   ↓       ↓   |
    ├---ā-     -b̄---┤
    |      ╱ ╲      |
    └-----┴---┴-----┘
```
"""
function _benv_ket_ba(
        benv_ket_b::AbstractTensorMap{T, S, 4, 2}, a::MPSTensor
    ) where {T, S}
    return @tensor benv_ket_ab[Dpq1 De1 Ds1 Duv1; dm] :=
        conj(a[Da1 da; Dpq1]) * benv_ket_b[Da1 De1 Ds1 Duv1; da dm]
end

function _als_tensor_Sp(
        benv_ket_b::AbstractTensorMap{T, S, 4, 2},
        m::GenericMPSTensor{S, 4}
    ) where {T, S}
    return @tensoropt Sp[Da1 da; Dpq1] := benv_ket_b[Da1 De1 Ds1 Duv1; da dm] *
        conj(m[Dpq1 dm De1 Ds1; Duv1])
end

function _als_tensor_Sq(
        benv_ket_ab::AbstractTensorMap{T, S, 4, 1},
        mu::GenericMPSTensor{S, 4}
    ) where {T, S}
    return @tensor Sq[Dpq1; Dqm1] := benv_ket_ab[Dpq1 De1 Ds1 Duv1; dm] *
        conj(mu[Dqm1 dm De1 Ds1; Duv1])
end

function _als_tensor_Su(
        benv_ket_ab::AbstractTensorMap{T, S, 4, 1},
        qm::GenericMPSTensor{S, 4}
    ) where {T, S}
    return @tensor Su[Dmu1; Duv1] := benv_ket_ab[Dpq1 De1 Ds1 Duv1; dm] *
        conj(qm[Dpq1 dm De1 Ds1; Dmu1])
end

function _als_tensor_Sv(
        benv_ket_a::AbstractTensorMap{T, S, 4, 2},
        m::GenericMPSTensor{S, 4}
    ) where {T, S}
    return @tensoropt Sv[Duv1 db; Db1] := benv_ket_a[Dpq1 De1 Ds1 Db1; dm db] *
        conj(m[Dpq1 dm De1 Ds1; Duv1])
end

function _solveproj_als_pinv(R::BondEnv, S::MPSBondTensor; kwargs...)
    R_inv = pinv(R; kwargs...)
    x = R_inv * permute(S, ((1, 2), ()))
    twistdual!(x, 1:2)
    return permute(x, ((1,), (2,)))
end

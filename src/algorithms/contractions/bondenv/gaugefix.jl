"""
Find positive approximant `Z† Z` of a norm tensor `benv`
(returns the "half environment" `Z`)
```
    ┌-----------------┐     ┌---------------┐
    | ┌----┐          |     |               |
    └-|    |-- 3  4 --┘     └-- Z -- 3  4 --┘
      |benv|            =       ↓
    ┌-|    |-- 1  2 --┐     ┌-- Z†-- 1  2 --┐
    | └----┘          |     |               |
    └-----------------┘     └---------------┘
```
"""
function positive_approx(benv::AbstractTensorMap{T, S, N, N}) where {T, S, N}
    # eigen-decomposition: benv = U D U'
    D, U = eigh_full!(project_hermitian(benv))
    # determine if `env` is (mostly) positive or negative
    sgn = sign(sum(D.data))
    # When optimizing the truncation of a bond,
    # its environment can always be multiplied by a number.
    # If `benv` is negative (e.g. obtained approximately from CTMRG),
    # we can multiply it by (-1).
    data = D.data
    @inbounds for i in eachindex(data)
        d = (sgn == -1) ? -data[i] : data[i]
        data[i] = (d > 0) ? sqrt(d) : zero(d)
    end
    Z = D * U'
    return Z
end

"""
Use QR decomposition to fix gauge of the half bond environment `Z`.
The reduced bond tensors `a`, `b` and `Z` are arranged as
```
    ┌---------------┐
    |               |
    └---Z---a---b---┘
        |   ↓   ↓
        ↓
```
Reference: 
- Physical Review B 90, 064425 (2014)
- Physical Review B 92, 035142 (2015)
"""
function fixgauge_benv(
        Z::AbstractTensorMap{T, S, 1, 2}, a::MPSTensor, b::MPSTensor
    ) where {T <: Number, S <: ElementarySpace}
    @assert !isdual(space(Z, 1))
    @assert !isdual(space(a, 2))
    @assert !isdual(space(b, 2))
    #= QR/LQ decomposition of Z 

        3 - Z - 2   =   2 - L - 1   3 - QL - 1
            ↓                           ↓
            1                           2

                    =   1 - QR - 3  1 - R - 2
                            ↓
                            2
    =#
    QL, L = left_orth!(permute(Z, ((2, 1), (3,)); copy = true); positive = true)
    QR, R = left_orth!(permute(Z, ((3, 1), (2,)); copy = true); positive = true)
    @debug "cond(L) = $(LinearAlgebra.cond(L)); cond(R) = $(LinearAlgebra.cond(R))"
    # TODO: find a better way to fix gauge that avoids `inv`
    Linv, Rinv = inv(L), inv(R)
    #= fix gauge of Z, a, b
        ┌---------------------------------------┐
        |                                       |
        └---Z---Rinv)---(R--a)--(b--L)---(Linv--┘
            |               ↓    ↓
            ↓

        -1 - R - 1 - a - -3   -1 - b - 1 - L - -3
                     ↓             ↓        
                    -2            -2

        ┌-----------------------------------------┐
        |                                         |
        └---Z-- 1 --Rinv-- -2      -3 --Linv-- 2 -┘
            ↓
            -1
    =#
    @plansor a[-1 -2; -3] := R[-1; 1] * a[1 -2; -3]
    @plansor b[-1 -2; -3] := b[-1 -2; 1] * L[-3; 1]
    @plansor Z[-1; -2 -3] := Z[-1; 1 2] * Rinv[1; -2] * Linv[2; -3]
    (isdual(space(R, 1)) == isdual(space(R, 2))) && twist!(a, 1)
    (isdual(space(L, 1)) == isdual(space(L, 2))) && twist!(b, 3)
    return Z, a, b, (Linv, Rinv)
end

"""
Apply the gauge transformation `Rinv` for `Z`
```
    ┌-----------------------┐
    └---Z--(X)--Rinv--   ---┘
        ↓
```
to `X`. For example, when `X` is a `PEPSTensor`,
```
        -2
         |
    -5 - X - 1 - Rinv - -3
         | ╲ 
        -4  -1
```
"""
function _fixgauge_benvX(X::PEPSOrth, Rinv::MPSBondTensor)
    return @plansor X[-1 -2 -3 -4] := X[-1 1 -3 -4] * Rinv[1; -2]
end
function _fixgauge_benvX(X::PEPSTensor, Rinv::MPSBondTensor)
    return @plansor X[-1; -2 -3 -4 -5] := X[-1; -2 1 -4 -5] * Rinv[1; -3]
end
function _fixgauge_benvX(X::PEPOOrth, Rinv::MPSBondTensor)
    return @plansor X[-1 -2 -3 -4 -5] := X[-1 -2 1 -4 -5] * Rinv[1; -3]
end
function _fixgauge_benvX(X::PEPOTensor, Rinv::MPSBondTensor)
    return @plansor X[-1 -2; -3 -4 -5 -6] := X[-1 -2; -3 1 -5 -6] * Rinv[1; -4]
end

"""
Apply the gauge transformation `Linv` for `Z`
```
    ┌-----------------------┐
    └---Z---  ---Linv--(Y)--┘
        ↓
```
to `Y`. For example, when `Y` is a `PEPSTensor`,
```
                   -2
                    |
    -5 - Linv - 1 - Y - -3
                    | ╲
                   -4  -1
```
"""
function _fixgauge_benvY(Y::PEPSOrth, Linv::MPSBondTensor)
    return @plansor Y[-1 -2 -3 -4] := Y[-1 -2 -3 1] * Linv[1; -4]
end
function _fixgauge_benvY(Y::PEPSTensor, Linv::MPSBondTensor)
    return @plansor Y[-1; -2 -3 -4 -5] := Y[-1; -2 -3 -4 1] * Linv[1; -5]
end
function _fixgauge_benvY(Y::PEPOOrth, Linv::MPSBondTensor)
    return @plansor Y[-1 -2 -3 -4 -5] := Y[-1 -2 -3 -4 1] * Linv[1; -5]
end
function _fixgauge_benvY(Y::PEPOTensor, Linv::MPSBondTensor)
    return @plansor Y[-1 -2; -3 -4 -5 -6] := Y[-1 -2; -3 -4 -5 1] * Linv[1; -6]
end

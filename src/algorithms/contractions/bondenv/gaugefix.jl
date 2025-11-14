"""
Replace bond environment `benv` by its positive approximant `Z† Z`
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
function positive_approx(benv::BondEnv)
    # eigen-decomposition: benv = U D U'
    D, U = eigh_full((benv + benv') / 2)
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
        Z::AbstractTensorMap{T, S, 1, 2},
        a::AbstractTensorMap{T, S, 1, 2},
        b::AbstractTensorMap{T, S, 2, 1},
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
    QL, L = left_orth(permute(Z, ((2, 1), (3,))))
    QR, R = left_orth(permute(Z, ((3, 1), (2,))))
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
    @plansor a[-1; -2 -3] := R[-1; 1] * a[1; -2 -3]
    @plansor b[-1 -2; -3] := b[-1 -2; 1] * L[-3; 1]
    @plansor Z[-1; -2 -3] := Z[-1; 1 2] * Rinv[1; -2] * Linv[2; -3]
    (isdual(space(R, 1)) == isdual(space(R, 2))) && twist!(a, 1)
    (isdual(space(L, 1)) == isdual(space(L, 2))) && twist!(b, 3)
    return Z, a, b, (Linv, Rinv)
end

"""
When the (half) bond environment `Z` consists of 
two `PEPSOrth` or `PEPOOrth` tensors `X`, `Y` as
```
    ┌-----------------------┐
    |                       |
    └---Z---(X)--   --(Y)---┘
        ↓
```
apply the gauge transformation `Linv`, `Rinv` for `Z` to `X`, `Y`:
```
        -1                                     -1
         |                                      |
    -4 - X - 1 - Rinv - -2      -4 - Linv - 1 - Y - -2
         |                                      |
        -3                                     -3
    
        -2                                     -2
         |                                      |
    -5 - X - 1 - Rinv - -3      -5 - Linv - 1 - Y - -3
         | ╲                                    | ╲
        -4  -1                                 -4  -1
```
"""
function _fixgauge_benvXY(
        X::PEPSOrth, Y::PEPSOrth, Linv::MPSBondTensor, Rinv::MPSBondTensor,
    )
    @plansor X[-1 -2 -3 -4] := X[-1 1 -3 -4] * Rinv[1; -2]
    @plansor Y[-1 -2 -3 -4] := Y[-1 -2 -3 1] * Linv[1; -4]
    return X, Y
end
function _fixgauge_benvXY(
        X::PEPOOrth, Y::PEPOOrth, Linv::MPSBondTensor, Rinv::MPSBondTensor,
    )
    @plansor X[-1 -2 -3 -4 -5] := X[-1 -2 1 -4 -5] * Rinv[1; -3]
    @plansor Y[-1 -2 -3 -4 -5] := Y[-1 -2 -3 -4 1] * Linv[1; -5]
    return X, Y
end

"""
Environment assisted truncation (EAT) of the bond between `a` and `b`.
The positive-definite environment is first expressed as `Z† Z`. 
```
            ↑
    |-----→ Z ←-----|
    |               |
    |    ↑     ↑    |
    |←-- a === b --→|
```

Reference: Physical Review B 98, 085155 (2018)
"""
function ea_truncate(
    Z::AbstractTensorMap{S,1,2},
    a::AbstractTensorMap{S},
    b::AbstractTensorMap{S},
    trscheme::TruncationScheme,
) where {S<:ElementarySpace} 
    @assert [isdual(space(Z, ax)) for ax in 1:3] == [0, 1, 1]
end

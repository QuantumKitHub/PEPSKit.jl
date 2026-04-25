const NNGate{T, S} = AbstractTensorMap{T, S, 2, 2}

"""
Apply 1-site `gate` on the PEPS or PEPO tensor `a`.
"""
function _apply_sitegate(
        a::PEPSTensor, gate::AbstractTensorMap{T, S, 1, 1}; purified::Bool = true
    ) where {T, S}
    @assert purified
    return gate * a
end

function _apply_sitegate(
        a::PEPOTensor, gate::AbstractTensorMap{T, S, 1, 1}; purified::Bool = true
    ) where {T, S}
    @plansor a′[p1 p2; n e s w] := gate[p1; p] * a[p p2; n e s w]
    if !purified
        @plansor a′[p1 p2; n e s w] := a′[p1 p; n e s w] * gate[p; p2]
    end
    return a′
end

"""
$(SIGNATURES)

Apply 2-site `gate` on the reduced bond tensors `a`, `b`
```
    -1← a --- 3 --- b ← -4          -2         -3
        ↓           ↓               ↓           ↓
        1           2               |----gate---|
        ↓           ↓       or      ↓           ↓
        |----gate---|               1           2
        ↓           ↓               ↓           ↓
        -2         -3           -1← a --- 3 --- b ← -4
```
"""
function _apply_gate(a::MPSTensor, b::MPSTensor, gate::NNGate, trunc::TruncationStrategy)
    V = space(b, 1)
    need_flip = isdual(V)
    if isdual(space(a, 2))
        @tensor a2b2[-1 -2; -3 -4] := gate[1 2; -2 -3] * a[-1 1 3] * b[3 2 -4]
    else
        @tensor a2b2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * a[-1 1 3] * b[3 2 -4]
    end
    a, s, b, ϵ = svd_trunc!(a2b2; trunc, alg = LAPACK_QRIteration())
    a, b = absorb_s(a, s, b)
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    b = permute(b, ((1, 2), (3,)))
    return a, s, b, ϵ
end

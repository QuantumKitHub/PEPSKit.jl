"""
$(SIGNATURES)

Use QR decomposition on two tensors `A`, `B` connected by a bond to get the reduced tensors.
When `A`, `B` are PEPSTensors,
```
        2                   1                                   1
        |                   |                                   |
    5 -A/B- 3   ====>   4 - X ← 2   1 ← a - 3   1 - b → 3   4 → Y - 2
        | ↘                 |            ↘           ↘          |
        4   1               3             2           2         3
```
When `A`, `B` are PEPOTensors, 
- If `gate_ax = 1`
```
    2   3                1  2                                1  2
      ↘ |                 ↘ |                                 ↘ |
    6 -A/B- 4   ====>   5 - X ← 3   1 ← a - 3   1 - b → 3   5 → Y - 3
        | ↘                 |            ↘           ↘          |
        5   1               4             2           2         4
```
- If `gate_ax = 2`
```
    2   3                   2         2           2             2
      ↘ |                   |          ↘           ↘            |
    6 -A/B- 4   ====>   5 - X ← 3   1 ← a - 3   1 - b → 3   5 → Y - 3
        | ↘                 | ↘                                 | ↘
        5   1               4  1                                4  1
```
"""
function _qr_bond(A::PT, B::PT; gate_ax::Int = 1) where {PT <: Union{PEPSTensor, PEPOTensor}}
    @assert 1 <= gate_ax <= numout(A)
    permA, permB, permX, permY = if A isa PEPSTensor
        ((2, 4, 5), (1, 3)), ((2, 3, 4), (1, 5)), (1, 4, 2, 3), Tuple(1:4)
    else
        if gate_ax == 1
            ((2, 3, 5, 6), (1, 4)), ((2, 3, 4, 5), (1, 6)), (1, 2, 5, 3, 4), Tuple(1:5)
        else
            ((1, 3, 5, 6), (2, 4)), ((1, 3, 4, 5), (2, 6)), (1, 2, 5, 3, 4), Tuple(1:5)
        end
    end
    X, a = left_orth(permute(A, permA); positive = true)
    Y, b = left_orth(permute(B, permB); positive = true)
    # no longer needed after TensorKit 0.15
    # @assert !isdual(space(a, 1))
    # @assert !isdual(space(b, 1))
    X, Y = permute(X, permX), permute(Y, permY)
    b = permute(b, ((3, 2), (1,)))
    return X, a, b, Y
end

"""
$(SIGNATURES)

Reconstruct the tensors connected by a bond from their `_qr_bond` results.
For PEPSTensors,
```
        -2                             -2
        |                               |
    -5- X - 1 - a - -3     -5 - b - 1 - Y - -3
        |        ↘               ↘      |
        -4        -1              -1   -4
```
For PEPOTensors
```
    -2  -3                          -2  -3
      ↘ |                             ↘ |
    -6- X - 1 - a - -4     -6 - b - 1 - Y - -4
        |        ↘               ↘      |
        -5        -1              -1   -5

        -3   -2              -2        -3
        |      ↘               ↘        |
    -6- X - 1 - a - -4     -6 - b - 1 - Y - -4
        | ↘                             | ↘
        -5 -1                          -5  -1
```
"""
function _qr_bond_undo(X::PEPSOrth, a::AbstractTensorMap, b::AbstractTensorMap, Y::PEPSOrth)
    @tensor A[-1; -2 -3 -4 -5] := X[-2 1 -4 -5] * a[1 -1 -3]
    @tensor B[-1; -2 -3 -4 -5] := b[-5 -1 1] * Y[-2 -3 -4 1]
    return A, B
end
function _qr_bond_undo(X::PEPOOrth, a::AbstractTensorMap, b::AbstractTensorMap, Y::PEPOOrth)
    if !isdual(space(a, 2))
        @tensor A[-1 -2; -3 -4 -5 -6] := X[-2 -3 1 -5 -6] * a[1 -1 -4]
        @tensor B[-1 -2; -3 -4 -5 -6] := b[-6 -1 1] * Y[-2 -3 -4 -5 1]
    else
        @tensor A[-1 -2; -3 -4 -5 -6] := X[-1 -3 1 -5 -6] * a[1 -2 -4]
        @tensor B[-1 -2; -3 -4 -5 -6] := b[-6 -2 1] * Y[-1 -3 -4 -5 1]
    end
    return A, B
end

"""
$(SIGNATURES)

Apply 2-site `gate` on the reduced matrices `a`, `b`
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
function _apply_gate(
        a::AbstractTensorMap, b::AbstractTensorMap,
        gate::AbstractTensorMap{T, S, 2, 2}, trunc::TruncationStrategy
    ) where {T <: Number, S <: ElementarySpace}
    V = space(b, 1)
    need_flip = isdual(V)
    if isdual(space(a, 2))
        @tensor a2b2[-1 -2; -3 -4] := gate[1 2; -2 -3] * a[-1 1 3] * b[3 2 -4]
    else
        @tensor a2b2[-1 -2; -3 -4] := gate[-2 -3; 1 2] * a[-1 1 3] * b[3 2 -4]
    end
    trunc = if trunc isa FixedSpaceTruncation
        need_flip ? truncspace(flip(V)) : truncspace(V)
    else
        trunc
    end
    a, s, b, ϵ = svd_trunc!(a2b2; trunc, alg = LAPACK_QRIteration())
    a, b = absorb_s(a, s, b)
    if need_flip
        a, s, b = flip(a, numind(a)), _fliptwist_s(s), flip(b, 1)
    end
    return a, s, b, ϵ
end

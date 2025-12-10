struct LocalApprox <: ApproximateAlgorithm
    trunc::TruncationStrategy
end

"""
Calculate the QR decomposition of 2-layer PEPO tensor
```
        ↓ ╱
    ----A2-←-           ┌-←-
      ╱ |               |
        ↓     =  (Q)-←--R
        | ╱             |
    ----A1-←-           └-←-
      ╱ ↓
```
Only `R` is calculated and returned.
"""
function qr_twolayer(A1::PEPOTensor, A2::PEPOTensor)
    @assert isdual(space(A1, 4)) && isdual(space(A2, 4))
    A2′ = twistdual(A2, [2, 3, 5, 6])
    A1′ = twistdual(A1, [1, 3, 5, 6])
    @tensoropt MdagM[x2 z z′ x2′] :=
        conj(A2[z z2; Y2 x2 y2 X2]) * A2′[z′ z2; Y2 x2′ y2 X2]
    @tensoropt MdagM[x1 x2; x1′ x2′] := MdagM[x2 z z′ x2′] *
        conj(A1[z1 z; Y1 x1 y1 X1]) * A1′[z1 z′; Y1 x1′ y1 X1]
    # TODO: switch to eigh
    _, s, R = svd_compact!(MdagM)
    R = sdiag_pow(s, 0.5) * R
    return R
end

"""
Calculate the LQ decomposition of 2-layer PEPO tensor
```
        ↓ ╱  
    --←-A2---    -←-┐
      ╱ |           |
        ↓      =    L--←-(Q)
        | ╱         |
    --←-A1---    -←-┘
      ╱ ↓    
```
Only `L` is calculated and returned.
"""
function lq_twolayer(A1::PEPOTensor, A2::PEPOTensor)
    @assert !isdual(space(A1, 6)) && !isdual(space(A2, 6))
    A2′ = twistdual(A2, [2, 3, 4, 5])
    A1′ = twistdual(A1, [1, 3, 4, 5])
    @tensoropt MMdag[x2 z z′ x2′] :=
        A2[z z2; Y2 X2 y2 x2] * conj(A2′[z′ z2; Y2 X2 y2 x2′])
    @tensoropt MMdag[x1 x2; x1′ x2′] := MMdag[x2 z z′ x2′] *
        A1[z1 z; Y1 X1 y1 x1] * conj(A1′[z1 z′; Y1 X1 y1 x1′])
    # TODO: switch to eigh
    L, s, _ = svd_compact!(MMdag)
    L = L * sdiag_pow(s, 0.5)
    return L
end

"""
Find the local projector `P1`, `P2` for the 
following truncation of two layers of InfinitePEPO
```
        ↓ ╱                 ↓ ╱
    ----A2-←-|╲       ╱|--←-B2---
      ╱ |    | ╲     ╱ |  ╱ |
        ↓    |P1├-←-┤P2|    ↓
        | ╱  | ╱     ╲ |    | ╱
    ----A1-←-|╱       ╲|--←-B1---
      ╱ ↓                 ╱ ↓
```
Reference: Physical Review B 100, 035449 (2019)
"""
function localapprox_projector(
        A1::PEPOTensor, A2::PEPOTensor, B1::PEPOTensor, B2::PEPOTensor;
        trunc::TruncationStrategy
    )
    R1 = qr_twolayer(A1, A2)
    R2 = lq_twolayer(B1, B2)
    u, s, vh = svd_compact!(R1 * R2)
    u, s, vh, ϵ = _truncate_compact((u, s, vh), trunc)
    sinv_sqrt = sdiag_pow(s, -0.5)
    P1 = R2 * vh' * sinv_sqrt
    P2 = sinv_sqrt * u' * R1
    return P1, s, P2, ϵ
end

"""
Compute an approximation to the product of two 1-layer InfinitePEPOs `ρ1`, `ρ2`
with virtual bond truncated with `LocalApprox`.
"""
function MPSKit.approximate(ρ1::InfinitePEPO, ρ2::InfinitePEPO, alg::LocalApprox)
    @assert size(ρ1) == size(ρ2)
    @assert size(ρ1, 3) == size(ρ2, 3) == 1
    Nr, Nc, = size(ρ1)
    ρ1 = standardize_virtual_spaces(ρ1)
    ρ2 = standardize_virtual_spaces(ρ2)
    # x-bond projectors: [r, c] on bond [r, c]--[r, c+1]
    Pxs = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        P1, s, P2, ϵ = localapprox_projector(
            ρ1[r, c], ρ2[r, c], ρ1[r, _next(c, Nc)], ρ2[r, _next(c, Nc)];
            trunc = alg.trunc
        )
        return (P1, P2)
    end
    # y-bond projectors: [r, c] on bond [r, c]--[r-1, c]
    Pys = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        # TODO: reduce repeated rotations
        P1, s, P2, ϵ = localapprox_projector(
            rotr90(ρ1[r, c]), rotr90(ρ2[r, c]),
            rotr90(ρ1[_prev(r, Nr), c]), rotr90(ρ2[_prev(r, Nr), c]);
            trunc = alg.trunc
        )
        return (P1, P2)
    end
    # apply projectors
    As = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        Pw, Pe = Pxs[r, _prev(c, Nc)][2], Pxs[r, c][1]
        Pn, Ps = Pys[r, c][1], Pys[_next(r, Nr), c][2]
        @tensoropt A[p1 p2; n e s w] :=
            (ρ1[r, c])[p1 p; n1 e1 s1 w1] * (ρ2[r, c])[p p2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[s; s1 s2] * Pw[w; w1 w2]
        return A
    end
    return InfinitePEPO(cat(As; dims = 3))
end

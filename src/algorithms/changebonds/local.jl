"""
$(TYPEDEF)

Abstract super type for algorithms to change virtual bonds
in two-dimensional tensor networks
"""
abstract type BondChangeAlgorithm end

"""
$(TYPEDEF)

Algorithm to truncate virtual bonds of a two-layer network 
with projectors that minimizes the local cost function,
which is the 2-norm of (e.g. for two layers of iPEPO)
```
        ↓ ╱      ↓ ╱            ↓ ╱                 ↓ ╱
    ----A2---←---B2---      ----A2-←-|╲       ╱|--←-B2---
      ╱ |      ╱ |            ╱ |    | ╲     ╱ |  ╱ |
        ↓        ↓       -      ↓    |P1├-←-┤P2|    ↓
        | ╱      | ╱            | ╱  | ╱     ╲ |    | ╱
    ----A1---←---B1---      ----A1-←-|╱       ╲|--←-B1---
      ╱ ↓      ╱ ↓            ╱ ↓                 ╱ ↓
```
on each bond of the network.
"""
struct LocalTruncation <: BondChangeAlgorithm
    trunc::TruncationStrategy
end

"""
Calculate the QR decomposition of 2-layer PEPO tensor
with the east virtual legs transferred to the R tensor
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
    MdagM = _get_MdagM(A1, A2)
    D, R = eigh_full!(MdagM)
    project_psd!(D)
    R = sdiag_pow(D, 0.5) * R'
    return R
end
function _get_MdagM(A1::PEPOTensor, A2::PEPOTensor)
    @assert isdual(virtualspace(A1, EAST)) && isdual(virtualspace(A2, EAST))
    A2′ = twistdual(A2, [2, 3, 5, 6])
    A1′ = twistdual(A1, [1, 3, 5, 6])
    @tensoropt MdagM[x2 z z′ x2′] :=
        conj(A2[z z2; Y2 x2 y2 X2]) * A2′[z′ z2; Y2 x2′ y2 X2]
    @tensoropt MdagM[x1 x2; x1′ x2′] := MdagM[x2 z z′ x2′] *
        conj(A1[z1 z; Y1 x1 y1 X1]) * A1′[z1 z′; Y1 x1′ y1 X1]
    project_hermitian!(MdagM)
    return MdagM
end

"""
Calculate the LQ decomposition of 2-layer PEPO tensor
with the west virtual legs transferred to the L tensor
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
    MMdag = _get_MMdag(A1, A2)
    D, L = eigh_full!(MMdag)
    project_psd!(D)
    L = L * sdiag_pow(D, 0.5)
    return L
end
function _get_MMdag(A1::PEPOTensor, A2::PEPOTensor)
    @assert !isdual(virtualspace(A1, WEST)) && !isdual(virtualspace(A2, WEST))
    A2′ = twistnondual(A2, [2, 3, 4, 5])
    A1′ = twistnondual(A1, [1, 3, 4, 5])
    @tensoropt MMdag[x2 z z′ x2′] :=
        A2′[z z2; Y2 X2 y2 x2] * conj(A2[z′ z2; Y2 X2 y2 x2′])
    @tensoropt MMdag[x1 x2; x1′ x2′] := MMdag[x2 z z′ x2′] *
        A1′[z1 z; Y1 X1 y1 x1] * conj(A1[z1 z′; Y1 X1 y1 x1′])
    project_hermitian!(MMdag)
    return MMdag
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
function virtual_projector(
        A1::PEPOTensor, A2::PEPOTensor, B1::PEPOTensor, B2::PEPOTensor;
        trunc::TruncationStrategy
    )
    R1 = qr_twolayer(A1, A2)
    R2 = lq_twolayer(B1, B2)
    u, s, vh, ϵ = svd_trunc!(R1 * R2; trunc)
    sinv_sqrt = sdiag_pow(s, -0.5)
    P1 = R2 * vh' * sinv_sqrt
    P2 = sinv_sqrt * u' * R1
    return P1, P2, (; s, ϵ)
end

"""
Truncate virtual bonds of the product of two 1-layer
InfinitePEPOs `ρ1`, `ρ2` with `LocalTruncation`.
"""
function MPSKit.changebonds(ρ1::InfinitePEPO, ρ2::InfinitePEPO, alg::LocalTruncation)
    # sanity checks
    (size(ρ1) == size(ρ2)) || error("Input PEPOs have different unit cell sizes.")
    (size(ρ1, 3) == 1) || error("ρ1 should have only one layer.")
    (size(ρ2, 3) == 1) || error("ρ2 should have only one layer.")
    all(all.(_check_virtual_dualness(ρ1))) || error("East and north virtual spaces in ρ1 should be dual spaces.")
    all(all.(_check_virtual_dualness(ρ2))) || error("East and north virtual spaces in ρ2 should be dual spaces.")
    # x-bond projectors: [r, c] on bond [r, c]--[r, c+1]
    Nr, Nc, = size(ρ1)
    Pxs_info = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        # TODO: support SiteDependentTruncation
        return virtual_projector(
            ρ1[r, c], ρ2[r, c], ρ1[r, _next(c, Nc)], ρ2[r, _next(c, Nc)];
            trunc = alg.trunc
        )
    end
    # y-bond projectors: [r, c] on bond [r, c]--[r-1, c]
    Pys_info = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        # TODO: reduce repeated rotations
        return virtual_projector(
            rotr90(ρ1[r, c]), rotr90(ρ2[r, c]),
            rotr90(ρ1[_prev(r, Nr), c]), rotr90(ρ2[_prev(r, Nr), c]);
            trunc = alg.trunc
        )
    end
    # apply projectors
    As = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        Pw, Pe = Pxs_info[r, _prev(c, Nc)][2], Pxs_info[r, c][1]
        Pn, Ps = Pys_info[r, c][1], Pys_info[_next(r, Nr), c][2]
        @tensoropt A[p1 p2; n e s w] :=
            (ρ1[r, c])[p1 p; n1 e1 s1 w1] * (ρ2[r, c])[p p2; n2 e2 s2 w2] *
            Pn[n1 n2; n] * Pe[e1 e2; e] * Ps[s; s1 s2] * Pw[w; w1 w2]
        return A
    end
    info = (; Pxs_info, Pys_info)
    return InfinitePEPO(cat(As; dims = 3)), info
end

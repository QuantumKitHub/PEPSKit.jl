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
struct LocalTruncation
    trunc::TruncationStrategy
end

"""
Calculate the `left_orth` of 2-layer PEPO tensor
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
function left_orth_twolayer(A1::PEPOTensor, A2::PEPOTensor)
    MdagM = _get_MdagM(A1, A2)
    D, R = eigh_full!(MdagM)
    # remove small negative eigenvalues due to numerical noises
    T = eltype(D.data)
    D.data .= sqrt.(max.(zero(T), D.data))
    return lmul!(D, R')
end
function _get_MdagM(A1::PEPOTensor, A2::PEPOTensor)
    @assert isdual(virtualspace(A1, EAST)) && isdual(virtualspace(A2, EAST))
    A2′ = twistdual(A2, [2, 3, 5, 6])
    A1′ = twistdual(A1, [1, 3, 5, 6])
    @autoopt @tensor MdagM[Dx1 Dx2; Dx1′ Dx2′] :=
        conj(A2[dz dz2; DY2 Dx2 Dy2 DX2]) * A2′[dz′ dz2; DY2 Dx2′ Dy2 DX2] *
        conj(A1[dz1 dz; DY1 Dx1 Dy1 DX1]) * A1′[dz1 dz′; DY1 Dx1′ Dy1 DX1]
    project_hermitian!(MdagM)
    return MdagM
end

"""
Calculate the `right_orth` of 2-layer PEPO tensor
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
function right_orth_twolayer(A1::PEPOTensor, A2::PEPOTensor)
    MMdag = _get_MMdag(A1, A2)
    D, L = eigh_full!(MMdag)
    # remove small negative eigenvalues due to numerical noises
    T = eltype(D.data)
    D.data .= sqrt.(max.(zero(T), D.data))
    return rmul!(L, D)
end
function _get_MMdag(A1::PEPOTensor, A2::PEPOTensor)
    @assert !isdual(virtualspace(A1, WEST)) && !isdual(virtualspace(A2, WEST))
    A2′ = twistnondual(A2, [2, 3, 4, 5])
    A1′ = twistnondual(A1, [1, 3, 4, 5])
    @autoopt @tensor MMdag[Dx1 Dx2; Dx1′ Dx2′] :=
        A2′[dz dz2; DY2 DX2 Dy2 Dx2] * conj(A2[dz′ dz2; DY2 DX2 Dy2 Dx2′]) *
        A1′[dz1 dz; DY1 DX1 Dy1 Dx1] * conj(A1[dz1 dz′; DY1 DX1 Dy1 Dx1′])
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
    R1 = left_orth_twolayer(A1, A2)
    R2 = right_orth_twolayer(B1, B2)
    u, s, vh, ϵ = svd_trunc!(R1 * R2; trunc)
    sinv_sqrt = sdiag_pow(s, -0.5)
    P1 = R2 * vh' * sinv_sqrt
    P2 = sinv_sqrt * u' * R1
    return P1, P2, (; s, ϵ)
end

"""
    compress((ρ1, ρ2), alg::LocalTruncation)

Compress two 1-layer iPEPOs into a 1-layer iPEPO by truncating the virtual bonds
with `LocalTruncation`. In the tuple `(ρ1, ρ2)`, `ρ1` is the lower layer and
`ρ2` is the upper layer.
"""
function compress(ρs::Tuple{<:InfinitePEPO, <:InfinitePEPO}, alg::LocalTruncation)
    ρ1, ρ2 = ρs
    # sanity checks
    size(ρ1) == size(ρ2) || error("Input PEPOs have different unit cell sizes.")
    size(ρ1, 3) == 1 || error("ρ1 should have only one layer.")
    size(ρ2, 3) == 1 || error("ρ2 should have only one layer.")
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
    ρ1′, ρ2′ = rotr90.(unitcell(ρ1)), rotr90.(unitcell(ρ2))
    Pys_info = map(Iterators.product(1:Nr, 1:Nc)) do (r, c)
        return virtual_projector(
            ρ1′[r, c], ρ2′[r, c], ρ1′[_prev(r, Nr), c], ρ2′[_prev(r, Nr), c];
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

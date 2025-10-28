#= 
# Mixed canonical form of an open boundary MPS
```
    |ψ⟩ =  M[1]---...---M[N]
            ↓            ↓
```
The bond between `M[n]` and `M[n+1]` is called 
the n-th (internal) bond (n = 1, ..., N - 1).

We perform QR and LQ decompositions: starting from
```
    M[1]---  =  Qa[1]-*-R[1]---
    ↓           ↓

    ---M[N]  =  --L[N-1]-*-Qb[N]
        ↓                   ↓
```
we successively calculate
```
    ---R[n-1]---M[n]---  =  ---Qa[n]-*-R[n]---- (n = 2, ..., N - 1)
                ↓               ↓

    --M[n+1]-*-L[n+1]--  =  ---L[n]-*-Qb[n+1]-- (n = N - 2, ..., 1)
        ↓                               ↓
```
Here `-*-` on the bond means a twist should be applied if
the codomain of R[n], Qb[n+1], L[n+1] is a dual space. 

NOTE: 
In TensorKit, the `isdual` of the domain and codomain 
of `R[n]` and `L[n]` for a given `n` are the same. 

For each bond (n = 1, ..., N - 1), we perform SVD
```
    R[n] L[n] = U[n]-←-s[n]-←-V†[n] (n = 1, ..., N - 1)
```
Then we define the projectors together with the Schmidt weight
```
    ---Pa[n]-←- = L[n] V[n]-←-(1/√s[n])-←-
    -←-Pb[n]--- = -←-(1/√s[n])-←-U†[n] R[n]
```
Since the domain and the codomain of R[n] and L[n] has the same `isdual`, 
the product `Pa Pb` is the identity operator:
```
    Pa[n]-←-Pb[n] = L[n] (R[n] L[n])⁻¹ R[n] = 1
```
The `isdual` for the domain and codomain of `Pa[n] Pb[n]` are also the same.

Note that when `Pa[n] Pb[n]` is identity on a dual space, 
a twist should be applied to put it to the bond. 

The canonical form is then defined by
```
    -←-M̃[n]-←- = -←-Pb[n-1]---M[n]-*-Pa[n]-←-
        ↓                      ↓
```
`-*-` means a twist should be applied if the codomain of `Pa[n]` is a dual space. 

Note that
```
    M̃[n]
    = 1/√s[n-1]←-U†[n-1](R[n-1]--M[n])-*-L[n] V[n]←-1/√s[n]
    = 1/√s[n-1]←-U†[n-1] Qa[n] (R[n]-*-L[n]) V[n]←-1/√s[n]
    = 1/√s[n-1]←-U†[n-1] Qa[n] U[n]←-s[n]←-(V†[n] V[n])←-1/√s[n]
    = 1/√s[n-1]←-U†[n-1] Qa[n] U[n]←-√s[n]
```
Then `M̃[n]` (n = 1, ..., N - 1) satisfies the (generalized) left-orthogonal condition
```
    ┌---←--M̃[n]--←-     ┌-←- 2
    |       |           |       
    s[n-1]  ↓       =   s[n]    (s[0] = 1)
    |       |           |
    └---→--M̃†[n]-→-     └-→- 1
```
Similarly, we can express M̃ using Qb
```
    M̃[n]
    = 1/√s[n-1]←-U†[n-1] R[n-1]--(M[n]-*-L[n]) V[n]←-1/√s[n]
    = 1/√s[n-1]←-U†[n-1] (R[n-1]--L[n-1]) Qb[n] V[n]←-1/√s[n]
    = -*-1/√s[n-1]←-U†[n-1] (R[n-1]-*-L[n-1]) Qb[n] V[n]←-1/√s[n]
    = -*-1/√s[n-1]←-(U†[n-1] U[n-1])←-s[n-1]←-V†[n-1] * Qb[n] V[n]←-1/√s[n]
    = -*-√s[n-1]←-V†[n-1] Qb[n] V[n]←-1/√s[n]
```
Here `-*-` is a twist to be applied when the codomain of `L[n-1]` is a dual space. 
Then `M̃[n]` (n = 2, ..., N) satisfies the (generalized) right-orthogonal condition
```
    -←-M̃[n]-←┐         1 -←-┐
        ↓    |              |       
        *    s[n]   =     s[n-1]   (s[N] = 1)
        ↓    |              |
    -→M̃†[n]-→┘         2 -→-┘
```
Here `-*-` is the twist on the physical axis. 

# Truncation of a bond on OBC-MPS

Suppose we want to truncate the bond between 
the n-th and the (n+1)-th sites such that the truncated ψ
```
    |ψ̃⟩  =  M[1]---...---M̃[n]---M̃[n+1]---...---M[N]
            ↓            ↓      ↓              ↓
```
maximizes the fidelity
```
            ⟨ψ|ψ̃⟩ ⟨ψ̃|ψ⟩
    F(ψ̃) = -------------
            ⟨ψ̃|ψ̃⟩ ⟨ψ|ψ⟩
```
This is simply done by using a truncated `Pa[n], Pb[n]`; then
```
            M[1]--...-M̃[n]--M̃[n+1]--...-M[N]
    ⟨ψ|ψ̃⟩ = |         |     |           |
            M†[1]-...-M†[n]-M†[n+1]-...-M†[N]

            ┌----M̃[n]--M̃[n+1]--┐
        = s[n-1] |     |     s[n+1]
            └----M†[n]-M†[n+1]-┘

            ┌--s̃[n]--┐
        =   |        |  =  s[n].s̃[n]
            └--s[n]--┘
```
Then the fidelity is just
```
    F(ψ̃) = (norm(s̃[n], 2) / norm(s[n], 2))^2
```
=#
#=
Perform QR decomposition through a PEPS tensor
```
             ╱           ╱
    --R0----M---  →  ---Q--*-R1--
          ╱ |         ╱ |
```
=#
function qr_through(
        R0::MPSBondTensor, M::GenericMPSTensor{S, 4}; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @tensor A[-1 -2 -3 -4; -5] := R0[-1; 1] * M[1 -2 -3 -4; -5]
    q, r = leftorth!(A; alg = QRpos())
    @assert isdual(domain(r, 1)) == isdual(codomain(r, 1))
    normalize && normalize!(r, Inf)
    return q, r
end
function qr_through(
        ::Nothing, M::GenericMPSTensor{S, 4}; normalize::Bool = true
    ) where {S <: ElementarySpace}
    q, r = leftorth(M; alg = QRpos())
    @assert isdual(domain(r, 1)) == isdual(codomain(r, 1))
    normalize && normalize!(r, Inf)
    return q, r
end

#=
Perform LQ decomposition through a tensor
```
             ╱           ╱
    --L0-*--Q---  ←  ---M--*-L1--
          ╱ |         ╱ |
```
=#
function lq_through(
        M::GenericMPSTensor{S, 4}, L1::MPSBondTensor; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @plansor A[-1 -2 -3 -4; -5] := M[-1 -2 -3 -4; 1] * L1[1; -5]
    l, q = rightorth!(permute(A, ((1,), (2, 3, 4, 5))); alg = LQpos())
    @assert isdual(domain(l, 1)) == isdual(codomain(l, 1))
    normalize && normalize!(l, Inf)
    return l, q
end
function lq_through(
        M::GenericMPSTensor{S, 4}, ::Nothing; normalize::Bool = true
    ) where {S <: ElementarySpace}
    l, q = rightorth(M, ((1,), (2, 3, 4, 5)); alg = LQpos())
    @assert isdual(domain(l, 1)) == isdual(codomain(l, 1))
    normalize && normalize!(l, Inf)
    return l, q
end

#=
Given a cluster `Ms`, find all `R`, `L` matrices on each internal bond
=#
function _get_allRLs(Ms::Vector{T}) where {T <: GenericMPSTensor{<:ElementarySpace, 4}}
    # M1 -- (R1,L1) -- M2 -- (R2,L2) -- M3
    N = length(Ms)
    # get the first R and the last L
    R_first = qr_through(nothing, Ms[1]; normalize = true)[2]
    L_last = lq_through(Ms[N], nothing; normalize = true)[1]
    Rs = Vector{typeof(R_first)}(undef, N - 1)
    Ls = Vector{typeof(L_last)}(undef, N - 1)
    Rs[1], Ls[end] = R_first, L_last
    # get remaining R, L matrices
    for n in 2:(N - 1)
        m = N - n + 1
        _, Rs[n] = qr_through(Rs[n - 1], Ms[n]; normalize = true)
        Ls[m - 1], _ = lq_through(Ms[m], Ls[m]; normalize = true)
    end
    return Rs, Ls
end

#=
Given the tensors `R`, `L` on a bond, construct 
the projectors `Pa`, `Pb` and the new bond weight `s`
such that the contraction of `Pa`, `s`, `Pb` is identity when `trunc = notrunc`,

The arrows between `Pa`, `s`, `Pb` are
```
    rev = false: - Pa --←-- Pb -
                    1 ← s ← 2

    rev = true:  - Pa --→-- Pb - 
                    2 → s → 1
```
=#
function _proj_from_RL(
        r::MPSBondTensor, l::MPSBondTensor;
        trunc::TruncationScheme = notrunc(), rev::Bool = false,
    )
    rl = r * l
    @assert isdual(domain(rl, 1)) == isdual(codomain(rl, 1))
    u, s, vh, ϵ = tsvd!(rl; trunc)
    sinv = PEPSKit.sdiag_pow(s, -1 / 2)
    Pa, Pb = l * vh' * sinv, sinv * u' * r
    if rev
        Pa, s, Pb = flip_svd(Pa, s, Pb)
    end
    return Pa, s, Pb, ϵ
end

#=
Given a cluster `Ms` and the pre-calculated `R`, `L` bond matrices,
find all projectors `Pa`, `Pb` and Schmidt weights `wts` on internal bonds.
=#
function _get_allprojs(
        Ms, Rs, Ls, trschemes::Vector{E}, revs::Vector{Bool}
    ) where {E <: TruncationScheme}
    N = length(Ms)
    @assert length(trschemes) == N - 1
    projs_errs = map(1:(N - 1)) do i
        trunc = if isa(trschemes[i], FixedSpaceTruncation)
            truncspace(space(Ms[i + 1], 1))
        else
            trschemes[i]
        end
        return _proj_from_RL(Rs[i], Ls[i]; trunc, rev = revs[i])
    end
    Pas = map(Base.Fix2(getindex, 1), projs_errs)
    wts = map(Base.Fix2(getindex, 2), projs_errs)
    Pbs = map(Base.Fix2(getindex, 3), projs_errs)
    # local truncation error on each bond
    ϵs = map(Base.Fix2(getindex, 4), projs_errs)
    return Pas, Pbs, wts, ϵs
end

#=
Find projectors to truncate internal bonds of the cluster `Ms`.
=#
function _cluster_truncate!(
        Ms::Vector{T}, trschemes::Vector{E}, revs::Vector{Bool}
    ) where {T <: GenericMPSTensor{<:ElementarySpace, 4}, E <: TruncationScheme}
    Rs, Ls = _get_allRLs(Ms)
    Pas, Pbs, wts, ϵs = _get_allprojs(Ms, Rs, Ls, trschemes, revs)
    # apply projectors
    # M1 -- (Pa1,wt1,Pb1) -- M2 -- (Pa2,wt2,Pb2) -- M3
    for (i, (Pa, Pb)) in enumerate(zip(Pas, Pbs))
        @plansor (Ms[i])[-1 -2 -3 -4; -5] := (Ms[i])[-1 -2 -3 -4; 1] * Pa[1; -5]
        @tensor (Ms[i + 1])[-1 -2 -3 -4; -5] := Pb[-1; 1] * (Ms[i + 1])[1 -2 -3 -4; -5]
    end
    return wts, ϵs, Pas, Pbs
end

#=
Apply the gate MPO `gs` on the cluster `Ms`.
When `gate_ax` is 1 or 2, the gate acts from the physical codomain or domain side.

e.g. Cluster in PEPS with `gate_ax = 1`:
```
         ╱       ╱       ╱
    --- M1 ---- M2 ---- M3 ---
      ╱ |     ╱ |     ╱ |
        ↓       ↓       ↓
        g1 -←-- g2 -←-- g3
        ↓       ↓       ↓
```

In the cluster, the axes of each tensor use the MPS order
```
    PEPS:           PEPO:
           3             3  4
          ╱              | ╱
    1 -- M -- 5     1 -- M -- 6
       ╱ |             ╱ |
      4  2            5  2
    M[1 2 3 4; 5]  M[1 2 3 4 5; 6]
```
=#
function _apply_gatempo!(
        Ms::Vector{T1}, gs::Vector{T2}; gate_ax::Int = 1
    ) where {T1 <: GenericMPSTensor{<:ElementarySpace, 4}, T2 <: AbstractTensorMap}
    @assert length(Ms) == length(gs)
    @assert gate_ax == 1
    @assert all(!isdual(space(g, 1)) for g in gs[2:end])
    # fusers to merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = map(Ms[2:end], gs[2:end]) do M, g
        V1, V2 = space(M, 1), space(g, 1)
        return isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
    end
    for (i, M) in enumerate(Ms[2:end])
        isdual(space(M, 1)) && twist!(Ms[i + 1], 1)
    end
    #= gate on codomain of PEPS
           -3                         -3                          -3
          ╱    ┌-┐          ┌-┐      ╱    ┌-┐           ┌-┐      ╱
    -1 --M--2--┤ |          | ├--2--M--4--┤ |           | ├--2--M-- -5
       ╱ |     | ├- -5  -1 -┤ |   ╱ |     | ├- -5   -1 -┤ |   ╱ |
     -4  1     | |          | | -4  1     | |           | | -4  1
         ├--3--┤ |          | ├--3--┼--5--┤ |           | ├--3--┤
         -2    └-┘          └-┘    -2     └-┘           └-┘    -2
    =#
    for (i, (g, M)) in enumerate(zip(gs, Ms))
        @assert !isdual(space(M, 2))
        if i == 1
            fr = fusers[i]
            @tensor (Ms[i])[-1 -2 -3 -4; -5] := M[-1 1 -3 -4; 2] * g[-2 1 3] * fr'[2 3; -5]
        elseif i == length(Ms)
            fl = fusers[i - 1]
            @tensor (Ms[i])[-1 -2 -3 -4; -5] := fl[-1; 2 3] * M[2 1 -3 -4; -5] * g[3 -2 1]
        else
            fl, fr = fusers[i - 1], fusers[i]
            @tensor (Ms[i])[-1 -2 -3 -4; -5] := fl[-1; 2 3] * M[2 1 -3 -4; 4] * g[3 -2 1 5] * fr'[4 5; -5]
        end
    end
    return Ms
end

function _apply_gatempo!(
        Ms::Vector{T1}, gs::Vector{T2}; gate_ax::Int = 1
    ) where {T1 <: GenericMPSTensor{<:ElementarySpace, 5}, T2 <: AbstractTensorMap}
    @assert length(Ms) == length(gs)
    @assert gate_ax == 1 || gate_ax == 2
    @assert all(!isdual(space(g, 1)) for g in gs[2:end])
    # fusers to merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = map(Ms[2:end], gs[2:end]) do M, g
        V1, V2 = space(M, 1), space(g, 1)
        return isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
    end
    for (i, M) in enumerate(Ms[2:end])
        isdual(space(M, 1)) && twist!(Ms[i + 1], 1)
    end
    #= gate on codomain of PEPO (gate_ax = 1)

        -3  -4                     -3  -4                      -3  -4
         | ╱   ┌-┐          ┌-┐     | ╱   ┌-┐           ┌-┐     | ╱
    -1 --M--2--┤ |          | ├--2--M--4--┤ |           | ├--2--M-- -6
       ╱ |     | ├- -6  -1 -┤ |   ╱ |     | ├- -6   -1 -┤ |   ╱ |
     -5  1     | |          | | -5  1     | |           | | -5  1
         ├--3--┤ |          | ├--3--┼--5--┤ |           | ├--3--┤
         -2    └-┘          └-┘    -2     └-┘           └-┘    -2

        gate on domain of PEPO (gate_ax = 2)

        -3     ┌-┐          ┌-┐    -3     ┌-┐           ┌-┐    -3
         ├--3--┤ |          | ├--3--┼--5--┤ |           | ├--3--┤
         1  -4 | ├- -6  -1 -┤ |     1  -4 | ├- -6   -1 -┤ |     1  -4
         | ╱   | |          | |     | ╱   | |           | |     | ╱
    -1 --M--2--┤ |          | ├--2--M--4--┤ |           | ├--2--M-- -6
       ╱ |     └-┘          └-┘   ╱ |     └-┘           └-┘   ╱ |
     -5 -2                      -5 -2                       -5 -2
    =#
    for (i, (g, M)) in enumerate(zip(gs, Ms))
        @assert !isdual(space(M, 2))
        if i == 1
            fr = fusers[i]
            if gate_ax == 1
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := M[-1 1 -3 -4 -5; 2] * g[-2 1 3] * fr'[2 3; -6]
            else
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := M[-1 -2 1 -4 -5; 2] * g[1 -3 3] * fr'[2 3; -6]
            end
        elseif i == length(Ms)
            fl = fusers[i - 1]
            if gate_ax == 1
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := fl[-1; 2 3] * M[2 1 -3 -4 -5; -6] * g[3 -2 1]
            else
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := fl[-1; 2 3] * M[2 -2 1 -4 -5; -6] * g[3 1 -3]
            end
        else
            fl, fr = fusers[i - 1], fusers[i]
            if gate_ax == 1
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := fl[-1; 2 3] * M[2 1 -3 -4 -5; 4] * g[3 -2 1 5] * fr'[4 5; -6]
            else
                @tensor (Ms[i])[-1 -2 -3 -4 -5; -6] := fl[-1; 2 3] * M[2 -2 1 -4 -5; 4] * g[3 1 -3 5] * fr'[4 5; -6]
            end
        end
    end
    return Ms
end

function _fuse_physicalspaces(O::GenericMPSTensor{S, 5}) where {S <: ElementarySpace}
    V1, V2 = codomain(O, 2), codomain(O, 3)
    F = isomorphism(Int, fuse(V1, V2), V1 ⊗ V2)
    @plansor O_fused[-1 -2 -4 -5; -6] := F[-2; 2 3] * O[-1 2 3 -4 -5; -6]
    return O_fused, F
end

function _unfuse_physicalspace(
        O::GenericMPSTensor{S, 4}, Vout::ElementarySpace, Vin::ElementarySpace = Vout'
    ) where {S <: ElementarySpace}
    F = isomorphism(Int, Vout ⊗ Vin, fuse(Vout ⊗ Vin))
    @plansor O_unfused[-1 -2 -3 -4 -5; -6] := F[-2 -3; 1] * O[-1 1 -4 -5; -6]
    return O_unfused, F
end

const openaxs_se = [(NORTH, SOUTH, WEST), (EAST, SOUTH), (NORTH, EAST, WEST)]
const invperms_se_peps = [((2,), (3, 5, 4, 1)), ((2,), (5, 3, 4, 1)), ((2,), (5, 3, 1, 4))]
const perms_se_peps = map(invperms_se_peps) do (p1, p2)
    p = invperm((p1..., p2...))
    return (p[1:(end - 1)], (p[end],))
end
const invperms_se_pepo = [((2, 3), (4, 6, 5, 1)), ((2, 3), (6, 4, 5, 1)), ((2, 3), (6, 4, 1, 5))]
const perms_se_pepo = map(invperms_se_pepo) do (p1, p2)
    p = invperm((p1..., p2...))
    return (p[1:(end - 1)], (p[end],))
end
#=
Obtain the 3-site cluster in the "southeast corner" of a square plaquette.
``` 
    r-1         M3
                |
                ↓
    r   M1 -←- M2
        c      c+1
```
=#
function get_3site_se(ψ::InfiniteState, env::SUWeight, row::Int, col::Int)
    Nr, Nc = size(ψ)
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    coords_se = [(row, col), (row, cp1), (rm1, cp1)]
    perms_se = isa(ψ, InfinitePEPS) ? perms_se_peps : perms_se_pepo
    Ms = map(zip(coords_se, perms_se, openaxs_se)) do (coord, perm, openaxs)
        M = absorb_weight(ψ.A[CartesianIndex(coord)], env, coord[1], coord[2], openaxs)
        return permute(M, perm)
    end
    return Ms
end

function _su3site_se!(
        ψ::InfiniteState, gs::Vector{T}, env::SUWeight,
        row::Int, col::Int, trschemes::Vector{E};
        gate_bothsides::Bool = true
    ) where {T <: AbstractTensorMap, E <: TruncationScheme}
    Nr, Nc = size(ψ)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    # southwest 3-site cluster and arrow direction within it
    Ms = get_3site_se(ψ, env, row, col)
    revs = [isdual(space(M, 1)) for M in Ms[2:end]]
    Vphys = [codomain(M, 2) for M in Ms]
    normalize!.(Ms, Inf)
    # sites in the cluster
    coords = ((row, col), (row, cp1), (rm1, cp1))
    # weights in the cluster
    wt_idxs = ((1, row, col), (2, row, cp1))
    # apply gate MPOs
    gate_axs = gate_bothsides ? (1:2) : (1:1)
    ϵs = nothing
    for gate_ax in gate_axs
        _apply_gatempo!(Ms, gs; gate_ax)
        if isa(ψ, InfinitePEPO)
            Ms = [first(_fuse_physicalspaces(M)) for M in Ms]
        end
        wts, ϵs, = _cluster_truncate!(Ms, trschemes, revs)
        if isa(ψ, InfinitePEPO)
            Ms = [first(_unfuse_physicalspace(M, Vphy)) for (M, Vphy) in zip(Ms, Vphys)]
        end
        for (wt, wt_idx) in zip(wts, wt_idxs)
            env[CartesianIndex(wt_idx)] = normalize(wt, Inf)
        end
    end
    invperms_se = isa(ψ, InfinitePEPS) ? invperms_se_peps : invperms_se_pepo
    for (M, coord, invperm, openaxs, Vphy) in zip(Ms, coords, invperms_se, openaxs_se, Vphys)
        # restore original axes order
        M = permute(M, invperm)
        # remove weights on open axes of the cluster
        M = absorb_weight(M, env, coord[1], coord[2], openaxs; inv = true)
        ψ.A[CartesianIndex(coord)] = normalize(M, Inf)
    end
    return ϵs
end

function su_iter(
        ψ::InfiniteState, gatempos::Vector{G}, alg::SimpleUpdate, env::SUWeight
    ) where {G <: AbstractMatrix}
    if ψ isa InfinitePEPO
        @assert size(ψ, 3) == 1
    end
    Nr, Nc = size(ψ)[1:2]
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError(
            "iPEPS unit cell size for simple update should be no smaller than (2, 2)."
        ),
    )
    ψ2, env2 = deepcopy(ψ), deepcopy(env)
    trscheme = alg.trscheme
    for i in 1:4
        Nr, Nc = size(ψ2)[1:2]
        for r in 1:Nr, c in 1:Nc
            gs = gatempos[i][r, c]
            trschemes = [
                truncation_scheme(trscheme, 1, r, c)
                truncation_scheme(trscheme, 2, r, _next(c, Nc))
            ]
            _su3site_se!(ψ2, gs, env2, r, c, trschemes; alg.gate_bothsides)
        end
        ψ2, env2 = rotl90(ψ2), rotl90(env2)
        trscheme = rotl90(trscheme)
    end
    wtdiff = compare_weights(env2, env)
    return ψ2, env2, (; wtdiff)
end

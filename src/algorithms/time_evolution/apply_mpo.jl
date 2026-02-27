#= 
# Mixed canonical form of an open boundary MPS
```
    |ψ⟩ =  M[1]-←-...-←-M[N]
            ↓            ↓
```
For convenience, assume all virtual arrows are ←.

We perform QR and LQ decompositions: starting from
```
    M[1]-←-  =  Qa[1]-←-R[1]-←-
    ↓           ↓

    -←-M[N]  =  -←-L[N-1]-←-Qb[N]
        ↓                   ↓
```
we successively calculate
```
    -←-R[n-1]-←-M[n]-←-    =  -←-Qa[n]-←-R[n]--←-- (n = 2, ..., N - 1)
                ↓                ↓

    -←-M[n+1]-←-L[n+1]-←-  =  -←-L[n]-←-Qb[n+1]-←- (n = N - 2, ..., 1)
        ↓                               ↓
```

For each bond (n = 1, ..., N - 1), we perform SVD
```
    R[n] L[n] = U[n] s[n] V†[n] (n = 1, ..., N - 1)
```
Then we define the projectors together with the Schmidt weight
```
    -←-Pa[n]-←- = L[n] V[n]-←-(1/√s[n])-←-
    -←-Pb[n]-←- = -←-(1/√s[n])-←-U†[n] R[n]
```
The product `Pa Pb` is the identity operator:
```
    Pa[n]-←-Pb[n] = L[n] (R[n] L[n])⁻¹ R[n] = 1
```

The canonical form is then defined by
```
    -←-M̃[n]-←- = -←-Pb[n-1]-←-M[n]-←-Pa[n]-←-
        ↓                      ↓
```

Note that
```
    M̃[n]
    = 1/√s[n-1] U†[n-1] (R[n-1] M[n]) L[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] (R[n] L[n]) V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] U[n] s[n] (V†[n] V[n]) 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] U[n] √s[n]
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
    = 1/√s[n-1] U†[n-1] R[n-1] (M[n] L[n]) V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] (R[n-1] L[n-1]) Qb[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] (R[n-1] L[n-1]) Qb[n] V[n] 1/√s[n]
    = 1/√s[n-1] (U†[n-1] U[n-1]) s[n-1] V†[n-1] Qb[n] V[n] 1/√s[n]
    = √s[n-1] V†[n-1] Qb[n] V[n] 1/√s[n]
```
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
the n-th and the (n+1)-th sites such that the truncated state
```
    |ψ̃⟩  =  M[1]-←-...-←-M̃[n]-←-M̃[n+1]-←-...-←-M[N]
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
"""
Perform QR decomposition through a PEPS tensor
```
             ╱            ╱
    -←-R0-←-M-←-  =>  ---Q-←-R1-←-
          ╱ |          ╱ |
```
"""
function qr_through(
        R0::MPSBondTensor, M::GenericMPSTensor{S, 4}; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @assert !isdual(codomain(R0, 1))
    @assert !isdual(domain(M, 1)) && !isdual(codomain(M, 1))
    @tensor A[-1 -2 -3 -4; -5] := R0[-1; 1] * M[1 -2 -3 -4; -5]
    _, r = left_orth!(A; positive = true)
    normalize && normalize!(r, Inf)
    return r
end
# for `M` at the left end of the MPS
function qr_through(
        ::Nothing, M::GenericMPSTensor{S, 4}; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @assert !isdual(domain(M, 1))
    _, r = left_orth(M; positive = true)
    normalize && normalize!(r, Inf)
    return r
end

"""
Perform LQ decomposition through a tensor
```
             ╱            ╱
    -←-L0-←-Q-←-  <=  -←-M-←-L1-←-
          ╱ |          ╱ |
```
"""
function lq_through(
        M::GenericMPSTensor{S, 4}, L1::MPSBondTensor; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @assert !isdual(domain(L1, 1))
    @assert !isdual(codomain(M, 1)) && !isdual(domain(M, 1))
    @tensor A[-1; -2 -3 -4 -5] := M[-1 -2 -3 -4; 1] * L1[1; -5]
    l, _ = right_orth!(A; positive = true)
    normalize && normalize!(l, Inf)
    return l
end
# for `M` at the right end of the MPS
function lq_through(
        M::GenericMPSTensor{S, 4}, ::Nothing; normalize::Bool = true
    ) where {S <: ElementarySpace}
    @assert !isdual(codomain(M, 1))
    A = permute(M, ((1,), (2, 3, 4, 5)))
    l, _ = right_orth!(A; positive = true)
    normalize && normalize!(l, Inf)
    return l
end

"""
Given a cluster `Ms`, find all `R`, `L` matrices on each internal bond
"""
function _get_allRLs(Ms::Vector{T}) where {T <: GenericMPSTensor{<:ElementarySpace, 4}}
    # M1 -- (R1,L1) -- M2 -- (R2,L2) -- M3
    N = length(Ms)
    # get the first R and the last L
    R_first = qr_through(nothing, Ms[1]; normalize = true)
    L_last = lq_through(Ms[N], nothing; normalize = true)
    Rs = Vector{typeof(R_first)}(undef, N - 1)
    Ls = Vector{typeof(L_last)}(undef, N - 1)
    Rs[1], Ls[end] = R_first, L_last
    # get remaining R, L matrices
    for n in 2:(N - 1)
        m = N - n + 1
        Rs[n] = qr_through(Rs[n - 1], Ms[n]; normalize = true)
        Ls[m - 1] = lq_through(Ms[m], Ls[m]; normalize = true)
    end
    return Rs, Ls
end

"""
Given the tensors `R`, `L` on a bond, construct 
the projectors `Pa`, `Pb` and the new bond weight `s`
such that the contraction of `Pa`, `s`, `Pb` is identity when `trunc = notrunc`,

The arrows between `Pa`, `s`, `Pb` are
```
    - Pa --←-- Pb -
       1 ← s ← 2
```
"""
function _proj_from_RL(
        r::MPSBondTensor, l::MPSBondTensor;
        trunc::TruncationStrategy = notrunc()
    )
    @assert isdual(domain(r, 1)) == isdual(codomain(r, 1)) == false
    @assert isdual(domain(l, 1)) == isdual(codomain(l, 1)) == false
    rl = r * l
    u, s, vh, ϵ = svd_trunc!(rl; trunc)
    sinv = sdiag_pow(s, -1 / 2)
    Pa, Pb = l * vh' * sinv, sinv * u' * r
    return Pa, s, Pb, ϵ
end

"""
Given a cluster `Ms` and the pre-calculated `R`, `L` bond matrices,
find all projectors `Pa`, `Pb` and Schmidt weights `wts` on internal bonds.
"""
function _get_allprojs(Ms, Rs, Ls, truncs::Vector{E}) where {E <: TruncationStrategy}
    N = length(Ms)
    @assert length(truncs) == N - 1
    projs_errs = map(1:(N - 1)) do i
        trunc = if isa(truncs[i], FixedSpaceTruncation)
            tspace = space(Ms[i + 1], 1)
            isdual(tspace) ? truncspace(flip(tspace)) : truncspace(tspace)
        else
            truncs[i]
        end
        return _proj_from_RL(Rs[i], Ls[i]; trunc)
    end
    Pas = map(Base.Fix2(getindex, 1), projs_errs)
    wts = map(Base.Fix2(getindex, 2), projs_errs)
    Pbs = map(Base.Fix2(getindex, 3), projs_errs)
    # local truncation error on each bond
    ϵs = map(Base.Fix2(getindex, 4), projs_errs)
    return Pas, Pbs, wts, ϵs
end

"""
Flip the virtual arrows in the MPS `Ms`
"""
function _flip_virtuals!(
        Ms::Vector{T}, flips::Vector{Bool}; inv::Bool = false
    ) where {T <: GenericMPSTensor}
    @assert length(flips) == length(Ms) - 1
    for (n, flip) in enumerate(flips)
        !flip && continue
        M1, M2 = Ms[n], Ms[n + 1]
        Ms[n] = TensorKit.flip(M1, numind(M1); inv)
        Ms[n + 1] = TensorKit.flip(M2, 1; inv)
    end
    return Ms
end

"""
Find projectors to truncate internal bonds of the cluster `Ms`.
"""
function _cluster_truncate!(
        Ms::Vector{T}, truncs::Vector{E}
    ) where {T <: GenericMPSTensor{<:ElementarySpace, 4}, E <: TruncationStrategy}
    Rs, Ls = _get_allRLs(Ms)
    Pas, Pbs, wts, ϵs = _get_allprojs(Ms, Rs, Ls, truncs)
    # apply projectors
    # M1 -- (Pa1,wt1,Pb1) -- M2 -- (Pa2,wt2,Pb2) -- M3
    for (i, (Pa, Pb)) in enumerate(zip(Pas, Pbs))
        @tensor (Ms[i])[-1 -2 -3 -4; -5] := (Ms[i])[-1 -2 -3 -4; 1] * Pa[1; -5]
        @tensor (Ms[i + 1])[-1 -2 -3 -4; -5] := Pb[-1; 1] * (Ms[i + 1])[1 -2 -3 -4; -5]
    end
    return wts, ϵs, Pas, Pbs
end

"""
Apply the gate MPO `gs` on the cluster `Ms`.
When `gate_ax` is 1 or 2, the gate acts from the physical codomain or domain side.

e.g. Cluster in PEPS with `gate_ax = 1`:
```
         ╱       ╱       ╱
    --- M1 -←-- M2 -←-- M3 ---
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
"""
function _apply_gatempo!(
        Ms::Vector{T1}, gs::Vector{T2}; gate_ax::Int = 1
    ) where {T1 <: GenericMPSTensor{<:ElementarySpace, 4}, T2 <: AbstractTensorMap}
    @assert length(Ms) == length(gs)
    @assert gate_ax == 1
    @assert all(!isdual(space(g, 1)) for g in gs[2:end])
    @assert all(!isdual(space(M, 1)) for M in Ms[2:end])
    # fusers to merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = map(Ms[2:end], gs[2:end]) do M, g
        V1, V2 = space(M, 1), space(g, 1)
        return isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
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
    @assert all(!isdual(space(M, 1)) for M in Ms[2:end])
    # fusers to merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = map(Ms[2:end], gs[2:end]) do M, g
        V1, V2 = space(M, 1), space(g, 1)
        return isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
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

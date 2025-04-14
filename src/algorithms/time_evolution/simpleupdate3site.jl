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
the n-th and the (n+1)-th sites such that the truncated state
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
"""
Perform QR decomposition through a PEPS tensor
```
             ╱           ╱
    --R0----M---  →  ---Q--*-R1--
          ╱ |         ╱ |
```
"""
function qr_through(
    R0::AbstractTensorMap{T,S,1,1}, M::AbstractTensorMap{T,S,1,4}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    @tensor A[-1; -2 -3 -4 -5] := R0[-1; 1] * M[1; -2 -3 -4 -5]
    q, r = leftorth(A, ((1, 2, 3, 4), (5,)); alg=QRpos())
    @assert isdual(domain(r, 1)) == isdual(codomain(r, 1))
    normalize && (r /= norm(r, Inf))
    return q, r
end
function qr_through(
    ::Nothing, M::AbstractTensorMap{T,S,1,4}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    q, r = leftorth(M, ((1, 2, 3, 4), (5,)); alg=QRpos())
    @assert isdual(domain(r, 1)) == isdual(codomain(r, 1))
    normalize && (r /= norm(r, Inf))
    return q, r
end

"""
Perform LQ decomposition through a tensor
```
             ╱           ╱
    --L0-*--Q---  ←  ---M--*-L1--
          ╱ |         ╱ |
```
"""
function lq_through(
    M::AbstractTensorMap{T,S,1,4}, L1::AbstractTensorMap{T,S,1,1}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    @plansor A[-1; -2 -3 -4 -5] := M[-1; -2 -3 -4 1] * L1[1; -5]
    l, q = rightorth!(A; alg=LQpos())
    @assert isdual(domain(l, 1)) == isdual(codomain(l, 1))
    normalize && (l /= norm(l, Inf))
    return l, q
end
function lq_through(
    M::AbstractTensorMap{T,S,1,4}, ::Nothing; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    l, q = rightorth(M; alg=LQpos())
    @assert isdual(domain(l, 1)) == isdual(codomain(l, 1))
    normalize && (l /= norm(l, Inf))
    return l, q
end

"""
Given a cluster `Ms`, find all `R`, `L` matrices on each internal bond
"""
function _get_allRLs(Ms::Vector{T}) where {T<:PEPSTensor}
    # M1 -- (R1,L1) -- M2 -- (R2,L2) -- M3
    N = length(Ms)
    Rs = Vector{AbstractTensorMap}(undef, N - 1)
    Ls = Vector{AbstractTensorMap}(undef, N - 1)
    for n in 1:(N - 1)
        m = N - n + 1
        _, Rs[n] = qr_through((n == 1) ? nothing : Rs[n - 1], Ms[n]; normalize=true)
        Ls[m - 1], _ = lq_through(Ms[m], (m == N) ? nothing : Ls[m]; normalize=true)
    end
    return Rs, Ls
end

"""
Given the tensors `R`, `L` on a bond, construct 
the projectors `Pa`, `Pb` and the new bond weight `s`
such that the contraction of `Pa`, `s`, `Pb` is identity when `trunc = notrunc`,

The arrows between `Pa`, `s`, `Pb` are
```
    rev = false: - Pa ← s ← Pb -
                    1 ← s ← 2

    rev = true:  - Pa → s → Pb - 
                    2 → s → 1
```
"""
function _proj_from_RL(
    r::AbstractTensorMap{T,S,1,1},
    l::AbstractTensorMap{T,S,1,1};
    trunc::TensorKit.TruncationScheme=notrunc(),
    rev::Bool=false,
) where {T<:Number,S<:ElementarySpace}
    rl = r * l
    @assert isdual(domain(rl, 1)) == isdual(codomain(rl, 1))
    u, s, vh, ϵ = tsvd!(rl; trunc)
    sinv = PEPSKit.sdiag_pow(s, -1)
    Pa, Pb = l * vh' * sinv, sinv * u' * r
    if rev
        Pa, Pb = flip(Pa, 2), flip(Pb, 1)
        s = permute(DiagonalTensorMap(flip(s, (1, 2))), ((2,), (1,)))
        @assert all(s.data .>= 0.0)
    end
    return Pa, s, Pb, ϵ
end

"""
Given a cluster `Ms` and the pre-calculated `R`, `L` bond matrices,
find all projectors `Pa`, `Pb` and Schmidt weights `wts` on internal bonds.
"""
function _get_allprojs(Ms, Rs, Ls, trunc::TensorKit.TruncationScheme, revs::Vector{Bool})
    N = length(Ms)
    projs_errs = map(1:(N - 1)) do i
        trunc2 = if isa(trunc, FixedSpaceTruncation)
            truncspace(space(Ms[i + 1], 1))
        else
            trunc
        end
        return _proj_from_RL(Rs[i], Ls[i]; trunc=trunc2, rev=revs[i])
    end
    Pas = map(Base.Fix2(getindex, 1), projs_errs)
    wts = map(Base.Fix2(getindex, 2), projs_errs)
    Pbs = map(Base.Fix2(getindex, 3), projs_errs)
    # local truncation error on each bond
    ϵs = map(Base.Fix2(getindex, 4), projs_errs)
    return Pas, Pbs, wts, ϵs
end

"""
Find projectors to truncate internal bonds of the cluster `Ms`
"""
function _cluster_truncate!(
    Ms::Vector{T}, trunc::TensorKit.TruncationScheme, revs::Vector{Bool}
) where {T<:PEPSTensor}
    Rs, Ls = _get_allRLs(Ms)
    Pas, Pbs, wts, ϵs = _get_allprojs(Ms, Rs, Ls, trunc, revs)
    # apply projectors
    # M1 -- (Pa1,wt1,Pb1) -- M2 -- (Pa2,wt2,Pb2) -- M3
    for (i, (Pa, Pb)) in enumerate(zip(Pas, Pbs))
        @plansor (Ms[i])[-1; -2 -3 -4 -5] := (Ms[i])[-1; -2 -3 -4 1] * Pa[1; -5]
        @tensor (Ms[i + 1])[-1; -2 -3 -4 -5] := Pb[-1; 1] * (Ms[i + 1])[1; -2 -3 -4 -5]
    end
    return wts, ϵs, Pas, Pbs
end

function _apply_gatempo!(
    Ms::Vector{T1}, gs::Vector{T2}
) where {T1<:PEPSTensor,T2<:AbstractTensorMap}
    @assert length(Ms) == length(gs)
    @assert all(!isdual(space(g, 1)) for g in gs[2:end])
    # fusers to merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = collect(
        begin
            V1, V2 = space(M, 1), space(g, 1)
            isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
        end for (M, g) in zip(Ms[2:end], gs[2:end])
    )
    for (i, M) in enumerate(Ms[2:end])
        isdual(space(M, 1)) && twist!(Ms[i + 1], 1)
    end
    for (i, (g, M)) in enumerate(zip(gs, Ms))
        @assert !isdual(space(M, 2))
        if i == 1
            fr = fusers[i]
            @tensor (Ms[i])[-1; -2 -3 -4 -5] := M[-1; 1 -3 -4 2] * g[-2 1 3] * fr'[2 3; -5]
        elseif i == length(Ms)
            fl = fusers[i - 1]
            @tensor (Ms[i])[-1; -2 -3 -4 -5] := fl[-1; 2 3] * M[2; 1 -3 -4 -5] * g[3 -2 1]
        else
            fl, fr = fusers[i - 1], fusers[i]
            @tensor (Ms[i])[-1; -2 -3 -4 -5] :=
                fl[-1; 2 3] * M[2; 1 -3 -4 4] * g[3 -2 1 5] * fr'[4 5; -5]
        end
    end
    return Ms
end

"""
Apply the gate MPO on the cluster and truncate the bond
```
         ╱       ╱       ╱
    --- M1 ---- M2 ---- M3 ---
      ╱ |     ╱ |     ╱ |
        ↓       ↓       ↓
        g1 -←-- g2 -←-- g3
        ↓       ↓       ↓
```
In the cluster, the axes of each PEPSTensor are reordered as
```
           3
          ╱
    1 -- M -- 5     M[1; 2 3 4 5]
       ╱ |
     4   2
```
"""
function apply_gatempo!(
    Ms::Vector{T1}, gs::Vector{T2}; trunc::TensorKit.TruncationScheme
) where {T1<:PEPSTensor,T2<:AbstractTensorMap}
    @assert length(Ms) == length(gs)
    revs = [isdual(space(M, 1)) for M in Ms[2:end]]
    _apply_gatempo!(Ms, gs)
    wts, ϵs, = _cluster_truncate!(Ms, trunc, revs)
    return wts, ϵs
end

const openaxs_sw = [(NORTH, EAST, WEST), (SOUTH, WEST), (NORTH, EAST, SOUTH)]
const sqrtwts_sw = [ntuple(dir -> !(dir in idxs), 4) for idxs in openaxs_sw]
const invperms_sw = [((2,), (1, 3, 5, 4)), ((2,), (1, 5, 3, 4)), ((2,), (3, 5, 4, 1))]
const perms_sw = [
    begin
        p = invperm((p1..., p2...))
        ((p[1],), p[2:end])
    end for (p1, p2) in invperms_sw
]
"""
Obtain the following 3-site cluster with `M2` at `[r, c]`
``` 
    r-1 M1
        |
        ↓
    r   M2 -←- M3
        c      c+1
```
"""
function get_3site_sw(peps::InfiniteWeightPEPS, row::Int, col::Int)
    Nr, Nc = size(peps)
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    coords_sw = [(rm1, col), (row, col), (row, cp1)]
    cluster = collect(
        permute(
            _absorb_weights(
                peps.vertices[CartesianIndex(coord)],
                peps.weights,
                coord[1],
                coord[2],
                Tuple(1:4),
                sqrtwts,
                false,
            ),
            perm,
        ) for (i, (coord, sqrtwts, perm)) in enumerate(zip(coords_sw, sqrtwts_sw, perms_sw))
    )
    return cluster
end

const openaxs_se = [(NORTH, SOUTH, WEST), (EAST, SOUTH), (NORTH, EAST, WEST)]
const sqrtwts_se = [ntuple(dir -> !(dir in idxs), 4) for idxs in openaxs_se]
const invperms_se = [((2,), (3, 5, 4, 1)), ((2,), (5, 3, 4, 1)), ((2,), (5, 3, 1, 4))]
const perms_se = [
    begin
        p = invperm((p1..., p2...))
        ((p[1],), p[2:end])
    end for (p1, p2) in invperms_se
]
"""
Obtain the following 3-site cluster
``` 
    r-1         M3
                |
                ↓
    r   M1 -←- M2
        c      c+1
```
"""
function get_3site_se(peps::InfiniteWeightPEPS, row::Int, col::Int)
    Nr, Nc = size(peps)
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    coords_se = [(row, col), (row, cp1), (rm1, cp1)]
    cluster = collect(
        permute(
            _absorb_weights(
                peps.vertices[CartesianIndex(coord)],
                peps.weights,
                coord[1],
                coord[2],
                Tuple(1:4),
                sqrtwts,
                false,
            ),
            perm,
        ) for (i, (coord, sqrtwts, perm)) in enumerate(zip(coords_se, sqrtwts_se, perms_se))
    )
    return cluster
end

function _su3site_cluster!(
    row::Int,
    col::Int,
    gs::Vector{T},
    peps::InfiniteWeightPEPS,
    alg::SimpleUpdate,
    cluster::Symbol,
) where {T<:AbstractTensorMap}
    Nr, Nc = size(peps)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert cluster in (:sw, :se)
    _peps_dualcheck(peps)
    rm1, cp1 = _prev(row, Nr), _next(col, Nc)
    Ms, coords, wt_idxs, invperms, openaxs = if cluster == :sw
        (
            get_3site_sw(peps, row, col),
            ((rm1, col), (row, col), (row, cp1)),
            ((2, row, col), (1, row, col)),
            invperms_sw,
            openaxs_sw,
        )
    else
        (
            get_3site_se(peps, row, col),
            ((row, col), (row, cp1), (rm1, cp1)),
            ((1, row, col), (2, row, cp1)),
            invperms_se,
            openaxs_se,
        )
    end
    wts, ϵ = apply_gatempo!(Ms, gs; trunc=alg.trscheme)
    for (wt, wt_idx) in zip(wts, wt_idxs)
        peps.weights[CartesianIndex(wt_idx)] = wt / norm(wt, Inf)
    end
    for (M, coord, invperm, axs) in zip(Ms, coords, invperms, openaxs)
        # restore original axes order
        M = permute(M, invperm)
        # remove weights on open axes of the cluster
        _allfalse = ntuple(Returns(false), length(axs))
        M = _absorb_weights(M, peps.weights, coord[1], coord[2], axs, _allfalse, true)
        peps.vertices[CartesianIndex(coord)] = M * (100.0 / norm(M, Inf))
    end
    return nothing
end

"""
    su3site_iter(gatempos::NamedTuple, peps::InfiniteWeightPEPS, alg::SimpleUpdate)

One round of 3-site simple update for Hamiltonian with 2nd neighbor terms. 
"""
function su3site_iter(gatempos::NamedTuple, peps::InfiniteWeightPEPS, alg::SimpleUpdate)
    peps2 = deepcopy(peps)
    for cluster in (:sw, :se), site in CartesianIndices(peps2.vertices)
        r, c = Tuple(site)
        gs = gatempos[cluster][r, c]
        _su3site_cluster!(r, c, gs, peps2, alg, cluster)
    end
    return peps2
end

function _peps_dualcheck(peps::InfiniteWeightPEPS)
    Nr, Nc = size(peps)
    for r in Nr, c in Nc
        @assert [isdual(space(peps.vertices[r, c], ax)) for ax in 1:5] == [0, 1, 1, 0, 0]
        @assert [isdual(space(peps.weights[1, r, c], ax)) for ax in 1:2] == [0, 1]
        @assert [isdual(space(peps.weights[2, r, c], ax)) for ax in 1:2] == [0, 1]
    end
    return nothing
end

"""
    simpleupdate3site(peps::InfiniteWeightPEPS, ham::LocalOperator, alg::SimpleUpdate; check_interval::Int=500)

Perform simple update for next-nearest neighbor Hamiltonian `ham`, 
where the evolution information is printed every `check_interval` steps. 
"""
function simpleupdate3site(
    peps::InfiniteWeightPEPS, ham::LocalOperator, alg::SimpleUpdate; check_interval::Int=500
)
    time_start = time()
    Nr, Nc = size(peps)
    (Nr >= 2 && Nc >= 2) || throw(
        ArgumentError(
            "iPEPS unit cell size for simple update should be no smaller than (2, 2)."
        ),
    )
    gate = get_expham(alg.dt, ham)
    # convert gates to 3-site MPOs
    gatempos = _get_gatempos(gate)
    wtdiff = 1.0
    wts0 = deepcopy(peps.weights)
    for count in 1:(alg.maxiter)
        time0 = time()
        peps = su3site_iter(gatempos, peps, alg)
        wtdiff = compare_weights(peps.weights, wts0)
        converge = (wtdiff < alg.tol)
        cancel = (count == alg.maxiter)
        wts0 = deepcopy(peps.weights)
        time1 = time()
        if ((count == 1) || (count % check_interval == 0) || converge || cancel)
            @info "Space of x-weight at [1, 1] = $(space(peps.weights[1, 1, 1], 1))"
            label = (converge ? "conv" : (cancel ? "cancel" : "iter"))
            message = @sprintf(
                "SU %s %-7d:  dt = %.0e,  weight diff = %.3e,  time = %.3f sec\n",
                label,
                count,
                alg.dt,
                wtdiff,
                time1 - ((converge || cancel) ? time_start : time0)
            )
            cancel ? (@warn message) : (@info message)
        end
        converge && break
    end
    return peps, wtdiff
end

#= 
# Mixed canonical form of an open boundary MPS

```
            |            |
    |ψ⟩  =  M[1] - ... - M[N]
```
We perform QR and LQ decompositions 
(for `n = 1, ..., N`, with R[0] = 1, L[N] = 1)
```
                |               |
    - R[n-1] -- M[n] --  =  -- Qa[n] -- R[n] -

        |                               |
    -- M[n] --- L[n] --  =  - L[n-1] -- Qb[n] --
```
We further perform SVD 
```
    R[n] L[n] = U[n] s[n] V†[n]
```
Then we insert identity
```
    Pa[n] = L[n] V[n] * 1/√s[n]
    Pb[n] = 1/√s[n] * U†[n] R[n]
```
Evidently
```
    Pa[n] Pb[n] = L[n] (R[n] L[n])⁻¹ R[n] = 1
```
The canonical form is then defined by
```
    M̃[n] = Pb[n-1] M[n] Pa[n]
```
Note that
```
    M̃[n]
    = 1/√s[n-1] U†[n-1] R[n-1] M[n] L[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] R[n] L[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] U[n] s[n] V†[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] Qa[n] U[n] √s[n]
```
Then `M̃[n]` satisfies the (generalized) left-orthogonal condition
```
        ┌--M̃†[n]--     ┌---
    s[n-1]   |      =  s[n]     (s[0] = 1)
        └---M̃[n]--     └---
```
Similarly, we can express M̃ using Qb
```
    M̃[n]
    = 1/√s[n-1] U†[n-1] R[n-1] L[n-1] Qb[n] V[n] 1/√s[n]
    = 1/√s[n-1] U†[n-1] U[n-1] s[n-1] V†[n-1] Qb[n] V[n] 1/√s[n]
    = √s[n-1] V†[n-1] Qb[n] V[n] 1/√s[n]
```
Then `M̃[n]` satisfies the (generalized) right-orthogonal condition
```
    --M̃†[n]--┐      --┐
        |   s[n] =   s[n-1]     (s[N] = 1)
    ---M̃[n]--┘      --┘
```

# Truncation of a bond on OBC-MPS

Suppose we want to truncate the bond between 
the n-th and the (n+1)-th sites such that the truncated state
```
            |            |      |              |
    |ψ̃⟩  =  M[1] - ... - M̃[n] - M̃[n+1] - ... - M[N]
```
maximizes the fidelity
```
            ⟨ψ|ψ̃⟩ ⟨ψ̃|ψ⟩
    F(ψ̃) = -------------
            ⟨ψ̃|ψ̃⟩ ⟨ψ|ψ⟩
```
This is simply done by using a truncated `Pa[n], Pb[n]`; then
```
            ┌-M†[1]-...-M†[n]-M†[n+1]-...-M†[N]-┐
    ⟨ψ|ψ̃⟩ = | |         |     |           |     |
            └-M[1]--...-M̃[n]--M̃[n+1]--...-M[N]--┘

            ┌----M†[n]-M†[n+1]-┐
        = s[n-1] |     |     s[n+1]
            └----M̃[n]--M̃[n+1]--┘

            ┌--s[n]--┐
        =   |        |  =  s[n].s̃[n]
            └--s̃[n]--┘
```
Then the fidelity is just
```
    F(ψ̃) = (norm(s̃[n], 2) / norm(s[n], 2))^2
```
=#

"""
Perform QR decomposition through a PEPS tensor
```
            | ╱         | ╱
    - R0 -- M --  →  -- Q -- R1 -
           ╱           ╱
```
"""
function qr_through(
    R0::AbstractTensorMap{T,S,1,1}, M::AbstractTensorMap{T,S,1,4}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    @plansor A[-1; -2 -3 -4 -5] := R0[-1; 1] * M[1; -2 -3 -4 -5]
    q, r = leftorth(A, ((1, 2, 3, 4), (5,)); alg=QRpos())
    normalize && (r /= norm(r, Inf))
    return q, r
end
function qr_through(
    ::Nothing, M::AbstractTensorMap{T,S,1,4}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    q, r = leftorth(M, ((1, 2, 3, 4), (5,)); alg=QRpos())
    normalize && (r /= norm(r, Inf))
    return q, r
end

"""
Perform LQ decomposition through a tensor
```
            | ╱         | ╱
    - L0 -- Q --  ←  -- M -- L1 -
           ╱           ╱
```
"""
function lq_through(
    M::AbstractTensorMap{T,S,1,4}, L1::AbstractTensorMap{T,S,1,1}; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    @plansor A[-1; -2 -3 -4 -5] := M[-1; -2 -3 -4 1] * L1[1; -5]
    l, q = rightorth(A; alg=LQpos())
    normalize && (l /= norm(l, Inf))
    return l, q
end
function lq_through(
    M::AbstractTensorMap{T,S,1,4}, ::Nothing; normalize::Bool=true
) where {T<:Number,S<:ElementarySpace}
    l, q = rightorth(M; alg=LQpos())
    normalize && (l /= norm(l, Inf))
    return l, q
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
    u, s, vh, ϵ = tsvd!(rl; trunc)
    sinv = PEPSKit.sdiag_pow(s, -1)
    Pa, Pb = l * vh' * sinv, sinv * u' * r
    if rev
        V = space(Pb, 1)
        f = isomorphism(flip(V), V)
        f2 = twist(f, 1)
        # Pa ← f†) → (f ← s ← f†) → (f ← Pb
        @tensor begin
            Pa[-1; -2] := Pa[-1; 1] * f'[1; -2]
            Pb[-1; -2] := f2[-1; 1] * Pb[1; -2]
            s[-1; -2] := f2[-1; 1] * s[1; 2] * f'[2; -2]
        end
        s = DiagonalTensorMap(permute(s, ((2,), (1,))))
        @assert all(s.data .>= 0.0)
    end
    return Pa, s, Pb, ϵ
end

function _apply_gatempo!(
    Ms::Vector{T1}, gs::Vector{T2}
) where {T1<:AbstractTensorMap,T2<:AbstractTensorMap}
    @assert length(Ms) == length(gs)
    for (i, (g, M)) in enumerate(zip(gs, Ms))
        @assert !isdual(space(M, 2))
        if i == 1
            @tensor (Ms[i])[:] := g[-2 1 -5] * M[-1 1 -3 -4 -6]
        elseif i == length(Ms)
            @assert !isdual(space(g, 1))
            @tensor (Ms[i])[:] := g[-1 -3 1] * M[-2 1 -4 -5 -6]
        else
            @assert !isdual(space(g, 1))
            @tensor (Ms[i])[:] := g[-1 -3 1 -6] * M[-2 1 -4 -5 -7]
        end
    end
    for (i, M) in enumerate(Ms[2:end])
        isdual(space(M, 2)) && twist!(Ms[i + 1], 2)
    end
    # merge axes on bonds in the gate-cluster product
    # M1 == f1† -- f1 == M2 == f2† -- f2 == M3
    fusers = collect(begin
        V1, V2 = space(M, 1), space(M, 2)
        isomorphism(fuse(V1, V2) ← V1 ⊗ V2)
    end for M in Ms[2:end])
    for (i, M) in enumerate(Ms)
        if i == 1
            @tensor (Ms[i])[-1; -2 -3 -4 -5] := M[-1 -2 -3 -4 1 2] * (fusers[i])'[1 2; -5]
        elseif i == length(Ms)
            @tensor (Ms[i])[-1; -2 -3 -4 -5] :=
                (fusers[i - 1])[-1; 1 2] * M[1 2 -2 -3 -4 -5]
        else
            @tensor (Ms[i])[-1; -2 -3 -4 -5] :=
                (fusers[i - 1])[-1; 1 2] * M[1 2 -2 -3 -4 3 4] * (fusers[i])'[3 4; -5]
        end
    end
    return Ms
end

"""
Find projectors to truncate internal bonds of the cluster `Ms`
"""
function _cluster_truncate!(
    Ms::Vector{T}, trunc::TensorKit.TruncationScheme, revs::Vector{Bool}
) where {T<:AbstractTensorMap}
    # M1 -- (R1,L1) -- M2 -- (R2,L2) -- M3
    N = length(Ms)
    Rs = Vector{AbstractTensorMap}(undef, N - 1)
    Ls = Vector{AbstractTensorMap}(undef, N - 1)
    for n in 1:(N - 1)
        m = N - n + 1
        _, Rs[n] = qr_through((n == 1) ? nothing : Rs[n - 1], Ms[n]; normalize=true)
        Ls[m - 1], _ = lq_through(Ms[m], (m == N) ? nothing : Ls[m]; normalize=true)
    end
    # find projectors on each internal bond
    Pas = Vector{AbstractTensorMap}(undef, N - 1)
    wts = Vector{DiagonalTensorMap}(undef, N - 1)
    Pbs = Vector{AbstractTensorMap}(undef, N - 1)
    # local truncation error on each bond
    ϵs = zeros(N - 1)
    for (i, (R, L, rev)) in enumerate(zip(Rs, Ls, revs))
        trunc2 = if isa(trunc, FixedSpaceTruncation)
            truncspace(space(Ms[i + 1], 1))
        else
            trunc
        end
        Pas[i], wts[i], Pbs[i], ϵs[i] = _proj_from_RL(R, L; trunc=trunc2, rev)
    end
    # apply projectors
    # M1 -- (Pa1,wt1,Pb1) -- M2 -- (Pa2,wt2,Pb2) -- M3
    for (i, (Pa, Pb)) in enumerate(zip(Pas, Pbs))
        @plansor (Ms[i])[-1; -2 -3 -4 -5] := (Ms[i])[-1; -2 -3 -4 1] * Pa[1; -5]
        @tensor (Ms[i + 1])[-1; -2 -3 -4 -5] := Pb[-1; 1] * (Ms[i + 1])[1; -2 -3 -4 -5]
    end
    return wts, ϵs
end

"""
Apply the gate MPO on the cluster and truncate the bond
```
        ↑       ↑       ↑
        g1 -←-- g2 -←-- g3
        ↑       ↑       ↑
        | ╱     | ╱     | ╱
    --- M1 ---- M2 ---- M3 ---
       ╱       ╱       ╱
```
In the cluster, the axes of each PEPSTensor are reordered as
```
         2  3
         | ╱
    1 -- M -- 5     M[1; 2 3 4 5]
        ╱
       4
```
"""
function apply_gatempo!(
    Ms::Vector{T1}, gs::Vector{T2}; trunc::TensorKit.TruncationScheme
) where {T1<:AbstractTensorMap,T2<:AbstractTensorMap}
    @assert length(Ms) == length(gs)
    revs = [isdual(space(M, 1)) for M in Ms[2:end]]
    _apply_gatempo!(Ms, gs)
    wts, ϵs = _cluster_truncate!(Ms, trunc, revs)
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
    cluster = Vector{AbstractTensorMap}(undef, 3)
    for (i, (coord, sqrtwts, perm)) in enumerate(zip(coords_sw, sqrtwts_sw, perms_sw))
        M = peps.vertices[CartesianIndex(coord)]
        M = _absorb_weights(M, peps.weights, coord[1], coord[2], Tuple(1:4), sqrtwts, false)
        cluster[i] = permute(M, perm)
    end
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
    cluster = Vector{AbstractTensorMap}(undef, 3)
    for (i, (coord, sqrtwts, perm)) in enumerate(zip(coords_se, sqrtwts_se, perms_se))
        M = peps.vertices[CartesianIndex(coord)]
        M = _absorb_weights(M, peps.weights, coord[1], coord[2], Tuple(1:4), sqrtwts, false)
        cluster[i] = permute(M, perm)
    end
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
    su3site_iter(gatempos::Dict{Symbol}, peps::InfiniteWeightPEPS, alg::SimpleUpdate)

One round of 3-site simple update for Hamiltonian with 2nd neighbor terms. 
"""
function su3site_iter(gatempos::Dict{Symbol}, peps::InfiniteWeightPEPS, alg::SimpleUpdate)
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

"""
    FullEnvTruncation

Algorithm struct for the full environment truncation (FET).
"""
@kwdef struct FullEnvTruncation
    trscheme::TensorKit.TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1e-8
    verbose::Bool = false
    check_int::Int = 1
end

"""
Given the bond environment `env`, calculate the inner product
between two states specified by the bond matrices `b1`, `b2`
```
                ┌──←──┐   ┌──←──┐
                │     │   │     │
                │   ┌─┴───┴─┐   │
    ⟨b1|b2⟩ =   b1† │  env  │   b2
                │   └─┬───┬─┘   │
                │     │   │     │
                └──←──┘   └──←──┘
```
"""
function inner_prod(
    env::AbstractTensorMap{T,S,2,2}, b1::AbstractTensor{T,S,2}, b2::AbstractTensor{T,S,2}
) where {T<:Number,S<:ElementarySpace}
    # dual check
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    val = @tensor conj(b1[1 2]) * env[1 2; 3 4] * b2[3 4]
    return val
end

"""
Given the bond environment `env`, calculate the fidelity
between two states specified by the bond matrices `b1`, `b2`
```
    F(b1, b2) = (⟨b1|b2⟩ ⟨b2|b1⟩) / (⟨b1|b1⟩ ⟨b2|b2⟩)
```
"""
function fidelity(
    env::AbstractTensorMap{T,S,2,2}, b1::AbstractTensor{T,S,2}, b2::AbstractTensor{T,S,2}
) where {T<:Number,S<:ElementarySpace}
    return abs2(inner_prod(env, b1, b2)) /
           real(inner_prod(env, b1, b1) * inner_prod(env, b2, b2))
end

"""
Given a fixed state `|b0⟩` with bond matrix `b0`, 
find the state `|b⟩` with truncated bond matrix `b = u s v†`
that maximizes the fidelity (not normalized by `⟨b0|b0⟩`)
```
    F(b) = ⟨b|b0⟩⟨b0|b⟩ / ⟨b|b⟩

                ┌──←──┐   ┌──←──┐   ┌──←──┐   ┌──←──┐
                v     │   │     │   │     │   │     v†
                ↑   ┌─┴───┴─┐   │   │   ┌─┴───┴─┐   ↓
                s   │  env  │   b0  b0† │  env  │   s
                ↑   └─┬───┬─┘   │   │   └─┬───┬─┘   ↓
                u†    │   │     │   │     │   │     u
                └──←──┘   └──←──┘   └──←──┘   └──←──┘
            = ──────────────────────────────────────────
                        ┌──←──┐   ┌──←──┐
                        v     │   │     v†
                        ↑   ┌─┴───┴─┐   ↓
                        s   │  env  │   s
                        ↑   └─┬───┬─┘   ↓
                        u†    │   │     u
                        └──←──┘   └──←──┘
```
- The bond environment `env` is positive definite 
    (Hermitian with positive (at least non-negative) eigenvalues). 
- The singular value spectrum `s` is truncated to desired dimension, 
    and normalized such that the maximum is 1.

The algorithm iteratively optimizes the vectors `l`, `r`
```
                      ┌─┐                     ┌─┐
          ┌─┐         │ ↓         ┌─┐         │ ↑
        →─┘ │       →─┘ s       ←─┘ │       ←─┘ v†
            l   =       ↓   ,       r   =       ↓
        ←─┐ │       ←─┐ u       ←─┐ │       ←─┐ s 
          └─┘         │ ↓         └─┘         │ ↓
                      └─┘                     └─┘
```

## Optimization of `r`

Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌─←─┐   ┌───┐
                │   │           │   │   │   │  
                │   └─←         │  ┌┴───┴┐  └─←
                p†          =  b0† │ env │ 
                │   ┌─←         │  └┬───┬┘  ┌─←
                │   │           │   │   │   u
                └───┘           └─←─┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         │   │   │   │
        ←─┘  ┌┴───┴┐  └─←     ←─┘  ┌┴───┴┐  └─←
             │  B  │        =      │ env │
        ←─┐  └┬───┬┘  ┌─←     ←─┐  └┬───┬┘  ┌─←
          │   │   │   │         u†  │   │   u
          └───┘   └───┘         └───┘   └───┘
```
Then (each index corresponds to a pair of fused indices)
```
    F(r,r†) = |p† r|² / (r† B r)
            = (r† p) (p† r) / (r† B r)
```
which is maximized when
```
    ∂F/∂(r̄_a) * (r† B r)²
    = p (p† r) (r† B r) - |p† r|² (B r) = 0
```
Note that `B` is positive (consequently `B† = B`). 
Then the solution for the vector `r` is
```
    r = B⁻¹ p
```
We can verify that (using `B† = B`)
```
    ∂F/∂(r̄_a) * (r† B r)²
    = p (p† B⁻¹ p) (p† B⁻¹ B B⁻¹ p) - |p† B⁻¹ p|² (B B⁻¹ p) 
    = 0
```
Then the bond matrix `u s v†` is updated by truncated SVD:
```
    ← u ← r →    ==>    ← u ← s ← v† →
```

## Optimization of `l`

The process is entirely similar. 
Define the vector `p` and the positive map `B` as
```
                ┌───┐           ┌─←─┐   ┌───┐
                │   │           │   │   │   v†
                │   └o→         │  ┌┴───┴┐  └o→
                p†          =  b0† │ env │ 
                │   ┌─←         │  └┬───┬┘  ┌─←
                │   │           │   │   │   │
                └───┘           └─←─┘   └───┘

          ┌───┐   ┌───┐         ┌───┐   ┌───┐
          │   │   │   │         v   │   │   v†
        →o┘  ┌┴───┴┐  └o→     →o┘  ┌┴───┴┐  └o→
             │  B  │        =      │ env │
        ←─┐  └┬───┬┘  ┌─←     ←─┐  └┬───┬┘  ┌─←
          │   │   │   │         │   │   │   │
          └───┘   └───┘         └───┘   └───┘
```
Here `o` is the parity tensor needed for the fermion case,
    which can be incorporated into `vh` by a `twist`. 
Then (each index corresponds to a pair of fused indices)
```
    F(l,l†) = |p† l|² / (l† B l)
```
which is maximized when
```
    l = B⁻¹ p
```
Then the bond matrix `u s v†` is updated by SVD:
```
    ← l ← v† →   ==>    ← u ← s ← v† →
```

## Returns

The SVD result of the new bond matrix `u`, `s`, `vh`.
The arrows among them are `← u ← s ← v† →`.

Reference: Physical Review B 98, 085155 (2018)
"""
function fullenv_truncate(
    env::AbstractTensorMap{T,S,2,2}, b0::AbstractTensor{T,S,2}, alg::FullEnvTruncation
) where {T<:Number,S<:ElementarySpace}
    # ensure fermion sign will not appear
    @assert [isdual(space(env, ax)) for ax in 1:4] == [0, 0, 1, 1]
    @assert [isdual(space(b0, ax)) for ax in 1:2] == [0, 0]
    # initialize truncated `u, s, v†`
    u, s, vh = tsvd(b0, ((1,), (2,)); trunc=alg.trscheme)
    # normalize `s` (bond matrices can always be normalized)
    s /= norm(s, Inf)
    s0 = deepcopy(s)
    diff_fid, diff_wt, fid, fid0 = NaN, NaN, 0.0, 0.0
    for iter in 1:(alg.maxiter)
        time0 = time()
        # update `← r →  =  ← s ← v† →`
        @tensor r[-1 -2] := s[-1 1] * vh[1 -2]
        @tensor p[-1 -2] := conj(u[1 -1]) * env[1 -2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(u[1 -1]) * env[1 -2; 3 -4] * u[3 -3]
        r, info_r = linsolve(x -> B * x, p, r, 0, 1)
        @tensor b1[-1; -2] := u[-1 1] * r[1 -2]
        u, s, vh = tsvd(b1; trunc=alg.trscheme)
        s /= norm(s, Inf)
        # update `← l ←  =  ← u ← s ←`
        vh2 = twist(vh, 1)
        @tensor l[-1 -2] := u[-1 1] * s[1 -2]
        @tensor p[-1 -2] := conj(vh2[-2 2]) * env[-1 2; 3 4] * b0[3 4]
        @tensor B[-1 -2; -3 -4] := conj(vh2[-2 2]) * env[-1 2; -3 4] * vh2[-4 4]
        l, info_l = linsolve(x -> B * x, p, l, 0, 1)
        @tensor b1[-1; -2] := l[-1 1] * vh[1 -2]
        u, s, vh = tsvd(b1; trunc=alg.trscheme)
        s /= norm(s, Inf)
        # determine convergence
        fid = fidelity(env, b0, permute(b1, (1, 2)))
        diff_wt = (space(s) == space(s0)) ? _singular_value_distance((s, s0)) : NaN
        diff_fid = fid - fid0
        # @assert diff_fid >= -1e-14 "Fidelity is decreasing by $diff_fid."
        time1 = time()
        message = @sprintf(
            "%4d:  fid = %10.5e,  Δfid = %10.4e,  |Δs| = %10.4e,  time = %.3e s\n",
            iter,
            fid,
            diff_fid,
            diff_wt,
            time1 - time0
        )
        s0 = deepcopy(s)
        fid0 = fid
        if iter == alg.maxiter
            @warn "FET cancel" * message
        end
        if alg.verbose && (iter == 1 || iter % alg.check_int == 0 || diff_wt < alg.tol)
            @info ((diff_wt < alg.tol) ? "FET conv  " : "FET iter  ") * message
        end
        if diff_wt < alg.tol
            break
        end
    end
    return u, s, vh, (; fid, diff_fid, diff_wt)
end

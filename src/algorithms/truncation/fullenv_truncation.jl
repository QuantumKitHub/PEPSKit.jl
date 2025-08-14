"""
$(TYPEDEF)

Algorithm struct for the full environment truncation (FET).

## Fields

$(TYPEDFIELDS)

## Constructors

    FullEnvTruncation(; kwargs...)

The truncation algorithm can be constructed from the following keyword arguments:

* `trscheme::TruncationScheme` : SVD truncation scheme when optimizing the new bond matrix.
* `maxiter::Int=50` : Maximal number of FET iterations.
* `tol::Float64=1e-15` : FET converges when fidelity change between two FET iterations is smaller than `tol`.
* `trunc_init::Bool=true` : Controls whether the initialization of the new bond matrix is obtained from truncated SVD of the old bond matrix. 
* `check_interval::Int=0` : Set number of iterations to print information. Output is suppressed when `check_interval <= 0`. 

## References

* [Glen Evenbly, Phys. Rev. B 98, 085155 (2018)](@cite evenbly_gauge_2018). 
"""
@kwdef struct FullEnvTruncation
    trscheme::TruncationScheme
    maxiter::Int = 50
    tol::Float64 = 1.0e-15
    trunc_init::Bool = true
    check_interval::Int = 0
end

"""
$(SIGNATURES)

Given the bond environment `benv`, calculate the inner product
between two states specified by the bond matrices `b1`, `b2`
```
            ┌--------------------┐
            |   ┌----┐           |
            └---|    |---- b2 ---┘
    ⟨b1|b2⟩ =   |benv|
            ┌---|    |---- b1†---┐
            |   └----┘           |
            └--------------------┘
```
"""
function inner_prod(
        benv::BondEnv{T, S}, b1::AbstractTensorMap{T, S, 1, 1}, b2::AbstractTensorMap{T, S, 1, 1}
    ) where {T <: Number, S <: ElementarySpace}
    val = @tensor conj(b1[1; 2]) * benv[1 2; 3 4] * b2[3; 4]
    return val
end

"""
$(SIGNATURES)

Given the bond environment `benv`, calculate the fidelity
between two states specified by the bond matrices `b1`, `b2`
```
    F(b1, b2) = (⟨b1|b2⟩ ⟨b2|b1⟩) / (⟨b1|b1⟩ ⟨b2|b2⟩)
```
"""
function fidelity(
        benv::BondEnv{T, S}, b1::AbstractTensorMap{T, S, 1, 1}, b2::AbstractTensorMap{T, S, 1, 1}
    ) where {T <: Number, S <: ElementarySpace}
    return abs2(inner_prod(benv, b1, b2)) /
        real(inner_prod(benv, b1, b1) * inner_prod(benv, b2, b2))
end

"""
$(SIGNATURES)

Apply a twist to domain or codomain indices that correspond to dual spaces
"""
function _linearmap_twist!(t::AbstractTensorMap)
    for ax in 1:numout(t)
        isdual(codomain(t, ax)) && twist!(t, ax)
    end
    for ax in 1:numin(t)
        isdual(domain(t, ax)) && twist!(t, numout(t) + ax)
    end
    return nothing
end

function _fet_message(
        iter::Int, fid::Float64, Δfid::Float64, Δwt::Float64, time_elapsed::Float64
    )
    return @sprintf("%5d: fid = %.8e, Δfid = %.8e, ", iter, fid, Δfid) *
        @sprintf("|Δs| = %.6e, time = %.4f s", Δwt, time_elapsed)
end

"""
    fullenv_truncate(benv::BondEnv{T,S}, b0::AbstractTensorMap{T,S,1,1}, alg::FullEnvTruncation) -> U, S, V, info

Perform full environment truncation algorithm from
[Phys. Rev. B 98, 085155 (2018)](@cite evenbly_gauge_2018) on `benv`.

Given a fixed state `|b0⟩` with bond matrix `b0`
and the corresponding positive-definite bond environment `benv`, 
find the state `|b⟩` with truncated bond matrix `b = u s v†`
that maximizes the fidelity (not normalized by `⟨b0|b0⟩`)
```
    F(b) = ⟨b|b0⟩⟨b0|b⟩ / ⟨b|b⟩

            ┌----------------------┐  ┌-----------------------┐
            |   ┌----┐             |  |   ┌----┐              |
            └---|    |---- b0 -----┘  └---|    |- u ← s ← v† -┘
                |benv|                    |benv|
            ┌---|    |-u† → s → v -┐  ┌---|    |----- b0† ----┐
            |   └----┘             |  |   └----┘              |
            └----------------------┘  └-----------------------┘
        = ───────────────────────────────────────────────────────
                        ┌-----------------------┐
                        |   ┌----┐              |
                        └---|    |- u ← s ← v† -┘
                            |benv|
                        ┌---|    |- u† → s → v -┐
                        |   └----┘              |
                        └-----------------------┘
```
The singular value spectrum `s` is truncated to desired dimension, 
and normalized such that the maximum is 1.
Note that `benv` is contracted to `b0` using `@tensor`, 
instead of acting on `b0` as a linear map.

The algorithm iteratively optimizes the vectors `l`, `r`
```
    --- l -←-  =  --- u ← s -←-  ,  -←- r ---  =  -←- s ← v† ---
```

## Optimization of `r`

Define the vector `p` and the positive map `B` as
```
        ┌---------------┐   ┌-----------------------┐
        |   ┌---┐       |   |   ┌----┐              |
        └---|   |-←   --┘   └---|    |- u ←      ---┘
            | p†|         =     |benv|
        ┌---|   |-------┐   ┌---|    |----- b0† ----┐
        |   └---┘       |   |   └----┘              |
        └---------------┘   └-----------------------┘

        ┌---------------┐   ┌-----------------------┐
        |   ┌---┐       |   |   ┌----┐              |
        └---|   |-←   --┘   └---|    |- u ←      ---┘
            | B |         =     |benv|
        ┌---|   |-→   --┐   ┌---|    |- u†→      ---┐
        |   └---┘       |   |   └----┘              |
        └---------------┘   └-----------------------┘
```
Then (each index corresponds to a pair of fused indices)
```
    F(r,r†) = |p† r|² / (r† B r)
            = (r† p) (p† r) / (r† B r)
```
which is maximized when
```
    ∂F/∂r̄ * (r† B r)²
    = p (p† r) (r† B r) - |p† r|² (B r) = 0
```
Note that `B` is positive (consequently `B† = B`). 
Then the solution for the vector `r` is
```
    r = B⁻¹ p
```
We can verify that (using `B† = B`)
```
    ∂F/∂r̄ * (r† B r)²
    = p (p† B⁻¹ p) (p† B⁻¹ B B⁻¹ p) - |p† B⁻¹ p|² (B B⁻¹ p) 
    = 0
```
Then the bond matrix `u s v†` is updated by truncated SVD:
```
    - u ← r -    ==>    - u ← s ← v† -
```

## Optimization of `l`

The process is entirely similar. 
Define the vector `p` and the positive map `B` as
```
        ┌---------------┐   ┌-----------------------┐
        |   ┌---┐       |   |   ┌----┐              |
        └---|   |-  ←-o-┘   └---|    |--   ←-o- v† -┘
            | p†|         =     |benv|
        ┌---|   |-------┐   ┌---|    |----- b0† ----┐
        |   └---┘       |   |   └----┘              |
        └---------------┘   └-----------------------┘

        ┌---------------┐   ┌-----------------------┐
        |   ┌---┐       |   |   ┌----┐              |
        └---|   |-  ←-o-┘   └---|    |--    ←-o- v†-┘
            | B |         =     |benv|
        ┌---|   |-  →-o-┐   ┌---|    |--    →-o- v -┐
        |   └---┘       |   |   └----┘              |
        └---------------┘   └-----------------------┘
```
Here `o` is the parity tensor (twist) necessary for fermions. 
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
    - l ← v† -   ==>    - u ← s ← v† -
```

## Return values

Returns the SVD result of the new bond matrix `U`, `S`, `V`, as well as an information
`NamedTuple` containing the following fields:

* `fid` : Last fidelity.
* `Δfid` : Last fidelity difference.
* `Δs` : Last singular value difference.
"""
function fullenv_truncate(
        b0::AbstractTensorMap{T, S, 1, 1}, benv::BondEnv{T, S}, alg::FullEnvTruncation
    ) where {T <: Number, S <: ElementarySpace}
    verbose = (alg.check_interval > 0)
    # `benv` is assumed to be positive; here we only check codomain(benv) == domain(benv).
    @assert codomain(benv) == domain(benv)
    time00 = time()
    # initialize u, s, vh with truncated or untruncated SVD
    u, s, vh = tsvd(b0; trunc = (alg.trunc_init ? alg.trscheme : notrunc()))
    b1 = similar(b0)
    # normalize `s` (bond matrices can always be normalized)
    s /= norm(s, Inf)
    s0 = deepcopy(s)
    Δfid, Δs, fid, fid0 = NaN, NaN, 0.0, 0.0
    for iter in 1:(alg.maxiter)
        time0 = time()
        # update `← r -  =  ← s ← v† -`
        @tensor r[-1 -2] := s[-1; 1] * vh[1; -2]
        @tensor p[-1 -2] := conj(u[1; -1]) * benv[1 -2; 3 4] * b0[3; 4]
        @tensor B[-1 -2; -3 -4] := conj(u[1; -1]) * benv[1 -2; 3 -4] * u[3; -3]
        _linearmap_twist!(p)
        _linearmap_twist!(B)
        r, info_r = linsolve(Base.Fix1(*, B), p, r, 0, 1)
        @tensor b1[-1; -2] = u[-1; 1] * r[1 -2]
        u, s, vh = tsvd(b1; trunc = alg.trscheme)
        s /= norm(s, Inf)
        # update `- l ←  =  - u ← s ←`
        @tensor l[-1 -2] := u[-1; 1] * s[1; -2]
        @tensor p[-1 -2] := conj(vh[-2; 2]) * benv[-1 2; 3 4] * b0[3; 4]
        @tensor B[-1 -2; -3 -4] := conj(vh[-2; 2]) * benv[-1 2; -3 4] * vh[-4; 4]
        _linearmap_twist!(p)
        _linearmap_twist!(B)
        l, info_l = linsolve(Base.Fix1(*, B), p, l, 0, 1)
        @tensor b1[-1; -2] = l[-1 1] * vh[1; -2]
        fid = fidelity(benv, b0, b1)
        u, s, vh = tsvd!(b1; trunc = alg.trscheme)
        s /= norm(s, Inf)
        # determine convergence
        Δs = (space(s) == space(s0)) ? _singular_value_distance((s, s0)) : NaN
        Δfid = fid - fid0
        s0 = deepcopy(s)
        fid0 = fid
        time1 = time()
        converge = (Δfid < alg.tol)
        cancel = (iter == alg.maxiter)
        showinfo =
            cancel || (verbose && (converge || iter == 1 || iter % alg.check_interval == 0))
        if showinfo
            message = _fet_message(
                iter, fid, Δfid, Δs, time1 - ((cancel || converge) ? time00 : time0)
            )
            if converge
                @info "FET conv" * message
            elseif cancel
                @warn "FET cancel" * message
            else
                @info "FET iter" * message
            end
        end
        converge && break
    end
    return u, s, vh, (; fid, Δfid, Δs)
end

"""
    FixedSpaceTruncation <: TensorKit.TruncationScheme

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

"""
    ProjectorAlgorithm

Abstract super type for all CTMRG projector algorithms.
"""
abstract type ProjectorAlgorithm end

function svd_algorithm(alg::ProjectorAlgorithm, (dir, r, c))
    if alg.svd_alg isa SVDAdjoint{<:FixedSVD}
        fwd_alg = alg.svd_alg.fwd_alg
        fix_svd = FixedSVD(fwd_alg.U[dir, r, c], fwd_alg.S[dir, r, c], fwd_alg.V[dir, r, c])
        return SVDAdjoint(; fwd_alg=fix_svd, rrule_alg=alg.svd_alg.rrule_alg)
    else
        return alg.svd_alg
    end
end

function truncation_scheme(alg::ProjectorAlgorithm, edge)
    if alg.trscheme isa FixedSpaceTruncation
        return truncspace(space(edge, 1))
    else
        return alg.trscheme
    end
end

"""
    struct HalfInfiniteProjector{S,T}(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.
"""
@kwdef struct HalfInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

"""
    struct FullInfiniteProjector{S,T}(; svd_alg=Defaults.svd_alg,
                                      trscheme=Defaults.trscheme, verbosity=0)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.
"""
@kwdef struct FullInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

# TODO: add `LinearAlgebra.cond` to TensorKit
# Compute condition number smax / smin for diagonal singular value TensorMap
function _condition_number(S::AbstractTensorMap)
    smax = maximum(first ∘ last, blocks(S))
    smin = maximum(last ∘ last, blocks(S))
    return smax / smin
end
@non_differentiable _condition_number(S::AbstractTensorMap)

"""
    compute_projector(L::AbstractTensorMap, R::AbstractTensorMap, svd_alg, alg)

Given the bond connecting the left and right tensors, e.g. L and R, we compute the projectors on the bond.
This is a general algorithm that can be used for any bond. The only thing you need worry about
is the left and right tensors. After the projection, the arrow of the bond is now (L⊙P_L)←(P_R⊙R).

```
L⊙R=(L⊙R)*(L⊙R)^-1*(L⊙R)=(L⊙R)*(U*S*V)^-1*(L⊙R)=(L⊙R)*V'S^{-1}U'*(L⊙R)=L⊙(R*V'*S^{-1/2})*(S^{-1/2}*U'*L)⊙R=L⊙P_L*P_R⊙R
```
""" 
#help function for the projection, particularly for the sign of fermions
⊙(t1::AbstractTensorMap, t2::AbstractTensorMap)=twist(t1, filter(i -> !isdual(space(t1, i)), domainind(t1)))*t2

function compute_projector(L::AbstractTensorMap, R::AbstractTensorMap, svd_alg, alg)
    if dim(codomain(L)) > dim(domain(L))
        _,L= leftorth!(L)
        R,_=rightorth!(R)
    end
    LR=L⊙R
    n_factor=norm(LR)
    LR=LR/n_factor
    
    U, S, V, truncation_error = PEPSKit.tsvd!(LR, svd_alg; trunc=alg.trscheme)

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    norm_factor = sqrt(n_factor)
    isqS = sdiag_pow(S, -0.5)
    PL = R * V' * isqS / norm_factor
    PR = isqS * U' * L / norm_factor
    
    truncation_error /= norm(S)
    condition_number = ignore_derivatives() do
        _condition_number(S)
    end
    return (PL, PR), (; truncation_error, condition_number, U, S, V)
end

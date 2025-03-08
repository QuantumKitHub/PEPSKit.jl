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
    compute_projector(L::AbstractTensorMap, R::AbstractTensorMap, alg::ProjectorAlgorithm)

Compute projection operators for the dimension truncation on the bond between tensors L and R.

# Visual Representation
             ----     ||       ----
    --->----|    |    ||      |    |.........
    ........|    |    ||      |    |---->----
    ........|    |----||->----|    |.........
    ---<--- |  L |....||......| R  |.........
    ........|    |----||-<----|    |----<----
    ---->---|    |    ||      |    |
             ----     ||       ----
                    Cut Here

# Description
Projection operators are essential for truncating bond dimensions while preserving the most important weights during the CTMRG iteration.

# Mathematical Foundation
The tensor contraction `L ⊙ R` can be treated as a map and decomposed using SVD:
    
    L ⊙ R = U * S * V'

where `*` denotes map composition (distinct from tensor contraction `⊙`). Thus we have 

    L ⊙ R = (L ⊙ R) * (L ⊙ R)⁻¹ * (L ⊙ R)
           = (L ⊙ R) * V' * S⁻¹ * U' * (L ⊙ R)
           = L ⊙ (R * V' * S^(-1/2)) * (S^(-1/2) * U' * L) ⊙ R
           = L ⊙ P_L * P_R ⊙ R

From this decomposition, we define the projectors:
    
    P_L = R * V' * S^(-1/2)
    P_R = S^(-1/2) * U' * L

These projectors allow us to rewrite: `L ⊙ R ≃ L ⊙ P_L * P_R ⊙ R`, implying that the bond arrow
(`←`) is from `P_R ⊙ R` to `L ⊙ P_L`: `L ⊙ P_L ← P_R ⊙ R`.

# Parameters
- `L::AbstractTensorMap`: Left tensor in the contraction
- `R::AbstractTensorMap`: Right tensor in the contraction
- `alg::ProjectorAlgorithm`: Algorithm specification for `PEPSKit.tsvd!` with fields `svd_alg` and `trscheme`.

# Returns
- `(P_L, P_R)`: A tuple of two tensor maps representing the left and right projectors.
- `(; truncation_error, condition_number, U, S, V)`: A named tuple containing:
  - `truncation_error`: Estimated error from dimension truncation
  - `condition_number`: Ratio of largest to smallest singular values
  - `U, S, V`: Components from the underlying SVD decomposition

# Usage Examples
- For half-infinite environment: Use `L = C1` and `R = C2`
- For full-infinite environment: Use `L = C4 ⊙ C1` and `R = C2 ⊙ C3`

# Implementation Note
For correct fermion sign handling in fPEPS:
- Linear algebra operations must use map composition (`*`)
- General tensor networks use tensor contraction (`⊙`)
- `@tensor` macro in `TensorKit.jl` handles more general tensor contraction (`⊙`) automatically. `⊙` is used for clarity in the formal derivation of the projectors.
"""
compute_projector(L::AbstractTensorMap, R::AbstractTensorMap, alg::ProjectorAlgorithm)

#helper function for projection, particularly for the sign of fermions
function ⊙(t1::AbstractTensorMap, t2::AbstractTensorMap)
    return twist(t1, filter(i -> !isdual(space(t1, i)), domainind(t1))) * t2
end

function compute_projector(
    L::AbstractTensorMap, R::AbstractTensorMap, alg::ProjectorAlgorithm
)
    if dim(codomain(L)) > dim(domain(L))
        _, L = leftorth!(L)
        R, _ = rightorth!(R)
    end
    LR = L ⊙ R
    n_factor = norm(LR)
    LR = LR / n_factor

    U, S, V, truncation_error = PEPSKit.tsvd!(LR, alg.svd_alg; trunc=alg.trscheme)

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

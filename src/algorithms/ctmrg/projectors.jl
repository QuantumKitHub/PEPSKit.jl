"""
$(TYPEDEF)

Abstract super type for all CTMRG projector algorithms.
"""
abstract type ProjectorAlgorithm end

const PROJECTOR_SYMBOLS = IdDict{Symbol,Type{<:ProjectorAlgorithm}}()

"""
    ProjectorAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `ProjectorAlgorithm` algorithm struct.
"""
function ProjectorAlgorithm(;
    alg=Defaults.projector_alg,
    svd_alg=(;),
    trscheme=(;),
    verbosity=Defaults.projector_verbosity,
)
    # replace symbol with projector alg type
    haskey(PROJECTOR_SYMBOLS, alg) ||
        throw(ArgumentError("unknown projector algorithm: $alg"))
    alg_type = PROJECTOR_SYMBOLS[alg]

    # parse SVD forward & rrule algorithm
    svd_algorithm = _alg_or_nt(SVDAdjoint, svd_alg)

    # parse truncation scheme
    truncation_scheme = if trscheme isa TruncationScheme
        trscheme
    elseif trscheme isa NamedTuple
        _TruncationScheme(; trscheme...)
    else
        throw(ArgumentError("unknown trscheme $trscheme"))
    end

    return alg_type(svd_algorithm, truncation_scheme, verbosity)
end

function svd_algorithm(alg::ProjectorAlgorithm, (dir, r, c))
    if alg.svd_alg isa SVDAdjoint{<:FixedSVD}
        fwd_alg = alg.svd_alg.fwd_alg
        fix_svd = if isfullsvd(alg.svd_alg.fwd_alg)
            FixedSVD(
                fwd_alg.U[dir, r, c],
                fwd_alg.S[dir, r, c],
                fwd_alg.V[dir, r, c],
                fwd_alg.U_full[dir, r, c],
                fwd_alg.S_full[dir, r, c],
                fwd_alg.V_full[dir, r, c],
            )
        else
            FixedSVD(
                fwd_alg.U[dir, r, c],
                fwd_alg.S[dir, r, c],
                fwd_alg.V[dir, r, c],
                nothing,
                nothing,
                nothing,
            )
        end
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
$(TYPEDEF)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.

## Fields

$(TYPEDFIELDS)

## Constructors

    HalfInfiniteProjector(; kwargs...)

Construct the half-infinite projector algorithm based on the following keyword arguments:

* `svd_alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg::Symbol=:$(Defaults.trscheme))` : Truncation scheme for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerr` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncdim` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:truncbelow` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `verbosity::Int=$(Defaults.projector_verbosity)` : Projector output verbosity which can be:
    0. Suppress output information
    1. Print singular value degeneracy warnings
"""
struct HalfInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S
    trscheme::T
    verbosity::Int
end
function HalfInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg=:halfinfinite, kwargs...)
end

PROJECTOR_SYMBOLS[:halfinfinite] = HalfInfiniteProjector

"""
$(TYPEDEF)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullInfiniteProjector(; kwargs...)

Construct the full-infinite projector algorithm based on the following keyword arguments:

* `svd_alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trscheme::Union{TruncationScheme,NamedTuple}=(; alg::Symbol=:$(Defaults.trscheme))` : Truncation scheme for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerr` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncdim` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:truncbelow` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `verbosity::Int=$(Defaults.projector_verbosity)` : Projector output verbosity which can be:
    0. Suppress output information
    1. Print singular value degeneracy warnings
"""
struct FullInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S
    trscheme::T
    verbosity::Int
end
function FullInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg=:fullinfinite, kwargs...)
end

PROJECTOR_SYMBOLS[:fullinfinite] = FullInfiniteProjector

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
    L::AbstractTensorMap, R::AbstractTensorMap, svd_alg::SVDAdjoint, trscheme::TruncationScheme
)
    # L = deepcopy(L) / norm(L)
    # R = deepcopy(R) / norm(R)
    # if dim(codomain(L)) > dim(domain(L))
    #     _, L = leftorth!(L)
    #     R, _ = rightorth!(R)
    # end
    LR = L ⊙ R
    n_factor = norm(LR)
    LR = LR / n_factor

    U, S, V, info = PEPSKit.tsvd!(LR, svd_alg; trunc=trscheme)

    # Check for degenerate singular values
    # Zygote.isderiving() && ignore_derivatives() do
    #     if alg.verbosity > 0 && is_degenerate_spectrum(S)
    #         svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
    #         @warn("degenerate singular values detected: ", svals)
    #     end
    # end

    norm_factor = sqrt(n_factor)
    isqS = sdiag_pow(S, -0.5)
    PL = R * V' * isqS / norm_factor
    PR = isqS * U' * L / norm_factor

    return (PL, PR), (; U, S, V, info...)
end
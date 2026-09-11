"""
$(TYPEDEF)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.

## Fields

$(TYPEDFIELDS)

## Constructors

    HalfInfiniteProjector(; kwargs...)

Construct the half-infinite projector algorithm based on the following keyword arguments:

* `decomposition_alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation strategy for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:FixedSpaceTruncation` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space 
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `verbosity::Int=$(Defaults.projector_verbosity)` : Projector output verbosity which can be:
    0. Suppress output information
    1. Print singular value degeneracy warnings
"""
struct HalfInfiniteProjector{S <: SVDAdjoint, T} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    verbosity::Int
end
function HalfInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :HalfInfiniteProjector, kwargs...)
end

PROJECTOR_SYMBOLS[:HalfInfiniteProjector] = HalfInfiniteProjector

"""
    compute_projector(enlarged_corners, alg::ProjectorAlgorithm)

Determine left and right projectors at the bond given determined by the enlarged corners
using the specified `alg`.
"""
function compute_projector(enlarged_corners, alg::HalfInfiniteProjector)
    # SVD half-infinite environment
    halfinf = half_infinite_environment(enlarged_corners...)
    svd_alg = decomposition_algorithm(alg)
    U, S, V, truncation_error = svd_trunc!(halfinf / norm(halfinf), svd_alg)

    # get some decomposition info
    truncation_error = truncation_error / norm(S) # normalize truncation error

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = contract_projectors(U, S, V, enlarged_corners...)
    return (P_left, P_right), (; U, S, V, truncation_error)
end

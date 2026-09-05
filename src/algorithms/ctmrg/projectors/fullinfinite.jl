"""
$(TYPEDEF)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.

## Fields

$(TYPEDFIELDS)

## Constructors

    FullInfiniteProjector(; kwargs...)

Construct the full-infinite projector algorithm based on the following keyword arguments:

* `decomposition_alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation scheme for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
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
struct FullInfiniteProjector{S <: SVDAdjoint, T} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    verbosity::Int
end
function FullInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :FullInfiniteProjector, kwargs...)
end

PROJECTOR_SYMBOLS[:FullInfiniteProjector] = FullInfiniteProjector

function compute_projector(enlarged_corners, alg::FullInfiniteProjector)
    halfinf_left = half_infinite_environment(enlarged_corners[1], enlarged_corners[2])
    halfinf_right = half_infinite_environment(enlarged_corners[3], enlarged_corners[4])

    # SVD full-infinite environment
    fullinf = full_infinite_environment(halfinf_left, halfinf_right)
    svd_alg = decomposition_algorithm(alg)
    U, S, V, truncation_error = svd_trunc!(fullinf / norm(fullinf), svd_alg)

    # get some decomposition info
    truncation_error = truncation_error / norm(S) # normalize truncation error

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = contract_projectors(U, S, V, halfinf_left, halfinf_right)
    return (P_left, P_right), (; U, S, V, truncation_error)
end

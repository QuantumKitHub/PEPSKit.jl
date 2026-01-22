"""
$(TYPEDEF)

Abstract super type for all CTMRG projector algorithms.
"""
abstract type ProjectorAlgorithm end

const PROJECTOR_SYMBOLS = IdDict{Symbol, Type{<:ProjectorAlgorithm}}()

"""
    ProjectorAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `ProjectorAlgorithm` algorithm struct.
"""
function ProjectorAlgorithm(;
        alg = Defaults.projector_alg,
        decomposition_alg = (;),
        trunc = (;),
        verbosity = Defaults.projector_verbosity,
    )
    # replace symbol with projector alg type
    haskey(PROJECTOR_SYMBOLS, alg) ||
        throw(ArgumentError("unknown projector algorithm: $alg"))
    alg_type = PROJECTOR_SYMBOLS[alg]

    # parse SVD forward & rrule algorithm

    decomposition_algorithm = if alg in [:halfinfinite, :fullinfinite]
        _alg_or_nt(SVDAdjoint, decomposition_alg)
    elseif alg in [:c4v_eigh,]
        _alg_or_nt(EighAdjoint, decomposition_alg)
    elseif alg in [:c4v_qr,]
        _alg_or_nt(QRAdjoint, decomposition_alg)
    end # TODO: how do we solve this in a proper way?

    # parse truncation scheme
    truncation_strategy = if trunc isa TruncationStrategy
        trunc
    elseif trunc isa NamedTuple
        _TruncationStrategy(; trunc...)
    else
        throw(ArgumentError("unknown trunc $trunc"))
    end

    return alg_type(decomposition_algorithm, truncation_strategy, verbosity)
end

decomposition_algorithm(alg::ProjectorAlgorithm) = alg.alg
function decomposition_algorithm(alg::ProjectorAlgorithm, (dir, r, c))
    decomp_alg = decomposition_algorithm(alg)
    if decomp_alg isa SVDAdjoint{<:FixedSVD}
        fwd_alg = decomp_alg.fwd_alg
        fix_svd = if isfullsvd(decomp_alg.fwd_alg)
            FixedSVD(
                fwd_alg.U[dir, r, c], fwd_alg.S[dir, r, c], fwd_alg.V[dir, r, c],
                fwd_alg.U_full[dir, r, c], fwd_alg.S_full[dir, r, c], fwd_alg.V_full[dir, r, c],
            )
        else
            FixedSVD(
                fwd_alg.U[dir, r, c], fwd_alg.S[dir, r, c], fwd_alg.V[dir, r, c],
                nothing, nothing, nothing,
            )
        end
        return SVDAdjoint(; fwd_alg = fix_svd, rrule_alg = decomp_alg.rrule_alg)
    else
        return decomp_alg
    end
end

function truncation_strategy(alg::ProjectorAlgorithm, edge)
    if alg.trunc isa FixedSpaceTruncation
        tspace = space(edge, 1)
        return isdual(tspace) ? truncspace(flip(tspace)) : truncspace(tspace)
    else
        return alg.trunc
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

* `alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation strategy for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
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
    alg::S
    trunc::T
    verbosity::Int
end
function HalfInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :halfinfinite, kwargs...)
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

* `alg::Union{<:SVDAdjoint,NamedTuple}=SVDAdjoint()` : SVD algorithm including the reverse rule. See [`SVDAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation scheme for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:fixedspace` : Keep virtual spaces fixed during projection
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
    alg::S
    trunc::T
    verbosity::Int
end
function FullInfiniteProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :fullinfinite, kwargs...)
end

PROJECTOR_SYMBOLS[:fullinfinite] = FullInfiniteProjector

"""
    compute_projector(enlarged_corners, coordinate, alg::ProjectorAlgorithm)

Determine left and right projectors at the bond given determined by the enlarged corners
and the given coordinate using the specified `alg`.
"""
function compute_projector(enlarged_corners, coordinate, alg::HalfInfiniteProjector)
    # SVD half-infinite environment
    halfinf = half_infinite_environment(enlarged_corners...)
    svd_alg = decomposition_algorithm(alg, coordinate)
    U, S, V, info = svd_trunc!(halfinf / norm(halfinf), svd_alg; trunc = alg.trunc)

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    @reset info.truncation_error = info.truncation_error / norm(S) # normalize truncation error
    P_left, P_right = contract_projectors(U, S, V, enlarged_corners...)
    return (P_left, P_right), (; U, S, V, info...)
end
function compute_projector(enlarged_corners, coordinate, alg::FullInfiniteProjector)
    halfinf_left = half_infinite_environment(enlarged_corners[1], enlarged_corners[2])
    halfinf_right = half_infinite_environment(enlarged_corners[3], enlarged_corners[4])

    # SVD full-infinite environment
    fullinf = full_infinite_environment(halfinf_left, halfinf_right)
    svd_alg = decomposition_algorithm(alg, coordinate)
    U, S, V, info = svd_trunc!(fullinf / norm(fullinf), svd_alg; trunc = alg.trunc)

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    @reset info.truncation_error = info.truncation_error / norm(S) # normalize truncation error
    P_left, P_right = contract_projectors(U, S, V, halfinf_left, halfinf_right)
    return (P_left, P_right), (; U, S, V, info...)
end

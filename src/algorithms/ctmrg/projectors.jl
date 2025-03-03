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
    struct HalfInfiniteProjector{S,T}(; svd_alg=$(Defaults.svd_alg),
                                      trscheme=$(Defaults.trscheme), verbosity=0)

Projector algorithm implementing projectors from SVDing the half-infinite CTMRG environment.
"""
@kwdef struct HalfInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

"""
    struct FullInfiniteProjector{S,T}(; svd_alg=$(Defaults.svd_alg),
                                      trscheme=$(Defaults.trscheme), verbosity=0)

Projector algorithm implementing projectors from SVDing the full 4x4 CTMRG environment.
"""
@kwdef struct FullInfiniteProjector{S<:SVDAdjoint,T} <: ProjectorAlgorithm
    svd_alg::S = Defaults.svd_alg
    trscheme::T = Defaults.trscheme
    verbosity::Int = 0
end

function select_algorithm(::Type{ProjectorAlgorithm}; alg::Union{Symbol,<:ProjectorAlgorithm}, svd_alg, trscheme, verbosity)
    # TODO
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
    compute_projector(enlarged_corners, coordinate, alg::ProjectorAlgorithm)

Determine left and right projectors at the bond given determined by the enlarged corners
and the given coordinate using the specified `alg`.
"""
function compute_projector(enlarged_corners, coordinate, alg::HalfInfiniteProjector)
    # SVD half-infinite environment
    halfinf = half_infinite_environment(enlarged_corners...)
    svd_alg = svd_algorithm(alg, coordinate)
    U, S, V, truncation_error = PEPSKit.tsvd!(halfinf, svd_alg; trunc=alg.trscheme)

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = contract_projectors(U, S, V, enlarged_corners...)
    truncation_error /= norm(S)
    condition_number = @ignore_derivatives(_condition_number(S))
    return (P_left, P_right), (; truncation_error, condition_number, U, S, V)
end
function compute_projector(enlarged_corners, coordinate, alg::FullInfiniteProjector)
    halfinf_left = half_infinite_environment(enlarged_corners[1], enlarged_corners[2])
    halfinf_right = half_infinite_environment(enlarged_corners[3], enlarged_corners[4])

    # SVD full-infinite environment
    fullinf = full_infinite_environment(halfinf_left, halfinf_right)
    svd_alg = svd_algorithm(alg, coordinate)
    U, S, V, truncation_error = PEPSKit.tsvd!(fullinf, svd_alg; trunc=alg.trscheme)

    # Check for degenerate singular values
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = contract_projectors(U, S, V, halfinf_left, halfinf_right)
    condition_number = @ignore_derivatives(_condition_number(S))
    return (P_left, P_right), (; truncation_error, condition_number, U, S, V)
end

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

    decomposition_algorithm = if alg in [:HalfInfiniteProjector, :FullInfiniteProjector]
        _alg_or_nt(SVDAdjoint, decomposition_alg)
    elseif alg in [:C4vEighProjector]
        _alg_or_nt(EighAdjoint, decomposition_alg)
    elseif alg in [:C4vQRProjector]
        _alg_or_nt(QRAdjoint, decomposition_alg)
    end # TODO: how do we solve this in a proper way?

    # qr-ctmrg does not need truncation or degeneracy checks
    if alg in [:C4vQRProjector]
        return alg_type(decomposition_algorithm)
    end

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

"""
    decomposition_algorithm(alg::ProjectorAlgorithm)

Return the tensor decomposition algorithm of the `alg` projector algorithm.
"""
decomposition_algorithm(alg::ProjectorAlgorithm) = alg.decomposition_alg

function truncation_strategy(alg::ProjectorAlgorithm, edge)
    if alg.trunc isa FixedSpaceTruncation
        tspace = space(edge, 1)
        return isdual(tspace) ? truncspace(flip(tspace)) : truncspace(tspace)
    else
        return alg.trunc
    end
end

"""
    _set_truncation(alg::ProjectorAlgorithm, trunc::TruncationStrategy)

Update the truncation strategy of a given projector algorithm, keeping all other settings
the same.
"""
function _set_truncation(alg::ProjectorAlgorithm, trunc::TruncationStrategy)
    alg = @set alg.trunc = trunc
    return alg
end

"""
    _set_decomposition_truncation(alg::ProjectorAlgorithm, trunc::TruncationStrategy)

Set the truncation strategy of the decomposition algorithm within a given projector
algorithm, keeping all other settings the same.
"""
function _set_decomposition_truncation(alg::ProjectorAlgorithm, trunc::TruncationStrategy)
    decomp_alg = decomposition_algorithm(alg)
    truncated_fwd_alg = TruncatedAlgorithm(decomp_alg.fwd_alg, trunc)
    decomp_alg = @set decomp_alg.fwd_alg = truncated_fwd_alg
    alg = @set alg.decomposition_alg = decomp_alg
    return alg
end

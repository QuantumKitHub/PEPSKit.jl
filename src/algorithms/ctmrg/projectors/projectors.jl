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
        _alg_or_nt(OrthAdjoint, decomposition_alg)
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

function truncation_strategy(alg::ProjectorAlgorithm, coordinate = nothing)
    if alg.trunc isa FixedSpaceTruncation
        throw(ArgumentError("`FixedSpaceTruncation` must be replaced by the current spaces of the environment first, see `fix_spaces`"))
    elseif alg.trunc isa SiteDependentTruncation
        (size(alg.trunc.truncs, 1) == 4 && !isnothing(coordinate)) || throw(
            ArgumentError("CTMRG supports a `SiteDependentTruncation` of the environment bonds (4 × rows × cols) with `SimultaneousCTMRG` and `SequentialCTMRG` only")
        )
        return alg.trunc[coordinate...]
    else
        return alg.trunc
    end
end

# replace a `FixedSpaceTruncation` by a truncation to the current spaces of the environment bonds
fix_spaces(alg::CTMRGAlgorithm, env) =
    @set alg.projector_alg = fix_spaces(alg.projector_alg, env)
function fix_spaces(alg::ProjectorAlgorithm, env)
    alg.trunc isa FixedSpaceTruncation || return alg
    return _set_truncation(alg, _fixedspace_truncation(env))
end
_fixedspace_truncation(env::CTMRGEnv) =
    SiteDependentTruncation(map(e -> _nondual_truncspace(space(e, 1)), env.edges))
_fixedspace_truncation(env::C4vCTMRGEnv) = _nondual_truncspace(space(edge(env, WEST, 1, 1), 1))
_nondual_truncspace(V) = truncspace(isdual(V) ? flip(V) : V)
@non_differentiable fix_spaces(args...)

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

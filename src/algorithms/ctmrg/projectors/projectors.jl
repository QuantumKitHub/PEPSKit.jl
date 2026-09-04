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
        orth_alg = Defaults.orth_alg,
        subspace = nothing,
        subspace_tol = Defaults.subspace_tol,
        min_subspace_iters = Defaults.min_subspace_iters,
        trunc = (;),
        verbosity = Defaults.projector_verbosity,
    )
    # replace symbol with projector alg type
    haskey(PROJECTOR_SYMBOLS, alg) ||
        throw(ArgumentError("unknown projector algorithm: $alg"))
    alg_type = PROJECTOR_SYMBOLS[alg]

    # parse SVD forward & rrule algorithm

    decomposition_algorithm = if alg in [:HalfInfiniteProjector, :FullInfiniteProjector, :SubspaceIterationProjector]
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

    if alg == :SubspaceIterationProjector
        truncation_strategy isa FixedSpaceTruncation ||
            throw(ArgumentError("SubspaceIterationProjector only supports FixedSpaceTruncation."))
        subspace isa ElementarySpace ||
            throw(ArgumentError("SubspaceIterationProjector requires an elementary `subspace`."))
        subspace_tol >= 0 || throw(ArgumentError("subspace_tol must be nonnegative."))
        min_subspace_iters >= 1 ||
            throw(ArgumentError("min_subspace_iters must be positive."))
        orth_alg = _alg_or_nt(QRAdjoint, orth_alg)
        return alg_type(
            decomposition_algorithm,
            truncation_strategy,
            orth_alg,
            subspace,
            subspace_tol,
            min_subspace_iters,
            verbosity,
        )
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
Collect the cyclic coordinate layout and target truncation for a projector.

``N`` is the number of enlarged corners involved in the construction of each projector.
- ``N = 2`` for half-infinite environment.
- ``N = 4`` for full-infinite environment.
"""
function _enlarged_corner_layout(
        coordinate::NTuple{3, Int}, env::CTMRGEnv,
        alg::ProjectorAlgorithm, ::Val{N},
    ) where {N}
    rowsize, colsize = size(env)[2:3]
    coordinate2 = _next_coordinate(coordinate, rowsize, colsize)
    trunc = truncation_strategy(alg, env.edges[coordinate[1], coordinate2[2:3]...])
    coordinates = if N == 2
        (coordinate, coordinate2)
    elseif N == 4
        coordinate3 = _next_coordinate(coordinate2, rowsize, colsize)
        coordinate4 = _next_coordinate(coordinate3, rowsize, colsize)
        (coordinate4, coordinate, coordinate2, coordinate3)
    else
        throw(ArgumentError("projectors require either two or four enlarged corners"))
    end
    return coordinates, trunc
end

"""Collect precomputed enlarged corners and the target truncation for simultaneous CTMRG."""
function enlarged_corner_inputs(
        corner_grid::AbstractArray{E, 3}, coordinate::NTuple{3, Int},
        env::CTMRGEnv, alg::ProjectorAlgorithm, count::Val{N},
    ) where {E, N}
    coordinates, trunc = _enlarged_corner_layout(coordinate, env, alg, count)
    enlarged_corners = map(coordinate -> corner_grid[coordinate...], coordinates)
    return enlarged_corners, trunc
end

"""Construct enlarged corners and collect the target truncation for sequential CTMRG."""
function enlarged_corner_inputs(
        network::InfiniteSquareNetwork, coordinate::NTuple{3, Int},
        env::CTMRGEnv, alg::ProjectorAlgorithm, count::Val{N},
    ) where {N}
    coordinates, trunc = _enlarged_corner_layout(coordinate, env, alg, count)
    enlarged_corners = map(coordinates) do coordinate
        return TensorMap(EnlargedCorner(network, env, coordinate))
    end
    return enlarged_corners, trunc
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

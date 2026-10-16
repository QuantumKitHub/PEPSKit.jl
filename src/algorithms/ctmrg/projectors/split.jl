"""
$(TYPEDEF)

Projector algorithm for [`SplitCTMRG`](@ref), constructing the green (Fig. 3), teal (Fig. 7), yellow (Fig. 8), and red (Fig. 9) projector pairs of [Naumann et al.](@cite naumann_variationally_2025).
Each projector pair follows [Fishman et al.](@cite fishman_faster_2018), Appendix A: SVD-factorize each environment half, keep its square-root-weighted factor facing the cut, and truncate the product of these factors.
The square-root weights of the half-environment singular values are masked using the relative cutoff of [`sdiag_pow`](@ref); the final projector pseudoinverse uses the same cutoff.
QR compression is applied to the environment halves for the yellow projectors and, when the truncation permits, the quarter-environments for the red projectors before factorization.
SVD compression is used instead when a QR/LQ triangular factor is numerically rank deficient.

## Fields

$(TYPEDFIELDS)

## Constructors

    SplitProjector(; kwargs...)

Construct a split projector algorithm with `decomposition_alg`, `trunc`, and `verbosity`, following [`FullInfiniteProjector`](@ref).
`trunc` controls the green and teal projectors; [`SplitCTMRG`](@ref) supplies `interlayer_trunc` for the yellow and red projectors.
The half-environment SVDs are untruncated; when `decomposition_alg` uses [`IterSVD`](@ref), these preliminary SVDs use the default dense algorithm instead.
"""
struct SplitProjector{S <: SVDAdjoint, T} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    verbosity::Int
end
function SplitProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :SplitProjector, kwargs...)
end

PROJECTOR_SYMBOLS[:SplitProjector] = SplitProjector

"""Use an untruncated dense decomposition for split environment factors."""
function _split_half_svd_algorithm(alg::SplitProjector)
    half_alg = decomposition_algorithm(alg)
    half_alg.fwd_alg isa IterSVD && (half_alg = SVDAdjoint())
    return @set half_alg.fwd_alg = TruncatedAlgorithm(half_alg.fwd_alg, notrunc())
end

"""Compress a split environment with QR/LQ, falling back to SVD for a rank-deficient triangular factor."""
function _split_compress_half(t::AbstractTensorMap, alg::SplitProjector, ::Val{left}) where {left}
    factor = left ? last(left_orth(t, QRAdjoint())) : first(right_orth(t, LQAdjoint()))
    fullrank = ignore_derivatives() do
        all(blocks(factor)) do (_, b)
            tolerance = MatrixAlgebraKit.default_pullback_rank_atol(b)
            return all(d -> abs(d) > tolerance, diagview(b))
        end
    end
    fullrank && return factor
    # Unpivoted QR/LQ can have internal zero pivots followed by nonzero pivots.
    # Its triangular pullback solve is then singular even when the last pivot is nonzero.
    U, S, V = svd_trunc_no_error(t, _split_half_svd_algorithm(alg))
    return left ? S * V : U * S
end

"""Mask negligible half-environment square-root weights to suppress amplified roundoff."""
function _split_fishman_sqrt(S::DiagonalTensorMap)
    mask = ignore_derivatives() do
        cutoff = eps(real(scalartype(S)))^(3 / 4) * norm(S, Inf)
        DiagonalTensorMap(ifelse.(S.data .>= cutoff, one(scalartype(S)), zero(scalartype(S))), space(S, 1))
    end
    return sdiag_pow(S, 0.5) * mask
end

"""Factor split environment halves into square-root-weighted Fishman factors facing their shared cut."""
function _split_fishman_halves(Q::AbstractTensorMap, Q_next::AbstractTensorMap, alg::SplitProjector)
    half_alg = _split_half_svd_algorithm(alg)
    _, S, V = svd_trunc_no_error(Q, half_alg)
    U_next, S_next, _ = svd_trunc_no_error(Q_next, half_alg)
    Q = _split_fishman_sqrt(S) * V
    Q_next = U_next * _split_fishman_sqrt(S_next)
    return Q / norm(Q), Q_next / norm(Q_next)
end

"""Compute a split projector pair, optionally QR-compressing the environment halves for the yellow or red projectors."""
function compute_projector(
        halves, alg::SplitProjector;
        trunc::TruncationStrategy = alg.trunc, compress::Bool = false,
    )
    Q, Q_next = halves
    # Normalize each half before multiplication.
    Q = Q / norm(Q)
    Q_next = Q_next / norm(Q_next)
    if compress
        Q = _split_compress_half(Q, alg, Val(true))
        Q_next = _split_compress_half(Q_next, alg, Val(false))
    end

    # All four colors use Fishman factors before the final truncated SVD.
    Q, Q_next = _split_fishman_halves(Q, Q_next, alg)
    environment = Q * Q_next
    svd_alg = decomposition_algorithm(_set_decomposition_truncation(alg, trunc))
    U, S, V, truncation_error = svd_trunc!(environment / norm(environment), svd_alg)
    truncation_error = truncation_error / norm(S)

    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    P_left, P_right = contract_projectors(U, S, V, Q, Q_next)
    return (P_left, P_right), (; U, S, V, truncation_error)
end

"""Resolve fixed-space truncation for the green and teal projectors or the yellow and red projectors."""
function _split_truncation(trunc::TruncationStrategy, V::ElementarySpace)
    return trunc isa FixedSpaceTruncation ? truncspace(isdual(V) ? flip(V) : V) : trunc
end

"""Construct the half-environments for the green or yellow projectors for each row of a directional move."""
function _split_initial_halves(
        col::Int, network, env::SplitCTMRGEnv, ::Val{interlayer}
    ) where {interlayer}
    return dtmap(axes(env, 2)) do r
        return interlayer ? _split_yellow_environment((r, col), network, env) :
            _split_green_environment((r, col), network, env)
    end
end

"""Compute a green projector pair at one cut from prepared half-environments."""
function _split_green_projector(
        (r, col), halves, env::SplitCTMRGEnv, alg::SplitProjector, ::Val{E} = Val(false);
        trunc::TruncationStrategy = alg.trunc,
    ) where {E}
    top = halves[_prev(r, size(env, 2))][1]
    bottom = halves[r][2]
    if E
        top = permute(top, ((1, 4), (3, 2)))
        bottom = permute(bottom, ((1, 4), (3, 2)))
    end
    K = edge(env, :ket, E ? EAST : WEST, r, E ? col + 1 : col)
    V = E ? codomain(K)[1] : domain(K)[1]
    Q, Q_next = E ? (top, bottom) : (bottom, top)
    return compute_projector((Q, Q_next), alg; trunc = _split_truncation(trunc, V))
end

"""Compute a yellow projector pair at one cut using QR/LQ compression of prepared half-environments."""
function _split_yellow_projector(
        (r, col), halves, env::SplitCTMRGEnv, alg::SplitProjector, ::Val{E} = Val(false);
        trunc::TruncationStrategy = alg.trunc,
    ) where {E}
    top, bottom = halves[r]
    if E
        top = permute(top, ((1, 4, 5), (3, 2)))
        bottom = permute(bottom, ((1, 5), (4, 2, 3)))
    end
    K = edge(env, :ket, E ? EAST : WEST, r, E ? col + 1 : col)
    V = E ? domain(K)[1] : codomain(K)[1]
    Q, Q_next = E ? (top, bottom) : (bottom, top)
    return compute_projector((Q, Q_next), alg; trunc = _split_truncation(trunc, V), compress = true)
end

"""Collect green or yellow projector pairs for one side of a column from prepared half-environments."""
function _split_projectors_from_halves(
        col::Int, halves, env::SplitCTMRGEnv, alg::SplitProjector, trunc::TruncationStrategy,
        ::Val{interlayer}, east::Val,
    ) where {interlayer}
    results = dtmap(axes(env, 2)) do r
        return interlayer ? _split_yellow_projector((r, col), halves, env, alg, east; trunc) :
            _split_green_projector((r, col), halves, env, alg, east; trunc)
    end
    projectors, info = _split_proj_and_info(results)
    return projectors, info.truncation_error
end

"""Compute a teal projector pair from adjacent-row green-projected halves using the outer truncation strategy."""
function _split_teal_projector(
        (r, col), halves, env::SplitCTMRGEnv, alg::SplitProjector;
        trunc::TruncationStrategy = alg.trunc,
    )
    top = halves[_prev(r, size(env, 2))][1]
    bottom = halves[r][2]
    V = domain(edge(env, :ket, WEST, r, col))[1]
    trunc = _split_truncation(trunc, V)
    return compute_projector((bottom, top), alg; trunc)
end

"""Check whether the reduced environment spectrum for the red projectors can supply the requested largest singular values."""
function _split_reduce_red(quarters::NamedTuple, trunc::TruncationStrategy)
    trunc isa TruncationSpace && trunc.by === abs && trunc.rev || return false
    (; NW, NE, SE, SW) = quarters
    # Respect both the reduced quarter ranks and the original half-QR dimensions.
    return all(blocksectors(trunc.space)) do c
        return blockdim(trunc.space, c) <= min(
            blockdim(codomain(SE), c), blockdim(domain(SE), c),
            blockdim(domain(NE), c), blockdim(codomain(NE), c),
            blockdim(domain(SW), c), blockdim(codomain(NW), c),
        )
    end
end

"""Compute the red projector pair using quarter QR factors when its truncation fits the reduced spectrum."""
function _split_red_projector(
        (r, col), quarters::NamedTuple, env::SplitCTMRGEnv, alg::SplitProjector;
        trunc::TruncationStrategy = alg.trunc,
    )
    V = codomain(edge(env, :ket, WEST, r, col))[1]
    trunc = _split_truncation(trunc, V)
    (; NW, NE, SE, SW) = quarters
    reduce = ignore_derivatives() do
        _split_reduce_red(quarters, trunc)
    end
    if reduce
        # SE = Qb * Rb and NE = Lt * Qt with isometric Qb and Qt.
        # Their factors retain the norms and nonzero spectrum of (SE * SW) * (NW * NE).
        Rb = _split_compress_half(SE, alg, Val(true))
        Lt = _split_compress_half(NE, alg, Val(false))
        return compute_projector((Rb * SW, NW * Lt), alg; trunc)
    end
    # Preserve zero-spectrum dimensions and custom truncation ordering on the full halves.
    return compute_projector((SE * SW, NW * NE), alg; trunc, compress = true)
end

"""Construct the split projector pairs for a split-CTMRG left move."""
function sequential_projectors(
        col::Int, network, env::SplitCTMRGEnv, alg::SplitProjector, interlayer_trunc::TruncationStrategy
    )
    # Prepare each side once; a one-column cell shares the same periodic halves.
    single_column = size(env, 3) == 1
    green_halves = _split_initial_halves(col, network, env, Val(false))
    green, eg = _split_projectors_from_halves(col, green_halves, env, alg, alg.trunc, Val(false), Val(false))
    green_east_halves = single_column ? green_halves : _split_initial_halves(col + 1, network, env, Val(false))
    green_east, ege = _split_projectors_from_halves(col + 1, green_east_halves, env, alg, alg.trunc, Val(false), Val(true))
    yellow_halves = _split_initial_halves(col, network, env, Val(true))
    yellow, ey = _split_projectors_from_halves(col, yellow_halves, env, alg, interlayer_trunc, Val(true), Val(false))
    yellow_east_halves = single_column ? yellow_halves : _split_initial_halves(col + 1, network, env, Val(true))
    yellow_east, eye = _split_projectors_from_halves(col + 1, yellow_east_halves, env, alg, interlayer_trunc, Val(true), Val(true))

    # Teal refines the outer cut using environments with green projectors inserted.
    teal_halves = Base.inferencebarrier(dtmap)(r -> _split_teal_environment((r, col), network, env, green, green_east), axes(env, 2))
    teals = Base.inferencebarrier(dtmap)(r -> _split_teal_projector((r, col), teal_halves, env, alg), axes(env, 2))

    # Red refines the interlayer cut using environments with yellow projectors inserted.
    reds = Base.inferencebarrier(dtmap)(axes(env, 2)) do r
        quarters = _split_red_quarters((r, col), network, env, yellow, yellow_east)
        return _split_red_projector((r, col), quarters, env, alg; trunc = interlayer_trunc)
    end
    teal, teal_info = _split_proj_and_info(teals)
    red, red_info = _split_proj_and_info(reds)
    truncation_error = max(eg, ege, ey, eye, teal_info.truncation_error, red_info.truncation_error)
    return (; green, teal, yellow, red), (; truncation_error)
end

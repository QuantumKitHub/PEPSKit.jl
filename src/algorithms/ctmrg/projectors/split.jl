"""
$(TYPEDEF)

Projector algorithm for [`SplitCTMRG`](@ref), using the green, teal, yellow, and red environments of Naumann et al., Phys. Rev. B **111**, 235116 (2025).
Each projector pair is constructed from a truncated SVD of the product of its environment halves.
The yellow and red halves are QR-compressed before multiplication.

## Fields

$(TYPEDFIELDS)

## Constructors

    SplitProjector(; kwargs...)

Construct a split projector algorithm with `decomposition_alg`, `trunc`, and `verbosity`, following [`FullInfiniteProjector`](@ref).
`trunc` controls the outer cuts; [`SplitCTMRG`](@ref) supplies `interlayer_trunc` for the yellow and red cuts.
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

"""Compute a split projector pair, optionally QR-compressing the yellow or red environment halves."""
function compute_projector(
        halves, alg::SplitProjector;
        trunc::TruncationStrategy = alg.trunc, compress::Bool = false,
    )
    Q, Q_next = halves
    # Normalize each half before multiplication.
    Q = Q / norm(Q)
    Q_next = Q_next / norm(Q_next)
    if compress
        qr_alg = QRAdjoint()
        _, Q = left_orth(Q, qr_alg)
        _, Q_next = left_orth(Q_next', qr_alg)
        Q_next = Q_next'
    end

    # All four colors use an SVD of the product of their prepared halves.
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

"""Resolve fixed-space truncation at an outer or interlayer cut."""
function _split_truncation(trunc::TruncationStrategy, V::ElementarySpace)
    return trunc isa FixedSpaceTruncation ? truncspace(isdual(V) ? flip(V) : V) : trunc
end

"""Compute green or yellow projector pairs before constructing the teal or red environments."""
function _split_initial_projectors(
        col::Int, network, env::SplitCTMRGEnv, alg::SplitProjector, trunc::TruncationStrategy,
        ::Val{interlayer}, ::Val{east}
    ) where {interlayer, east}
    halves = dtmap(axes(env, 2)) do r
        return interlayer ? _split_yellow_environment((r, col), network, env) :
            _split_green_environment((r, col), network, env)
    end
    results = dtmap(axes(env, 2)) do r
        if interlayer
            top, bottom = halves[r]
            if east
                top = permute(top, ((1, 4, 5), (3, 2)))
                bottom = permute(bottom, ((1, 5), (4, 2, 3)))
            end
            Q, Q_next = east ? (top, bottom) : (bottom, top)
            K = edge(env, :ket, east ? EAST : WEST, r, east ? col + 1 : col)
            V = east ? domain(K)[1] : codomain(K)[1]
        else
            top = halves[_prev(r, size(env, 2))][1]
            bottom = halves[r][2]
            if east
                top = permute(top, ((1, 4), (3, 2)))
                bottom = permute(bottom, ((1, 4), (3, 2)))
            end
            Q, Q_next = east ? (top, bottom) : (bottom, top)
            K = edge(env, :ket, east ? EAST : WEST, r, east ? col + 1 : col)
            V = east ? codomain(K)[1] : domain(K)[1]
        end
        return compute_projector((Q, Q_next), alg; trunc = _split_truncation(trunc, V), compress = interlayer)
    end
    projectors, info = _split_proj_and_info(results)
    return projectors, info.truncation_error
end

"""Construct the green, teal, yellow, and red projector pairs for a split-CTMRG left move."""
function sequential_projectors(
        col::Int, network, env::SplitCTMRGEnv, alg::SplitProjector, interlayer_trunc::TruncationStrategy
    )
    # Initial projectors on both sides of the two-site refinement environments.
    green, eg = _split_initial_projectors(col, network, env, alg, alg.trunc, Val(false), Val(false))
    green_east, ege = _split_initial_projectors(col + 1, network, env, alg, alg.trunc, Val(false), Val(true))
    yellow, ey = _split_initial_projectors(col, network, env, alg, interlayer_trunc, Val(true), Val(false))
    yellow_east, eye = _split_initial_projectors(col + 1, network, env, alg, interlayer_trunc, Val(true), Val(true))

    # Teal refines the outer cut using environments with green projectors inserted.
    teal_halves = dtmap(r -> _split_teal_environment((r, col), network, env, green, green_east), axes(env, 2))
    teals = dtmap(axes(env, 2)) do r
        top = teal_halves[_prev(r, size(env, 2))][1]
        bottom = teal_halves[r][2]
        V = domain(edge(env, :ket, WEST, r, col))[1]
        trunc = _split_truncation(alg.trunc, V)
        return compute_projector((bottom, top), alg; trunc)
    end

    # Red refines the interlayer cut using environments with yellow projectors inserted.
    reds = dtmap(axes(env, 2)) do r
        top, bottom = _split_red_environment((r, col), network, env, yellow, yellow_east)
        V = codomain(edge(env, :ket, WEST, r, col))[1]
        trunc = _split_truncation(interlayer_trunc, V)
        return compute_projector((bottom, top), alg; trunc, compress = true)
    end
    teal, teal_info = _split_proj_and_info(teals)
    red, red_info = _split_proj_and_info(reds)
    truncation_error = max(eg, ege, ey, eye, teal_info.truncation_error, red_info.truncation_error)
    return (; green, teal, yellow, red), (; truncation_error)
end

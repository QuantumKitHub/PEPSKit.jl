"""
$(TYPEDEF)

Wrapper for an LQ decomposition algorithm with a defined reverse rule, used by `right_orth(!)`.
See [`QRAdjoint`](@ref) for details of the constructor options and defaults.
"""
struct LQAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end

LQAdjoint(; kwargs...) = LQAdjoint(_orthogonalization_algorithms(; kwargs...)...)

"""Compute an LQ decomposition using the forward algorithm and reverse rule configured by `LQAdjoint`."""
MatrixAlgebraKit.right_orth(t, alg::LQAdjoint) = right_orth!(copy(t), alg)
"""Factorize the input in place using the LQ forward algorithm configured by `LQAdjoint`."""
MatrixAlgebraKit.right_orth!(t, alg::LQAdjoint) = right_orth!(t, alg.fwd_alg)

# right_orth! rrule wrapping MatrixAlgebraKit's lq_pullback!
function ChainRulesCore.rrule(
        ::typeof(right_orth!),
        t::AbstractTensorMap,
        alg::LQAdjoint{F, R},
    ) where {F <: MatrixAlgebraKit.Algorithm, R <: FullPullback}
    LQ = right_orth(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    """Propagate LQ cotangents with the configured gauge-sensitivity tolerance."""
    function right_orth!_pullback(ΔLQ)
        Δt = zeros(scalartype(t), space(t))
        MatrixAlgebraKit.lq_pullback!(Δt, t, LQ, unthunk.(ΔLQ); gauge_atol = gtol(ΔLQ))
        return NoTangent(), Δt, NoTangent()
    end
    function right_orth!_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return LQ, right_orth!_pullback
end

module PEPSKitMooncakeExt

using PEPSKit, TensorKit, Mooncake, MatrixAlgebraKit
using PEPSKit: SVDAdjoint
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, primal, rrule!!, arrayify, @is_primitive

_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

Mooncake.tangent_type(::Type{<:PEPSKit.SVDAdjoint}) = Mooncake.NoTangent

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc), TensorKit.AbstractTensorMap, SVDAdjoint}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.svd_trunc)}, t_dt::CoDual{<:TensorKit.AbstractTensorMap}, alg_dalg::CoDual{SVDAdjoint{F, R}}) where {F, R <: PEPSKit.FullPullback}
    # TODO: filter out any decomposition algorithm that doesn't give access to the full spectrum
    t, Δt = arrayify(t_dt)
    alg = primal(alg_dalg)
    # requires access to the full decomposition
    U, S, V⁺ = svd_compact!(t, alg.fwd_alg.alg)
    (Ũ, S̃, Ṽ⁺), inds = MatrixAlgebraKit.truncate(svd_trunc!, (U, S, V⁺), alg.fwd_alg.trunc)
    truncerror = MatrixAlgebraKit.truncation_error(diagview(S), inds)

    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    output = (Ũ, S̃, Ṽ⁺, truncerror)
    USVᴴtrunc = (Ũ, S̃, Ṽ⁺)
    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
    ΔUSVᴴtrunc = last.(arrayify.(USVᴴtrunc, Base.front(Mooncake.tangent(output_codual))))
    function svd_trunc!_full_pullback((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
        _warn_pullback_truncerror(dϵ)
        Δt = MatrixAlgebraKit.svd_pullback!(
            Δt, t, (U, S, V⁺), ΔUSVᴴtrunc, inds;
            gauge_atol = gtol(ΔUSVᴴtrunc), degeneracy_atol = alg.rrule_alg.degeneracy_atol,
        )
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc!_full_pullback
end

function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.svd_trunc)}, t_dt::CoDual{<:TensorKit.AbstractTensorMap}, alg_dalg::CoDual{SVDAdjoint{F, R}}) where {F, R <: PEPSKit.TruncPullback}
    t, Δt = arrayify(t_dt)
    alg = primal(alg_dalg)
    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    output = svd_trunc(t, alg)

    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))
    function svd_trunc!_trunc_pullback((_, _, _, dϵ)::Tuple{NoRData, NoRData, NoRData, Real})
        Utrunc, Strunc, Vᴴtrunc, ϵ = Mooncake.primal(output_codual)
        dUtrunc_, dStrunc_, dVᴴtrunc_, _ = Mooncake.tangent(output_codual)
        _warn_pullback_truncerror(dϵ)
        U, dU = arrayify(Utrunc, dUtrunc_)
        S, dS = arrayify(Strunc, dStrunc_)
        Vᴴ, dVᴴ = arrayify(Vᴴtrunc, dVᴴtrunc_)
        MatrixAlgebraKit.svd_trunc_pullback!(Δt, t, (U, S, Vᴴ), (dU, dS, dVᴴ))
        MatrixAlgebraKit.zero!(dU)
        MatrixAlgebraKit.zero!(dS)
        MatrixAlgebraKit.zero!(dVᴴ)
        return NoRData(), NoRData(), NoRData()
    end
    return output_codual, svd_trunc!_trunc_pullback
end

end

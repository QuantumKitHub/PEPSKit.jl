module PEPSKitMooncakeExt

using PEPSKit, TensorKit, Mooncake, MatrixAlgebraKit
using PEPSKit: SVDAdjoint, EighAdjoint, QRAdjoint
using Mooncake: DefaultCtx, CoDual, Dual, NoRData, primal, rrule!!, arrayify, @is_primitive

_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

Mooncake.tangent_type(::Type{<:PEPSKit.SVDAdjoint}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.EighAdjoint}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.QRAdjoint}) = Mooncake.NoTangent

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

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(eigh_trunc), TensorKit.AbstractTensorMap, EighAdjoint}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_trunc)}, t_dt::CoDual{<:TensorKit.AbstractTensorMap}, alg_dalg::CoDual{EighAdjoint{F, R}}) where {F, R <: PEPSKit.FullPullback}
    t, dt = arrayify(t_dt)
    alg = primal(alg_dalg)
    
    D, V = eigh_full!(t; alg.fwd_alg.alg)
    (D̃, Ṽ), inds = MatrixAlgebraKit.truncate(eigh_trunc!, (D, V), alg.fwd_alg.trunc)
    ϵ = MatrixAlgebraKit.truncation_error(diagview(D), inds)
    
    DVtrunc = (D̃, Ṽ)
    # pack output
    DVtrunc_dDVtrunc = Mooncake.zero_fcodual((DVtrunc..., ϵ))

    # define pullback
    dDVtrunc = last.(arrayify.(DVtrunc, Base.front(Mooncake.tangent(DVtrunc_dDVtrunc))))

    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    function eigh_trunc!_full_pullback((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
        _warn_pullback_truncerror(dϵ)
        MatrixAlgebraKit.eigh_pullback!(dt, t, (D, V), dDVtrunc, inds; gauge_atol = gtol(dDVtrunc), degeneracy_atol = alg.rrule_alg.degeneracy_atol)
        MatrixAlgebraKit.zero!.(dDVtrunc) # since this is allocated in this function this is probably not required
        return ntuple(Returns(NoRData()), 3)
    end
    return DVtrunc_dDVtrunc, eigh_trunc!_full_pullback
end

function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.eigh_trunc)}, t_dt::CoDual{<:TensorKit.AbstractTensorMap}, alg_dalg::CoDual{EighAdjoint{F, R}}) where {F, R <: PEPSKit.TruncPullback}
    t, dt = arrayify(t_dt)
    alg = primal(alg_dalg)
    
    D, V, truncerror = eigh_trunc(t, alg)
    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    output = (D, V, truncerror)
    output_codual = CoDual(output, Mooncake.fdata(Mooncake.zero_tangent(output)))

    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)
    function eigh_trunc!_trunc_pullback((_, _, dϵ)::Tuple{NoRData, NoRData, Real})
        _warn_pullback_truncerror(dϵ)
        Dtrunc, Vtrunc, ϵ = Mooncake.primal(output_codual)
        dDtrunc_, dVtrunc_, dϵ = Mooncake.tangent(output_codual)
        D, dD = arrayify(Dtrunc, dDtrunc_)
        V, dV = arrayify(Vtrunc, dVtrunc_)
        MatrixAlgebraKit.eigh_trunc_pullback!(dt, t, (D, V), (dD, dV); gauge_atol = gtol((dD, dV)), degeneracy_atol = alg.rrule_alg.degeneracy_atol)
        MatrixAlgebraKit.zero!(dD) # since this is allocated in this function this is probably not required
        MatrixAlgebraKit.zero!(dV) # since this is allocated in this function this is probably not required
        return ntuple(Returns(NoRData()), 3)
    end
    return output_codual, eigh_trunc!_trunc_pullback
end

@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(left_orth), TensorKit.AbstractTensorMap, QRAdjoint}
function Mooncake.rrule!!(::CoDual{typeof(MatrixAlgebraKit.left_orth)}, t_dt::CoDual{<:TensorKit.AbstractTensorMap}, alg_dalg::CoDual{QRAdjoint})
    t, dt = arrayify(t_dt)
    alg = primal(alg_dalg)

    QR = left_orth(t, alg)
    gtol = PEPSKit._get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    output_codual = Mooncake.zero_fcodual(QR)
    dQ_, dR_ = Mooncake.tangent(output_codual)
    Q, dQ = arrayify(Q, dQ_)
    R, dR = arrayify(R, dR_)
    function left_orth_pullback(::NoRData)
        MatrixAlgebraKit.qr_pullback!(dt, t, QR, (dQ, dR); gauge_atol = gtol(dQR))
        return ntuple(Returns(NoRData()), 3)
    end
    return output_codual, left_orth_pullback
end

end

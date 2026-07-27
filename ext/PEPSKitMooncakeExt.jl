module PEPSKitMooncakeExt

using PEPSKit, MPSKit, TensorKit, Mooncake, MatrixAlgebraKit
using PEPSKit: SVDAdjoint, EighAdjoint, QRAdjoint, CTMRGAlgorithm, FixedPointGradient, sdiag_pow, eachcoordinate
import PEPSKit: real_inner
using Mooncake: DefaultCtx, MinimalCtx, CoDual, Dual, NoRData, primal, tangent, rrule!!, arrayify, @is_primitive

function Mooncake.arrayify(ψ::PEPSKit.InfinitePEPS{T}, dψ) where {T}
    Δψmat = map((a, da) -> Mooncake.arrayify(a, da)[2], ψ.A, dψ.fields.A)
    Δψ = PEPSKit.InfinitePEPS{T}(Δψmat)
    return ψ, Δψ
end

_warn_pullback_truncerror(dϵ::Real; tol = MatrixAlgebraKit.defaulttol(dϵ)) =
    abs(dϵ) ≤ tol || @warn "Pullback ignores non-zero tangents for truncation error"

Mooncake.tangent_type(::Type{<:PEPSKit.SVDAdjoint}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.EighAdjoint}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.QRAdjoint}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.CTMRGAlgorithm}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{<:PEPSKit.FixedPointGradient}) = Mooncake.NoTangent

Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit.eachcoordinate), Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit.eachcoordinate), Any, Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit._next_coordinate), Int, Int}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit._set_decomposition_truncation), Any, Any}
Mooncake.@zero_derivative Mooncake.MinimalCtx Tuple{typeof(PEPSKit.CTMRGEnv), Union{PEPSKit.InfinitePartitionFunction, PEPSKit.InfinitePEPS}, Vararg}

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(svd_trunc), TensorKit.AbstractTensorMap, SVDAdjoint}
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
        return NoRData(), NoRData(), NoRData(), zero(dϵ)
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

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(eigh_trunc), TensorKit.AbstractTensorMap, EighAdjoint}
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

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(left_orth), TensorKit.AbstractTensorMap, QRAdjoint}
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

PEPSKit.real_inner(_, η₁::Mooncake.Tangent, η₂::Mooncake.Tangent) = Mooncake._dot(η₁, η₂)

# Follows the `map` rrule from ChainRules.jl but specified for the case of one AbstractArray that is being mapped
# https://github.com/JuliaDiff/ChainRules.jl/blob/e245d50a1ae56ce46fc8c1f0fe9b925964f1146e/src/rulesets/Base/base.jl#L243
@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof(PEPSKit.dtmap), Any, AbstractArray}
function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual{<:NamedTuple{(:scheduler,), Tuple{R}}}, ::CoDual{typeof(PEPSKit.dtmap)}, f_df::CoDual, A_dA::CoDual{<:AbstractArray}) where {R}
    scheduler = get(Mooncake.primal(kw), :scheduler, PEPSKit.Defaults.scheduler[])
    f = Mooncake.primal(f_df)
    A, ΔA = Mooncake.arrayify(A_dA)
    el_rrules = tmap(A; scheduler) do a
        cache = Mooncake.prepare_pullback_cache(f, a)
        return Mooncake.value_and_pullback!!(cache, f, a)
    end
    y = map(first, el_rrules)
    y_dy = Mooncake.zero_fcodual(y)
    Δys = Mooncake.arrayify(y_dy)[2]
    function dtmap_pullback(::NoRData)
        backevals = tmap(el_rrules, Δys; scheduler) do el_rrule, Δy
            last(el_rrule)(Δy)
        end
        ΔA .= map(last, backevals)
        return ntuple(Returns(NoRData()), 5)
    end
    return y_dy, dtmap_pullback
end

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof(PEPSKit.dtmap!!), Any, AbstractArray, AbstractArray}
function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual, ::CoDual{typeof(PEPSKit.dtmap!!)}, f_df::CoDual, C_dC::CoDual{<:AbstractArray}, A_dA::CoDual{<:AbstractArray})
    C, dtmap_pullback = rrule(config, dtmap, f, A; kwargs...)
    function dtmap!!_pullback(dy)
        dtmap, df, dA = dtmap_pullback(dy)
        return dtmap, df, NoTangent, dA
    end
    return C_dC, dtmap!!_pullback
end

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof(PEPSKit.sdiag_pow), AbstractTensorMap, Real}
function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual{<:NamedTuple{(:tol,), Tuple{<:Real}}}, ::CoDual{typeof(PEPSKit.sdiag_pow)}, s_ds::CoDual{<:AbstractTensorMap}, p_dp::CoDual{<:Real})
    s, Δs = arrayify(s_ds)
    tol = get(primal(kw), :tol, eps(real(TensorKit.scalartype(s)))^(3 / 4))
    tol *= norm(s, Inf)
    pow = primal(p_df)
    spow = sdiag_pow(s, pow; tol)
    spow_minus1_conj = scale!(sdiag_pow(s', pow - 1; tol), pow)
    spow_dspow = Mooncake.zero_fcodual(spow)
    spow, Δspow = arrayify(spow_dspow)
    function sdiag_pow_pullback(::NoRData)
        PEPSKit._elementwise_mult(Δs, spow_minus1_conj)
        return NoRData(), NoRData(), NoRData(), NoRData(), zero(pow)
    end
    return spow_dspow, sdiag_pow_pullback
end

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(PEPSKit.sdiag_pow), AbstractTensorMap, Real}
function Mooncake.rrule!!(::CoDual{typeof(PEPSKit.sdiag_pow)}, s_ds::CoDual{<:AbstractTensorMap}, p_dp::CoDual{<:Real})
    s, Δs = arrayify(s_ds)
    tol = eps(real(TensorKit.scalartype(s)))^(3 / 4)
    tol *= norm(s, Inf)
    pow = primal(p_dp)
    spow = sdiag_pow(s, pow; tol)
    spow_minus1_conj = scale!(sdiag_pow(s', pow - 1; tol), pow)
    spow_dspow = Mooncake.zero_fcodual(spow)
    spow, Δspow = arrayify(spow_dspow)
    function sdiag_pow_pullback(::NoRData)
        PEPSKit._elementwise_mult(Δs, spow_minus1_conj)
        return NoRData(), NoRData(), zero(pow)
    end
    return spow_dspow, sdiag_pow_pullback
end

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(PEPSKit.CTMRGEnv), Any, Any}
function Mooncake.rrule!!(::CoDual{typeof(PEPSKit.CTMRGEnv)}, c_dc::CoDual{Array{C, 3}}, e_de::CoDual{Array{T, 3}}) where {C, T}
    corners, dcorners = arrayify(c_dc)
    edges, dedges = arrayify(e_de)
    env = CTMRGEnv(corners, edges)
    denv = CTMRGEnv(dcorners, dedges)
    ctmrgenv_pullback(::NoRData) = NoRData(), env.corners, env.edges
    return Mooncake.CoDual(env, denv), ctmrgenv_pullback
end

Mooncake.tangent_type(::Type{NamedTuple{(:converged, :convergence_error, :contraction_metrics), Tuple{Bool, Float64, NamedTuple{(:truncation_error,), Tuple{Float64}}}}}) = Mooncake.NoTangent
Mooncake.tangent_type(::Type{NamedTuple{(:alg_rrule,), Tuple{NamedTuple{(:solver_alg,), Tuple{NamedTuple{(:orth, :krylovdim, :maxiter, :tol, :eager, :verbosity), Tuple{O, Int, Int, Float64, Bool, Int}}}}}}}) where {O} = Mooncake.NoTangent

@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(Core.kwcall), NamedTuple, typeof(PEPSKit.hook_pullback), Any, Vararg{Any}}
function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual{<:NamedTuple{(:alg_rrule,), Tuple{R}}}, hpb::CoDual{typeof(PEPSKit.hook_pullback)}, f_df::CoDual, args_dargs::CoDual...) where {R}
    alg_rrule = Mooncake.primal(kw)[:alg_rrule]
    y, f_pullback = PEPSKit._rrule(alg_rrule, Mooncake.primal(f_df), Mooncake.primal.(args_dargs)...)
    hook_pullback_pullback(Δ) = (NoRData(), f_pullback(Δ)...)
    return y, hook_pullback_pullback
end

# compute the CTMRG gradient through fixed-point differentiation
@is_primitive Mooncake.MinimalCtx Mooncake.ReverseMode Tuple{typeof(MPSKit.leading_boundary), Any, Any, CTMRGAlgorithm}
function Mooncake.rrule!!(::CoDual{typeof(MPSKit.leading_boundary)}, envinit_denvinit::CoDual, state_dstate::CoDual, alg_dalg::CoDual{<:CTMRGAlgorithm})
    alg = Mooncake.primal(alg_dalg)
    state = Mooncake.primal(state_dstate)
    envinit = Mooncake.primal(envinit_denvinit)
    #PEPSKit._check_algorithm_combination(alg, gradmode)

    env, = MPSKit.leading_boundary(envinit, state, alg)

    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = PEPSKit._set_fixed_truncation(alg) # fix spaces during differentiation
    alg_gauge = PEPSKit._scrambling_env_gauge(alg) # select appropriate gauge-fixing algorithm
    env_conv, info = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(state), env, alg_fixed)
    signs, corner_phases, edge_phases = PEPSKit.compute_gauge_fix_gauge(env_conv, env, alg_gauge)
    # prepare its pullback
    #sig = Tuple{typeof(gauge_fixed_iteration), typeof(state), typeof(env), typeof(alg_fixed), typeof(signs), typeof(corner_phases), typeof(edge_phases)}
    #rule = Mooncake.build_rrule(gauge_fixed_iteration, state, env, alg_fixed, signs, corner_phases, edge_phases)
    #_, env_vjp = Mooncake.value_and_gradient!!(rule, gauge_fixed_iteration, state, env, alg_fixed, signs, corner_phases, edge_phases)
    #out, env_vjp = Mooncake.rrule!!(CoDual(gauge_fixed_iteration, Mooncake.NoFData()), Mooncake.zero_fcodual(state), Mooncake.zero_fcodual(env))
    cache = Mooncake.prepare_pullback_cache(PEPSKit.gauge_fixed_iteration, state, env, alg_fixed, signs, corner_phases, edge_phases)
    _, env_vjp = Mooncake.value_and_pullback!!(cache, env_conv, PEPSKit.gauge_fixed_iteration, state, env, alg_fixed, signs, corner_phases, edge_phases)
    # split off state and environment parts
    ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
    ∂f∂x(x)::typeof(env) = env_vjp(x)[3]

    output_doutput = Mooncake.zero_fcodual((env, info))
    denv = Mooncake.tangent(output_doutput)[1]
    env, Δenv = Mooncake.arrayify(env, denv)
    function leading_boundary_fixed_pullback(::NoRData)
        # evaluate the geometric sum
        ∂F∂env = PEPSKit.fixedpoint_gradient(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode.solver_alg)
        return ntuple(Returns(NoRData()), 4)
    end
    return (env, invo), leading_boundary_fixed_pullback
end

end

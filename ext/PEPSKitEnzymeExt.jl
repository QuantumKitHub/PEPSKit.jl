module PEPSKitEnzymeExt

using PEPSKit, MPSKit, TensorKit, MatrixAlgebraKit
using PEPSKit: SVDAdjoint, EighAdjoint, QRAdjoint, CTMRGAlgorithm, FixedPointGradient, sdiag_pow, dtmap, dtmap!!
using PEPSKit: InfiniteSquareNetwork, InfinitePEPS, InfinitePEPO, _stack_tuples
using PEPSKit: unitcell, ket, bra, pepo
using PEPSKit: _periodic_getindex_dispatch
using PEPSKit: _split_corners_edges
using TensorKit: AbstractTensorMap
using ChainRulesCore: ignore_derivatives
using VectorInterface: add!, One
import PEPSKit: real_inner
using Enzyme
using Enzyme.EnzymeCore: EnzymeRules

@inline EnzymeRules.inactive_type(::Type{<:SVDAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{<:QRAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{<:EighAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{<:CTMRGAlgorithm}) = true
@inline EnzymeRules.inactive_type(::Type{<:PEPSKit.GradientAlgorithm}) = true

@inline EnzymeRules.inactive(::typeof(PEPSKit.checklattice), args...) = nothing
@inline EnzymeRules.inactive(::typeof(ignore_derivatives), args...) = nothing
@inline EnzymeRules.inactive(::typeof(PEPSKit.eachcoordinate), args...) = nothing

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(MatrixAlgebraKit.svd_trunc_no_error)},
        ::Type{RT},
        t::Annotation,
        alg::Const{<:SVDAdjoint{F, R}}
    ) where {RT, F, R <: PEPSKit.FullPullback}
    # requires access to the full decomposition
    U, S, V⁺ = svd_compact(t.val, alg.val.fwd_alg.alg)
    (Ũ, S̃, Ṽ⁺), inds = MatrixAlgebraKit.truncate(svd_trunc!, (U, S, V⁺), alg.val.fwd_alg.trunc)
    truncerror = MatrixAlgebraKit.truncation_error(diagview(S), inds)

    output = (Ũ, S̃, Ṽ⁺, truncerror)
    USVᴴtrunc = (Ũ, S̃, Ṽ⁺)
    primal = EnzymeRules.needs_primal(config) ? USVᴴtrunc : nothing
    dret = if EnzymeRules.needs_shadow(config)
        (zero(USVᴴtrunc[1]), zero(USVᴴtrunc[2]), zero(USVᴴtrunc[3]))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, dret, (dret, (U, S, V⁺), inds))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        func::Const{typeof(MatrixAlgebraKit.svd_trunc_no_error)},
        ::Type{RT},
        cache,
        t::Annotation,
        alg::Const{<:SVDAdjoint{F, R}}
    ) where {RT, F, R <: PEPSKit.FullPullback}
    dUSVᴴtrunc, USV⁺, ind = cache
    U, S, V⁺ = USV⁺
    gtol = PEPSKit._get_pullback_gauge_tol(alg.val.rrule_alg.verbosity)
    if !isa(t, Const)
        MatrixAlgebraKit.svd_pullback!(
            t.dval, t.val, (U, S, V⁺), dUSVᴴtrunc, ind;
            gauge_atol = gtol(dUSVᴴtrunc), degeneracy_atol = alg.val.rrule_alg.degeneracy_atol,
        )
    end
    return ntuple(Returns(nothing), 2)
end

"""
Shared implementation of the CTMRG fixed-point gradient.

Both the bare `leading_boundary` rule and the `hook_pullback` rule below need the
same augmented-forward work; they differ only in where the solver algorithm comes
from. Keeping one implementation means the two cannot drift apart.
"""
function _leading_boundary_augmented(config, envinit::Annotation, state::Annotation, alg::Const)
    env, info = MPSKit.leading_boundary(envinit.val, state.val, alg.val)
    alg_fixed = PEPSKit._set_fixed_truncation(alg.val)
    alg_gauge = PEPSKit._scrambling_env_gauge(alg.val)
    env_conv, _ = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(state.val), env, alg_fixed)
    shadow = EnzymeRules.needs_shadow(config) ? Enzyme.make_zero((env, info)) : nothing
    denv = isnothing(shadow) ? nothing : shadow[1]
    primal = EnzymeRules.needs_primal(config) ? (env, info) : nothing
    signs, corner_phases, edge_phases = PEPSKit.compute_gauge_fix_gauge(
        env_conv, env, alg_gauge,
    )
    cache = (env, denv, alg_fixed, signs, corner_phases, edge_phases)
    return primal, shadow, cache
end

function _leading_boundary_reverse!(config, cache, state::Annotation, solver_alg)
    env, denv, alg_fixed, signs, corner_phases, edge_phases = cache
    function gauge_fixed_iteration(A, x)
        x′ = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1]
        return PEPSKit.fix_phases(x′, signs, corner_phases, edge_phases)
    end
    inner_mode = Enzyme.set_runtime_activity(ReverseSplitWithPrimal, config)
    fwd, rev = Enzyme.autodiff_thunk(inner_mode, Const{typeof(gauge_fixed_iteration)}, Duplicated, typeof(state), Duplicated{typeof(env)})
    # NOTE: the vjp MUST NOT touch the caller's shadows.  `denv` is the incoming
    # cotangent and is simultaneously `∂E∂x` for the fixed-point solve, and
    # `state.dval` already holds the gradient contributions accumulated by the
    # rest of the reverse sweep. Zeroing either silently destroys the whole gradient.
    # Instead we allocate fresh shadows per evaluation. The Krylov solver also retains
    # the returned vectors, so they MUST NOT alias a buffer we reuse.
    function vjp(Δ)
        dstate = Enzyme.make_zero(state.val)
        denv_scratch = Enzyme.make_zero(env)
        state_dup = Duplicated(state.val, dstate)
        env_dup = Duplicated(env, denv_scratch)

        # Enzyme's split-mode tape is single-use, and this vjp is called once per
        # Krylov iteration, so build a fresh one for each evaluation.
        tape, _, out_shadow = fwd(Const(gauge_fixed_iteration), state_dup, env_dup)
        add!(out_shadow, Δ, One(), One())
        rev(Const(gauge_fixed_iteration), state_dup, env_dup, tape)
        return dstate, denv_scratch
    end
    ∂f∂A(x)::typeof(state.val) = vjp(x)[1]
    ∂f∂x(x)::typeof(env) = vjp(x)[2]
    ∂A = PEPSKit.fixedpoint_gradient(denv, ∂f∂x, ∂f∂A, denv, solver_alg)
    if !isa(state, Const)
        add!(state.dval, ∂A, One(), One())
    end
    return nothing
end

"""
The solver algorithm `hook_pullback` was asked for.

`alg_rrule = nothing` selects naive AD through the CTMRG iterations in the
ChainRules path; there is no Enzyme equivalent yet, so it falls back to the
default fixed-point solver, which is what this rule did unconditionally before.
"""
@inline function _fixedpoint_solver_alg(kw::NamedTuple)
    gradmode = get(kw, :alg_rrule, nothing)
    isnothing(gradmode) && return PEPSKit.FixedPointGradient().solver_alg
    return gradmode.solver_alg
end

const _LeadingBoundary = typeof(MPSKit.leading_boundary)

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        kw::Const{<:NamedTuple},
        ::Const{typeof(PEPSKit.hook_pullback)},
        ::Const{_LeadingBoundary},
        envinit::Annotation,
        state::Annotation,
        alg::Const{<:CTMRGAlgorithm},
    ) where {RT}
    primal, shadow, cache = _leading_boundary_augmented(config, envinit, state, alg)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        cache,
        kw::Const{<:NamedTuple},
        ::Const{typeof(PEPSKit.hook_pullback)},
        ::Const{_LeadingBoundary},
        envinit::Annotation,
        state::Annotation,
        alg::Const{<:CTMRGAlgorithm},
    ) where {RT}
    _leading_boundary_reverse!(config, cache, state, _fixedpoint_solver_alg(kw.val))
    return ntuple(Returns(nothing), 6)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(MPSKit.leading_boundary)},
        ::Type{RT},
        envinit::Annotation,
        state::Annotation,
        alg::Const{<:CTMRGAlgorithm}
    ) where {RT}
    primal, shadow, cache = _leading_boundary_augmented(config, envinit, state, alg)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(MPSKit.leading_boundary)},
        ::Type{RT},
        cache,
        envinit::Annotation,
        state::Annotation,
        alg::Const{<:CTMRGAlgorithm}
    ) where {RT}
    _leading_boundary_reverse!(config, cache, state, PEPSKit.FixedPointGradient().solver_alg)
    return ntuple(Returns(nothing), 3)
end

@inline _dtmap_elem(src::Const, i) = Const(src.val[i])
@inline _dtmap_elem(src::Annotation, i) = Duplicated(src.val[i], src.dval[i])

for pb in (:svd_pullback!, :eig_pullback!, :eigh_pullback!)
    @eval function MatrixAlgebraKit.$pb(
            Δt::AbstractTensorMap, ::Nothing, F, ΔF,
            inds = TensorKit.SectorDict(c => Colon() for c in TensorKit.blocksectors(Δt));
            kwargs...
        )
        for (c, Δb) in TensorKit.blocks(Δt)
            haskey(inds, c) || continue
            Fc = TensorKit.block.(F, Ref(c))
            ΔFc = TensorKit.block.(ΔF, Ref(c))
            MatrixAlgebraKit.$pb(Δb, nothing, Fc, ΔFc, inds[c]; kwargs...)
        end
        return Δt
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(_split_corners_edges)},
        ::Type{RT},
        ce::Annotation{<:AbstractArray},
    ) where {RT}
    primal_val = (map(first, ce.val), map(last, ce.val))
    primal = EnzymeRules.needs_primal(config) ? primal_val : nothing
    shadow = if EnzymeRules.needs_shadow(config) && !isa(ce, Const)
        (map(first, ce.dval), map(last, ce.dval))
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(_split_corners_edges)},
        ::Type{RT},
        cache,
        ce::Annotation{<:AbstractArray},
    ) where {RT}
    return (nothing,)
end

"""
How Enzyme should carry the mapped function's return value.

The CTMRG routines map to tensors, which travel as `Duplicated`.  `expectation_value`
maps to scalars, and a scalar is immutable: asking for `Duplicated` there hands back a
`Base.RefValue` that the destination array cannot store.  Scalars must be `Active`,
which also changes the reverse call -- the cotangent is passed in rather than
accumulated into a shadow.
"""
@inline _dtmap_retann(::Type{T}) where {T} = Duplicated{T}
@inline _dtmap_retann(::Type{T}) where {T <: Number} = Active{T}

function _dtmap_augmented!(config, f::FA, dst, src) where {FA <: Annotation}
    ET = eltype(src.val)
    DT = eltype(dst.val)
    SA = src isa Const ? Const{ET} : Duplicated{ET}
    # Propagate the caller's runtime-activity setting into the nested thunk.
    # Without this the inner differentiation runs with static activity while the
    # outer one does not, and derivative contributions are silently dropped.
    mode = Enzyme.set_runtime_activity(ReverseSplitWithPrimal, config)
    fwd, rev = Enzyme.autodiff_thunk(mode, FA, _dtmap_retann(DT), SA)

    inds = collect(eachindex(src.val))
    tapes = Vector{Any}(undef, length(inds))
    elems = Vector{Any}(undef, length(inds))
    for (k, i) in enumerate(inds)
        arg = _dtmap_elem(src, i)
        tape, primal, shadow = fwd(f, arg)
        dst.val[i] = primal
        # an `Active` return has no shadow to store; its cotangent is seeded in reverse
        if !isa(dst, Const) && !(DT <: Number)
            dst.dval[i] = shadow
        end
        tapes[k] = tape
        elems[k] = arg
    end
    return (rev, inds, tapes, elems)
end

function _dtmap_reverse!(f::FA, dst, cache) where {FA <: Annotation}
    rev, inds, tapes, elems = cache
    DT = eltype(dst.val)
    for k in eachindex(tapes)
        if DT <: Number
            seed = isa(dst, Const) ? zero(DT) : dst.dval[inds[k]]
            rev(f, elems[k], seed, tapes[k])
        else
            rev(f, elems[k], tapes[k])
        end
    end
    return nothing
end

#= turn this off until the needsReRouting fix is merged at Enzyme
function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(dtmap!!)},
        ::Type{RT},
        f::FA,
        dst::Annotation{<:AbstractArray},
        src::Annotation{<:AbstractArray},
    ) where {RT, FA <: Annotation}
    cache = _dtmap_augmented!(config, f, dst, src)
    primal = EnzymeRules.needs_primal(config) ? dst.val : nothing
    shadow = if EnzymeRules.needs_shadow(config) && !isa(dst, Const)
        dst.dval
    else
        nothing
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(dtmap!!)},
        ::Type{RT},
        cache,
        f::FA,
        dst::Annotation{<:AbstractArray},
        src::Annotation{<:AbstractArray},
    ) where {RT, FA <: Annotation}
    _dtmap_reverse!(f, dst, cache)
    return (nothing, nothing, nothing)
end

=#

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{Type{InfiniteSquareNetwork}},
        ::Type{RT},
        top::Annotation{<:InfinitePEPS},
        mid::Annotation{<:InfinitePEPO},
        bot::Annotation{<:InfinitePEPS},
    ) where {RT}
    netw = InfiniteSquareNetwork(top.val, mid.val, bot.val)
    primal = EnzymeRules.needs_primal(config) ? netw : nothing
    shadow = EnzymeRules.needs_shadow(config) ? Enzyme.make_zero(netw) : nothing
    return EnzymeRules.augmented_rule_return_type(config, RT)(primal, shadow, shadow)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{Type{InfiniteSquareNetwork}},
        ::Type{RT},
        cache,
        top::Annotation{<:InfinitePEPS},
        mid::Annotation{<:InfinitePEPO},
        bot::Annotation{<:InfinitePEPS},
    ) where {RT}
    Δnetwork = cache
    aliased = !isa(top, Const) && !isa(bot, Const) && (top.dval === bot.dval)
    w = aliased ? 0.5 : 1.0
    !isa(top, Const) && add!(top.dval, InfinitePEPS(map(ket, unitcell(Δnetwork))), w, One())
    !isa(bot, Const) && add!(bot.dval, InfinitePEPS(map(bra, unitcell(Δnetwork))), w, One())
    !isa(mid, Const) && add!(mid.dval, InfinitePEPO(_stack_tuples(map(pepo, unitcell(Δnetwork)))), One(), One())
    return (nothing, nothing, nothing)
end


const _PeriodicElt = Union{AbstractTensorMap, Tuple{Vararg{AbstractTensorMap}}}

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(_periodic_getindex_dispatch)},
        ::Type{RT},
        A::Annotation,
        data::Annotation{<:AbstractArray{<:_PeriodicElt}},
        J::Annotation,
    ) where {RT}
    primal = _periodic_getindex_dispatch(A.val, data.val, J.val)
    shadow = if EnzymeRules.needs_shadow(config)
        # `data.dval === data.val` means Enzyme shared the container between
        # primal and shadow because it considers it inactive. Accumulating into
        # it would write the primal, so hand back a throwaway zero instead.
        if isa(data, Const) || data.dval === data.val
            Enzyme.make_zero(primal)
        else
            _periodic_getindex_dispatch(A.val, data.dval, J.val)
        end
    else
        nothing
    end
    p = EnzymeRules.needs_primal(config) ? primal : nothing
    return EnzymeRules.AugmentedReturn(p, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(_periodic_getindex_dispatch)},
        ::Type{RT},
        cache,
        A::Annotation,
        data::Annotation{<:AbstractArray{<:_PeriodicElt}},
        J::Annotation,
    ) where {RT}
    return (nothing, nothing, nothing)
end

end

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

@inline EnzymeRules.inactive(::typeof(PEPSKit.checklattice), args...) = nothing
@inline EnzymeRules.inactive(::typeof(ignore_derivatives), args...) = nothing

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

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        kw::Const{<:NamedTuple},
        ::Const{typeof(PEPSKit.hook_pullback)},
        f::Const,
        args::Annotation...
    ) where {RT}
    alg_rrule = get(kw.val, :alg_rrule, nothing)
    primal, rrule_func = PEPSKit._rrule(alg_rrule, f.val, map(arg -> getfield(arg, :val), args)...)
    shadow = Enzyme.make_zero(primal)
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, rrule_func))
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(Core.kwcall)},
        ::Type{RT},
        cache,
        kw::Const{<:NamedTuple},
        ::Const{typeof(PEPSKit.hook_pullback)},
        args::Annotation...
    ) where {RT}
    shadow, rrule_func = cache
    rrule_func(shadow)
    return ntuple(Returns(nothing), 2 + length(args))
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth{1},
        ::Const{typeof(MPSKit.leading_boundary)},
        ::Type{RT},
        envinit::Annotation,
        state::Annotation,
        alg::Const{<:CTMRGAlgorithm}
    ) where {RT}
    #PEPSKit._check_algorithm_combination(alg, gradmode)
    env, info = MPSKit.leading_boundary(envinit.val, state.val, alg.val)
    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = PEPSKit._set_fixed_truncation(alg.val) # fix spaces during differentiation
    alg_gauge = PEPSKit._scrambling_env_gauge(alg.val) # select appropriate gauge-fixing algorithm
    env_conv, _ = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(state.val), env, alg_fixed)
    shadow = EnzymeRules.needs_shadow(config) ? Enzyme.make_zero((env, info)) : nothing
    denv = isnothing(shadow) ? nothing : shadow[1]
    primal = EnzymeRules.needs_primal(config) ? (env, info) : nothing
    signs, corner_phases, edge_phases = PEPSKit.compute_gauge_fix_gauge(
        env_conv, env, alg_gauge,
    )
    cache = (env, denv, alg_fixed, signs, corner_phases, edge_phases)
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

        # `out_shadow` is the object the tape will read back, so it must not be
        # replaced with fresh tensors -- doing so orphans the seed and the reverse
        # sweep returns zero.  Where the space metadata is missing (`ℂ^0`), rebuild
        # around the *existing* `data` buffer so the identity Enzyme recorded is
        # preserved.
        nrep = 0
        for i in eachindex(out_shadow.corners)
            sv = space(env.corners[i])
            space(out_shadow.corners[i]) == sv && continue
            length(out_shadow.corners[i].data) == length(env.corners[i].data) || continue
            out_shadow.corners[i] = TensorMap(out_shadow.corners[i].data, sv)
            nrep += 1
        end
        for i in eachindex(out_shadow.edges)
            sv = space(env.edges[i])
            space(out_shadow.edges[i]) == sv && continue
            length(out_shadow.edges[i].data) == length(env.edges[i].data) || continue
            out_shadow.edges[i] = TensorMap(out_shadow.edges[i].data, sv)
            nrep += 1
        end
        # seed the output shadow (`copyto!` is not defined for CTMRGEnv)
        add!(out_shadow, Δ, One(), One())
        rev(Const(gauge_fixed_iteration), state_dup, env_dup, tape)
        return dstate, denv_scratch
    end
    # split off state and environment parts
    ∂f∂A(x)::typeof(state.val) = vjp(x)[1]
    ∂f∂x(x)::typeof(env) = vjp(x)[2]
    # evaluate the geometric sum
    # TODO: thread the caller's `FixedPointGradient` through instead of defaulting.
    ∂A = PEPSKit.fixedpoint_gradient(
        denv, ∂f∂x, ∂f∂A, denv, PEPSKit.FixedPointGradient().solver_alg
    )
    if !isa(state, Const)
        _accum!(state.dval, ∂A)
    end
    return ntuple(Returns(nothing), 3)
end

@inline _dtmap_elem(src::Const, i) = Const(src.val[i])
@inline _dtmap_elem(src::Annotation, i) = Duplicated(src.val[i], src.dval[i])

# VectorInterface has no `add!` for `InfiniteSquareNetwork`.
@noinline function _accum!(@nospecialize(dst), @nospecialize(src), depth = 0)
    depth > 5 && return nothing
    if dst isa AbstractTensorMap
        dst.data .+= src.data
    elseif dst isa Tuple || dst isa NamedTuple
        for i in 1:length(dst)
            _accum!(dst[i], src[i], depth + 1)
        end
    elseif dst isa AbstractArray
        for i in eachindex(dst)
            (isassigned(dst, i) && isassigned(src, i)) || continue
            _accum!(dst[i], src[i], depth + 1)
        end
    elseif isstructtype(typeof(dst)) && !(dst isa Number) && !(dst isa Type) &&
            fieldcount(typeof(dst)) > 0
        for k in 1:fieldcount(typeof(dst))
            _accum!(getfield(dst, k), getfield(src, k), depth + 1)
        end
    end
    return nothing
end

"""
Repair shadow tensors whose space metadata was zeroed.

Enzyme re-boxes shadow tensors of mixed pointer/inline layout when they cross out
of a nested augmented-forward thunk: the GC-pointer field (`data`) survives, but
the inline field (`space`) comes back all-zero, so the tensor reads as `ℂ^0`.
Measured directly: the shadow `data` buffer is the *same object* as the correctly
spaced shadow produced upstream and has the correct length -- only the metadata is
gone.  Rebuilding the tensor around that same buffer restores the space without
disturbing accumulation, which still targets the identical vector.

Field traversal is unrolled through `Val` so every `getfield` stays type stable; a
runtime loop over `fieldnames` makes this dynamic and blows up compile time inside
the rule body.
"""
@inline _repair_one!(@nospecialize(v), @nospecialize(d)) = nothing

function _repair_one!(v::AbstractArray{<:AbstractTensorMap}, d::AbstractArray)
    @inbounds for i in eachindex(v)
        (isassigned(v, i) && isassigned(d, i)) || continue
        sv = space(v[i])
        space(d[i]) == sv && continue
        length(d[i].data) == length(v[i].data) || continue
        d[i] = TensorMap(d[i].data, sv)
    end
    return nothing
end

@inline _repair_fields!(@nospecialize(v), @nospecialize(d), ::Val{0}) = nothing
@inline function _repair_fields!(@nospecialize(v), @nospecialize(d), ::Val{N}) where {N}
    _repair_fields!(v, d, Val(N - 1))
    _repair_one!(getfield(v, N), getfield(d, N))
    return nothing
end

@noinline function _repair_pair!(@nospecialize(v), @nospecialize(d))
    _repair_fields!(v, d, Val(fieldcount(typeof(v))))
    return nothing
end

@noinline function _repair_shadow_spaces!(f)
    isa(f, Const) && return nothing
    _repair_pair!(f.val, f.dval)
    return nothing
end

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

function _dtmap_augmented!(config, f::FA, dst, src) where {FA <: Annotation}
    ET = eltype(src.val)
    SA = src isa Const ? Const{ET} : Duplicated{ET}
    # Propagate the caller's runtime-activity setting into the nested thunk.
    # Without this the inner differentiation runs with static activity while the
    # outer one does not, and derivative contributions are silently dropped.
    mode = Enzyme.set_runtime_activity(ReverseSplitWithPrimal, config)
    fwd, rev = Enzyme.autodiff_thunk(mode, FA, Duplicated, SA)

    _repair_shadow_spaces!(f)

    inds = collect(eachindex(src.val))
    tapes = Vector{Any}(undef, length(inds))
    elems = Vector{Any}(undef, length(inds))
    for (k, i) in enumerate(inds)
        arg = _dtmap_elem(src, i)
        tape, primal, shadow = fwd(f, arg)
        dst.val[i] = primal
        isa(dst, Const) || (dst.dval[i] = shadow)
        tapes[k] = tape
        elems[k] = arg
    end
    return (rev, inds, tapes, elems)
end

function _dtmap_reverse!(f::FA, cache) where {FA <: Annotation}
    rev, inds, tapes, elems = cache
    for k in eachindex(tapes)
        rev(f, elems[k], tapes[k])
    end
    return nothing
end

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
    _dtmap_reverse!(f, cache)
    return (nothing, nothing, nothing)
end

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

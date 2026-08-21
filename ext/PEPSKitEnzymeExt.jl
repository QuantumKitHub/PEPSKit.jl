module PEPSKitEnzymeExt

using PEPSKit, MPSKit, TensorKit, MatrixAlgebraKit
using PEPSKit: SVDAdjoint, EighAdjoint, QRAdjoint, CTMRGAlgorithm, FixedPointGradient, sdiag_pow, dtmap, dtmap!!
using PEPSKit: InfiniteSquareNetwork, InfinitePEPS, InfinitePEPO, _stack_tuples
using PEPSKit: unitcell, ket, bra, pepo
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

# Without this, Enzyme differentiates through `ignore_derivatives`
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
    # This creates new output shadow matrices, we use USVᴴtrunc to ensure the
    # eltypes and dimensions are correct.
    # These new shadow matrices are "filled in" with the accumulated
    # results from earlier in reverse-mode AD after this function exits
    # and before `reverse` is called.
    dret = if EnzymeRules.needs_shadow(config)
        (zero(USVᴴtrunc[1]), Diagonal(zero(USVᴴtrunc[2].diag)), zero(USVᴴtrunc[3]))
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
        t.dval = MatrixAlgebraKit.svd_pullback!(
            t.dval, t.val, (U, S, V⁺), dUSVᴴtrunc, ind;
            gauge_atol = gtol(dUSVᴴtrunc), degeneracy_atol = alg.val.rrule_alg.degeneracy_atol,
        )
    end
    return ntuple(Returns(nothing), 3)
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
    # prepare its pullback
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(gauge_fixed_iteration)}, Duplicated, typeof(state), Duplicated{typeof(env)})
    # implement the VJP-getting
    state_dup = isa(state, Const) ? Duplicated(state.val, Enzyme.make_zero(state.val)) : state
    env_dup = Duplicated(env, denv)
    tape, _, out_shadow = fwd(Const(gauge_fixed_iteration), state_dup, env_dup)
    function vjp(Δ)
        Enzyme.make_zero!(state_dup.dval)
        Enzyme.make_zero!(env_dup.dval)
        copyto!(out_shadow, Δ)          # seed the output shadow
        rev(Const(gauge_fixed_iteration), state_dup, env_dup, tape)
        return state_dup.dval, env_dup.dval
    end
    # split off state and environment parts
    ∂f∂A(x)::typeof(state.val) = vjp(x)[1]
    ∂f∂x(x)::typeof(env) = vjp(x)[2]
    # evaluate the geometric sum
    #PEPSKit.fixedpoint_gradient(denv, ∂f∂x, ∂f∂A, denv, gradmode.solver_alg)
    PEPSKit.fixedpoint_gradient(denv, ∂f∂x, ∂f∂A, denv, PEPSKit.Defaults.gradient_fixedpoint_solver_alg)
    return ntuple(Returns(nothing), 4)
end

# --- dtmap / dtmap!! -------------------------------------------------------
#
# `dtmap!!(f, dst, src)` is `tmap!`; `dtmap(f, A)` is `tmap`. Both take their
# scheduler as a *keyword*, so a rule with a positional `scheduler` argument can
# never match -- which is why the previous rules here never fired.
#
# Differentiating them generically is not merely slow, it is wrong: Enzyme
# allocates the shadow for the result array without carrying each element's
# space, and for TensorMaps whose spaces differ per element (the four CTMRG
# directions) that yields zero-dimensional spaces and a `SpaceMismatch` on the
# reverse sweep. Going element by element keeps every shadow tied to the value
# it belongs to.
#
# The scheduler is dropped while differentiating: these run serially, as the
# `@fwdthreads` macro already does for the backward pass.

@inline _dtmap_elem(src::Const, i) = Const(src.val[i])
@inline _dtmap_elem(src::Annotation, i) = Duplicated(src.val[i], src.dval[i])

"""
Run the augmented forward of `f` over every element of `src`, writing primals
into `dst.val` and each element's Enzyme-allocated shadow into `dst.dval`, so
that downstream accumulation lands directly on the object the reverse sweep
will read back.
"""
function _dtmap_augmented!(f::FA, dst, src) where {FA <: Annotation}
    ET = eltype(src.val)
    SA = src isa Const ? Const{ET} : Duplicated{ET}
    fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, FA, Duplicated, SA)

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
    cache = _dtmap_augmented!(f, dst, src)
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

# No rule for `dtmap` itself: it is `tmap`, which Enzyme already differentiates
# correctly (verified against finite differences), and the rules that used to
# live here could never fire -- they took `scheduler` as a positional argument
# while it is only ever a keyword. Adding a rule here is not free: an earlier
# version of this patch did, and it broke `dtmap` with an
# `AugmentedRuleReturnError` where the generic path had worked.

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
    !isa(top, Const) && add!(top.dval, InfinitePEPS(map(ket, unitcell(Δnetwork))), One(), One())
    !isa(bot, Const) && add!(bot.dval, InfinitePEPS(map(bra, unitcell(Δnetwork))), One(), One())
    !isa(mid, Const) && add!(mid.dval, InfinitePEPO(_stack_tuples(map(pepo, unitcell(Δnetwork)))), One(), One())
    return (nothing, nothing, nothing)
end

end

module PEPSKitEnzymeExt

using PEPSKit, MPSKit, TensorKit, MatrixAlgebraKit
using PEPSKit: SVDAdjoint, EighAdjoint, QRAdjoint, CTMRGAlgorithm, FixedPointGradient, sdiag_pow
import PEPSKit: real_inner
using Enzyme
using Enzyme.EnzymeCore: EnzymeRules

@inline EnzymeRules.inactive_type(::Type{SVDAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{QRAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{EighAdjoint}) = true
@inline EnzymeRules.inactive_type(::Type{CTMRGAlgorithm}) = true

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    func::Const{typeof(MatrixAlgebraKit.svd_trunc)},
    ::Type{RT},
    t::Annotation,
    alg::Const{<:SVDAdjoint{F, R}}
    ) where {RT, F, R <: PEPSKit.FullPullback}
    # requires access to the full decomposition
    U, S, V⁺ = svd_compact(t.val, alg.val.fwd_alg.alg)
    (Ũ, S̃, Ṽ⁺), inds = MatrixAlgebraKit.truncate(svd_trunc!, (U, S, V⁺), alg.val.fwd_alg.trunc)
    truncerror = MatrixAlgebraKit.truncation_error(diagview(S), inds)

    gtol = PEPSKit._get_pullback_gauge_tol(alg.val.rrule_alg.verbosity)
    output = (Ũ, S̃, Ṽ⁺, truncerror)
    USVᴴtrunc = (Ũ, S̃, Ṽ⁺)
    primal = EnzymeRules.needs_primal(config) ? USVᴴ′ : nothing
    # This creates new output shadow matrices, we use USVᴴ′ to ensure the
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
    func::Const{typeof(MatrixAlgebraKit.svd_trunc)},
    ::Type{RT},
    cache,
    t::Annotation,
    alg::Const{<:SVDAdjoint{F, R}}
    ) where {RT, F, R <: PEPSKit.FullPullback}
    dUSVᴴtrunc, USV⁺, ind = cache
    U, S, V⁺ = USV⁺
    _warn_pullback_truncerror(dϵ)
    if !isa(t, Const)
        t.dval = MatrixAlgebraKit.svd_pullback!(
            t.dval, t.val, (U, S, V⁺), ΔUSVᴴtrunc, ind;
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
    args::Annotation...) where {RT}
    alg_rrule = get(kw.val, :alg_rrule, nothing)
    println("IN AUGMENTED PRIMAL")
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
    args::Annotation...) where {RT}
    println("IN REVERSE")
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
    alg::Const{<:CTMRGAlgorithm}) where {RT}
    #PEPSKit._check_algorithm_combination(alg, gradmode)
    env, = MPSKit.leading_boundary(envinit.val, state.val, alg.val)
    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = PEPSKit._set_fixed_truncation(alg.val) # fix spaces during differentiation
    alg_gauge = PEPSKit._scrambling_env_gauge(alg.val) # select appropriate gauge-fixing algorithm
    env_conv, info = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(state.val), env.val, alg_fixed)
    shadow = Enzyne.make_zero((env, info))
    return EnzymeRules.AugmentedReturn(primal, shadow, (env_conv, alg_gauge, alg_fixed))
end

function EnzymeRules.reverse(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(MPSKit.leading_boundary)},
    ::Type{RT},
    cache,
    envinit::Annotation,
    state::Annotation,
    alg::Const{<:CTMRGAlgorithm}) where {RT}
    env_conv, alg_gauge, alg_fixed = cache
    signs, corner_phases, edge_phases = PEPSKit.compute_gauge_fix_gauge(env_conv, env.val, alg_gauge)
    function gauge_fixed_iteration(A, x)
        x′ = PEPSKit.ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1]
        return PEPSKit.fix_phases(x′, signs, corner_phases, edge_phases)
    end
    # prepare its pullback
    sig = Tuple{typeof(gauge_fixed_iteration), typeof(state), typeof(env)}
    rule = Mooncake.build_rrule(gauge_fixed_iteration, state, env)
    println("RUN AUTODIFF IN RRULE")
    env_vjp = Enzyme.autodiff(rule, gauge_fixed_iteration, state, env)
    # split off state and environment parts
    ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
    ∂f∂x(x)::typeof(env) = env_vjp(x)[3]
    # evaluate the geometric sum
    PEPSKit.fixedpoint_gradient(env.dval, ∂f∂x, ∂f∂A, env.dval, gradmode.solver_alg)
    return ntuple(Returns(NoRData()), 4)
end

end

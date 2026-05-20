abstract type GradMode{A} end

const GRADIENT_MODE_SYMBOLS = IdDict{Symbol, Type{<:GradMode}}(
    :FixedPointGradient => FixedPointGradient,
)

struct FixedPointGradient{A}
    solver_alg::A
end

const FIXEDPOINT_SOLVER_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :GMRES => GMRES, :BiCGStab => BiCGStab, :Arnoldi => Arnoldi,
    :GeomSum => GeomSum, :ManualIter => ManualIter,

)

_default_solver_alg(::Type{<:FixedPointGradient}) = Defaults.gradient_fixedpoint_solver_alg
_select_solver_alg_symbol(::Type{<:FixedPointGradient}, solver_alg) =
    FIXEDPOINT_SOLVER_SYMBOLS[solver_alg]

_pad_solver_kwargs(::Type, solver_kwargs) = solver_kwargs
function _pad_solver_kwargs(::Type{<:KrylovKit.KrylovAlgorithm}, solver_kwargs)
    solver_kwargs = (;
        eager = Defaults.gradient_fixedpoint_solver_eager,
        solver_kwargs...,
    )
    return solver_kwargs
end

"""
    GradMode(; kwargs...)

Keyword argument parser returning the appropriate `GradMode` algorithm struct.
"""
function GradMode(;
        alg = Defaults.gradient_alg,
        tol = Defaults.gradient_tol,
        maxiter = Defaults.gradient_maxiter,
        verbosity = Defaults.gradient_verbosity,
        solver_alg = (;),
    )
    # replace symbol with GradMode alg type
    haskey(GRADIENT_MODE_SYMBOLS, alg) ||
        throw(ArgumentError("unknown GradMode algorithm: $alg"))
    alg_type = GRADIENT_MODE_SYMBOLS[alg]

    # parse solver algorithm
    solver = if solver_alg isa NamedTuple # determine linear/eigen solver algorithm
        solver_kwargs = (;
            alg = _default_solver_alg(alg_type),
            tol,
            maxiter,
            verbosity,
            solver_alg...,
        ) # overwrite with specified kwargs

        # parse solver algorithm type
        solver_alg_type = _select_solver_alg_symbol(alg_type, solver_alg.alg)

        # pad solver_kwargs based on solver type requirements
        solver_kwargs = _pad_solver_kwargs(solver_alg_type, solver_kwargs)

        # remove `alg` keyword argument
        solver_kwargs = Base.structdiff(solver_kwargs, (; alg = nothing))

        solver_alg_type(; solver_kwargs...)
    else
        solver_alg
    end

    return alg_type(solver)
end

"""
$(TYPEDEF)

Algorithm for solving the fixed-point gradient linear problem as a geometric sum.

## Fields

$(TYPEDFIELDS)

## Constructors

    GeomSum(; kwargs...)

Construct the `GeomSum` algorithm struct based on the following keyword arguments:

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance for the difference of norms of two consecutive summands in the geometric sum.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity that can be one of the following:
    0. Suppress output information
    1. Print convergence warnings
    2. Information at each gradient iteration
"""
@kwdef struct GeomSum
    tol::Real = Defaults.gradient_tol
    maxiter::Int = Defaults.gradient_maxiter
    verbosity::Int = Defaults.gradient_verbosity
end

"""
$(TYPEDEF)

Algorithm for solving the fixed-point gradient linear problem using manual iteration.

## Fields

$(TYPEDFIELDS)

## Constructors

    ManualIter(; kwargs...)

Construct the `ManualIter` algorithm struct based on the following keyword arguments:

* `tol::Real=$(Defaults.gradient_tol)` : Convergence tolerance for the norm difference of two consecutive `dx` contributions.
* `maxiter::Int=$(Defaults.gradient_maxiter)` : Maximal number of gradient iterations.
* `verbosity::Int=$(Defaults.gradient_verbosity)` : Output information verbosity that can be one of the following:
    0. Suppress output information
    1. Print convergence warnings
    2. Information at each gradient iteration
"""
@kwdef struct ManualIter
    tol::Real = Defaults.gradient_tol
    maxiter::Int = Defaults.gradient_maxiter
    verbosity::Int = Defaults.gradient_verbosity
end

"""
    _check_algorithm_combination(boundary_alg, gradient_alg_or_symmetrization)
    _check_algorithm_combination(boundary_alg, gradient_alg, symmetrization)

Check for allowed combinations of gradient algorithm and boundary algorithm to be used for
computing the gradient of a `leading_boundary` call. Throws an error containing a
recommended fix if the combination is not allowed or broken.
"""
function _check_algorithm_combination(boundary_alg, gradient_alg_or_symmetrization) end
function _check_algorithm_combination(boundary_alg, gradient_alg, symmetrization)
    _check_algorithm_combination(boundary_alg, gradient_alg)
    _check_algorithm_combination(boundary_alg, symmetrization)
    return nothing
end
function _check_algorithm_combination(::SequentialCTMRG, ::FixedPointGradient)
    msg = "The `:FixedPointGradient` algorithm is not compatible with `SequentialCTMRG` since the sequential \
          application of SVDs does not allow to differentiate through a fixed set of \
          gauges; select SimultaneousCTMRG instead to use :fixed mode"
    throw(ArgumentError(msg))
end
function _check_algorithm_combination(::C4vCTMRG, symm::Union{Nothing, <:SymmetrizationStyle})
    if !(symm isa RotateReflect)
        msg = "C4vCTMRG optimization is compatible only with RotateReflect symmetrization. \
            Make sure to set `symmetrization = RotateReflect()`."
        throw(ArgumentError(msg))
    end
    return nothing
end

#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

_scrambling_env_gauge(::CTMRGAlgorithm) = ScramblingEnvGauge()
_scrambling_env_gauge(::C4vCTMRG) = ScramblingEnvGaugeC4v()

function _set_fixed_truncation(alg::CTMRGAlgorithm)
    alg_fixed = @set alg.projector_alg = _set_truncation(alg.projector_alg, FixedSpaceTruncation())
    return alg_fixed
end

# compute the CTMRG gradient through fixed-point differentiation
function _rrule(
        gradmode::FixedPointGradient,
        config::RuleConfig,
        ::typeof(MPSKit.leading_boundary),
        envinit,
        state,
        alg::CTMRGAlgorithm,
    )
    _check_algorithm_combination(alg, gradmode)

    env, = leading_boundary(envinit, state, alg)

    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = _set_fixed_truncation(alg) # fix spaces during differentiation
    alg_gauge = _scrambling_env_gauge(alg) # TODO: make this a field in GradMode?
    env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, alg_fixed)
    signs, corner_phases, edge_phases = compute_gauge_fix_gauge(env_conv, env, alg_gauge)
    function gauge_fixed_iteration(A, x)
        return fix_phases(
            ctmrg_iteration(InfiniteSquareNetwork(A), x, alg_fixed)[1],
            signs, corner_phases, edge_phases,
        )
    end
    # prepare its pullback
    _, env_vjp = rrule_via_ad(config, gauge_fixed_iteration, state, env)
    # split off state and environment parts
    ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
    ∂f∂x(x)::typeof(env) = env_vjp(x)[3]

    function leading_boundary_fixed_pullback((Δenv′, Δinfo))
        Δenv = unthunk(Δenv′)

        # evaluate the geometric sum
        ∂F∂env = fixedpoint_gradient(Δenv, ∂f∂x, ∂f∂A, Δenv, gradmode.solver_alg)

        return NoTangent(), ZeroTangent(), ∂F∂env, NoTangent()
    end

    return (env, info), leading_boundary_fixed_pullback
end

@doc """
    fixedpoint_gradient(∂e∂x, ∂f∂x, ∂f∂p, y0, alg)

TODO: explain fixed-point differentiation.

Compute the gradient of the CTMRG fixed point by solving the following equation:

dx = ∑ₙ (∂f∂x)ⁿ ∂f∂A dA = (1 - ∂f∂x)⁻¹ ∂f∂A dA

where `∂F∂x` is the gradient of the cost function with respect to the PEPS tensors, `∂f∂x`
is the partial gradient of the CTMRG iteration with respect to the environment tensors,
`∂f∂A` is the partial gradient of the CTMRG iteration with respect to the PEPS tensors, and
`y0` is the initial guess for the fixed-point iteration. The function returns the gradient
`dx` of the fixed-point iteration.
"""
fixedpoint_gradient

# TODO: can we construct an implementation that does not need to evaluate the vjp
# twice if both ∂f∂A and ∂f∂x are needed?
function fixedpoint_gradient(∂E∂x, ∂f∂x, ∂f∂A, _, alg::GeomSum)
    g = ∂E∂x
    dx = ∂f∂A(g) # n = 0 term: ∂F∂x ∂f∂A
    ϵ = 2 * alg.tol
    for i in 1:(alg.maxiter)
        g = ∂f∂x(g)
        Σₙ = ∂f∂A(g)
        dx += Σₙ
        ϵnew = norm(Σₙ)  # TODO: normalize this error?
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 &&
            @printf("Gradient iter: %3d   ‖Σₙ‖: %.2e   Δ‖Σₙ‖: %.2e\n", i, ϵnew, Δϵ)
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Σₙ‖ = $ϵ"
        end
    end
    return dx
end

function fixedpoint_gradient(∂E∂x, ∂f∂x, ∂f∂A, y₀, alg::ManualIter)
    y = deepcopy(y₀)  # Do not mutate y₀
    dx = ∂f∂A(y)
    ϵ = 1.0
    for i in 1:(alg.maxiter)
        y′ = ∂E∂x + ∂f∂x(y)

        dxnew = ∂f∂A(y′)
        ϵnew = norm(dxnew - dx)
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 && @printf(
            "Gradient iter: %3d   ‖Cᵢ₊₁-Cᵢ‖/N: %.2e   Δ‖Cᵢ₊₁-Cᵢ‖/N: %.2e\n", i, ϵnew, Δϵ
        )
        y = y′
        dx = dxnew
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Cᵢ₊₁-Cᵢ‖ = $ϵ"
        end
    end
    return dx
end

function fixedpoint_gradient(∂E∂x, ∂f∂x, ∂f∂A, y₀, alg::KrylovKit.LinearSolver)
    y, info = reallinsolve(∂f∂x, ∂E∂x, y₀, alg, 1, -1)
    if alg.solver_alg.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

function fixedpoint_gradient(∂E∂x, ∂f∂x, ∂f∂A, x₀, alg::KrylovKit.KrylovAlgorithm)
    function f(X)
        y = ∂f∂x(X[1])
        return (VI.add!!(y, ∂E∂x, X[2]), X[2])
    end
    X₀ = (x₀, one(scalartype(x₀)))
    _, vecs, info = realeigsolve(f, X₀, 1, :LM, alg)
    if alg.solver_alg.verbosity > 0 && info.converged < 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end
    if norm(vecs[1][2]) < 1.0e-2 * alg.tol
        @warn "Fixed-point gradient computation using Arnoldi failed:\n\tauxiliary component should be finite but was $(vecs[1][2])\n\tpossibly the Jacobian does not have a unique eigenvalue 1"
        @info "Falling back to linear solver for fixed-point gradient computation."
        backup_ls_alg = GMRES(; tol = alg.tol, maxiter = alg.maxiter, verbosity = alg.verbosity)
        return fixedpoint_gradient(∂E∂x, ∂f∂x, ∂f∂A, x₀, backup_ls_alg)
    else
        y = VI.scale!!(vecs[1][1], inv(vecs[1][2]))
    end

    return ∂f∂A(y)
end

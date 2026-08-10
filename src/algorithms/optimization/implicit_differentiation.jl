abstract type GradientAlgorithm{A} end

const GRADIENT_ALGORITHM_SYMBOLS = IdDict{Symbol, Type{<:GradientAlgorithm}}()

# default solver algorithm for each gradient algorithm type
_default_solver_alg(::Type{T}) where {T <: GradientAlgorithm} =
    throw(ArgumentError("No default solver algorithm defined for gradient algorithm $(T)"))
# map solver algorithm symbol to solver algorithm type for each gradient algorithm type
_select_solver_alg_symbol(::Type{T}, solver_alg) where {T <: GradientAlgorithm} =
    throw(ArgumentError("No solver algorithm symbols specified for gradient algorithm $(T)"))
# add algorithm-specific keyword arguments to solver kwargs if needed
_pad_solver_kwargs(::Type, solver_kwargs) = solver_kwargs

"""
    GradientAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `GradientAlgorithm` algorithm struct.
"""
function GradientAlgorithm(;
        alg = Defaults.gradient_alg,
        tol = Defaults.gradient_tol,
        maxiter = Defaults.gradient_maxiter,
        verbosity = Defaults.gradient_verbosity,
        solver_alg = (;),
    )
    # replace symbol with GradientAlgorithm alg type
    haskey(GRADIENT_ALGORITHM_SYMBOLS, alg) ||
        throw(ArgumentError("unknown GradientAlgorithm algorithm: $alg"))
    alg_type = GRADIENT_ALGORITHM_SYMBOLS[alg]

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
        solver_alg_type = _select_solver_alg_symbol(alg_type, solver_kwargs.alg)

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

#
# Fixed-point gradient computation
#


"""
$(TYPEDEF)

Gradient algorithm for computing the gradient of a fixed-point problem through
implicit fixed-point differentiation.

## Fields

$(TYPEDFIELDS)

## Constructors

    FixedPointGradient(; kwargs...)

Construct a fixed-point gradient algorithm struct based on keyword arguments.
The supported keywords are:

* `tol::Real=$(Defaults.gradient_tol)`
* `maxiter::Int=$(Defaults.gradient_maxiter)`
* `verbosity::Int=$(Defaults.gradient_verbosity)`
* `solver_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=:$(Defaults.gradient_fixedpoint_solver_alg))`: solver algorithm for the `FixedPointGradient` gradient algorithm.
    - `:GMRES` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref) for details
    - `:BiCGStab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref) for details
    - `:Arnoldi` : Arnoldi Krylov algorithm, see [`KrylovKit.Arnoldi`](@extref) for details
    - `:GeomSum` : Geometric sum approximation of the Neumann series of the inverse Jacobian, see [`PEPSKit.GeomSum`](@ref) for details
    - `:ManualIter` : Manual fixed-point iteration, see [`PEPSKit.ManualIter`](@ref) for details
"""
struct FixedPointGradient{A} <: GradientAlgorithm{A}
    solver_alg::A
end
FixedPointGradient(; kwargs...) = GradientAlgorithm(; alg = :FixedPointGradient, kwargs...)
GRADIENT_ALGORITHM_SYMBOLS[:FixedPointGradient] = FixedPointGradient

const FIXEDPOINT_SOLVER_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :GMRES => GMRES, :BiCGStab => BiCGStab, :Arnoldi => Arnoldi,
)

_default_solver_alg(::Type{<:FixedPointGradient}) = Defaults.gradient_fixedpoint_solver_alg
_select_solver_alg_symbol(::Type{<:FixedPointGradient}, solver_alg) =
    FIXEDPOINT_SOLVER_SYMBOLS[solver_alg]
function _pad_solver_kwargs(::Type{<:Arnoldi}, solver_kwargs)
    solver_kwargs = (;
        eager = Defaults.gradient_fixedpoint_solver_eager,
        solver_kwargs...,
    )
    return solver_kwargs
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
FIXEDPOINT_SOLVER_SYMBOLS[:GeomSum] = GeomSum

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
FIXEDPOINT_SOLVER_SYMBOLS[:ManualIter] = ManualIter

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
function _check_algorithm_combination(::C4vCTMRG, symm::Union{Nothing, <:SymmetrizationStyle})
    if !(symm isa RotateReflect)
        msg = "C4vCTMRG optimization is compatible only with combined Hermitian reflection and rotation symmetrization. \
            Make sure to set `symmetrization = RotateReflect()` or implement an equivalent custom symmetrization scheme."
        @warn msg
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
        envinit::CTMRGEnv,
        state,
        alg::CTMRGAlgorithm,
    )
    _check_algorithm_combination(alg, gradmode)

    env, = leading_boundary(envinit, state, alg)

    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = _set_fixed_truncation(alg) # fix spaces during differentiation
    alg_gauge = _scrambling_env_gauge(alg) # select appropriate gauge-fixing algorithm
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

@doc raw"""
    fixedpoint_gradient(x̆, ∂ₓf, ∂ₚf, y₀, alg)

Evaluates the VJP action  ``x̆ ∂ₚx`` for an intermediate variable ``x \equiv x(p)``
which satisfies the fixed-point equation ``x = f(x, p)``, given the VJP actions ``∂ₓf``
and ``∂ₚf`` of the iterating function ``f``.

More specifically, given a cost function ``E(x(p), p)`` defined in terms of a set of
variational parameters ``p`` and a set of intermediate variables ``x`` that depend on ``p``,
``x \equiv x(p)``, the gradient of the cost function is given by

```math
dE/dp = ∂ₓE ∂ₚx + ∂ₚE.
```

Given the fixed-point equation ``x = f(x, p)``, the VJP action of the Jacobian ``∂ₚx``` on
the adjoint ``x̆ = ∂ₓE`` in the first term of this expression can be evaluated through
implicit differentiation of the fixed-point condition as
```math
x̆ ∂ₚx = x̆ (1 - ∂ₓf)⁻¹ ∂ₚf = ∑ₙ x̆ (∂ₓf)ⁿ ∂ₚf.
```

This can be used to differentiate contraction routines, where ``p`` are the variational
parameters of a tensor network, ``x̆ = ∂ₓE`` is the partial
derivative of the cost function with respect to the contraction environment ``x``, and ``f``
is a single iteration of the contraction algorithm.
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
    if alg.verbosity > 0 && info.converged != 1
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
    if alg.verbosity > 0 && info.converged < 1
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

#
# Implicit gradient computation
#


"""
$(TYPEDEF)

Gradient algorithm for computing the gradient of an optimization procedure by
differentiating an implicit algebraic characterization of the solution.

## Fields

$(TYPEDFIELDS)

## Constructors

    ImplicitGradient(; kwargs...)

Construct an implicit gradient algorithm struct based on keyword arguments.
The supported keywords are:

* `tol::Real=$(Defaults.gradient_tol)`
* `maxiter::Int=$(Defaults.gradient_maxiter)`
* `verbosity::Int=$(Defaults.gradient_verbosity)`
* `solver_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=:$(Defaults.gradient_fixedpoint_solver_alg))`: solver algorithm for the `ImplicitGradient` gradient algorithm.
    - `:GMRES` : GMRES iterative linear solver, see [`KrylovKit.GMRES`](@extref) for details
    - `:BiCGStab` : BiCGStab iterative linear solver, see [`KrylovKit.BiCGStab`](@extref) for details
"""
struct ImplicitGradient{A} <: GradientAlgorithm{A}
    solver_alg::A
end
ImplicitGradient(; kwargs...) = GradientAlgorithm(; alg = :ImplicitGradient, kwargs...)
GRADIENT_ALGORITHM_SYMBOLS[:ImplicitGradient] = ImplicitGradient

const IMPLICIT_SOLVER_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :GMRES => GMRES, :BiCGStab => BiCGStab,
)

_default_solver_alg(::Type{<:ImplicitGradient}) = Defaults.gradient_implicit_solver_alg
_select_solver_alg_symbol(::Type{<:ImplicitGradient}, solver_alg) =
    IMPLICIT_SOLVER_SYMBOLS[solver_alg]

function _check_algorithm_combination(::CTMRGAlgorithm, ::ImplicitGradient)
    msg = "The `:ImplicitGradient` algorithm is currently only implemented for the `C4vCTMRG` algorithm."
    throw(ArgumentError(msg))
end
_check_algorithm_combination(::C4vCTMRG, ::ImplicitGradient) = nothing

function _gauge_fix_c4v_projector(::C4vCTMRG{<:C4vEighProjector}, signs, info)
    return info.V * signs[1]'
end
function _gauge_fix_c4v_projector(::C4vCTMRG{<:C4vQRProjector}, signs, info)
    return info.Q * signs[1]'
end

function _check_algorithm_combination(::SequentialCTMRG, ::ImplicitGradient)
    msg = "The `:ImplicitGradient` algorithm is not yet compatible with the `SequentialCTMRG` algorithm; select \
          SimultaneousCTMRG instead to use :ImplicitGradient"
    throw(ArgumentError(msg))
end

function _check_algorithm_combination(::SimultaneousCTMRG{<:FullInfiniteProjector}, ::ImplicitGradient)
    msg = "The `:ImplicitGradient` algorithm is not yet compatible with the `FullInfiniteProjector` scheme, \
          since the corresponding charageristic equations have not yet been implemented; select \
          SimultaneousCTMRG with HalfInfiniteProjector instead to use :ImplicitGradient"
    throw(ArgumentError(msg))
end


@doc raw"""
    implicit_gradient(x̆, ∂ₓF, ∂ₚF, y₀, alg)

Evaluates the VJP action  ``x̆ ∂ₚx`` for an intermediate variable ``x \equiv x(p)``
which satisfies the characteristic equation ``F(x, p) = 0``, given the VJP actions ``∂ₓF``
and ``∂ₚF`` of the function ``F`` encoding the characteristic equation.

More specifically, given a cost function ``E(x(p), p)`` defined in terms of a set of
variational parameters ``p`` and a set of intermediate variables ``x`` that depend on ``p``,
``x \equiv x(p)``, the gradient of the cost function is given by

```math
dE/dp = ∂ₓE ∂ₚx + ∂ₚE.
```

Given the characteristic equation ``F(x, p) = 0``, the VJP action of the Jacobian ``∂ₚx``` on
the adjoint ``x̆ = ∂ₓE`` in the first term of this expression can be evaluated through
implicit differentiation of the characteristic equation as
```math
x̆ ∂ₚx = x̆ (∂ₓF)⁻¹ (-∂ₚF).
```

This can be used to differentiate contraction routines, where ``p`` are the variational
parameters of a tensor network, ``x̆ = ∂ₓE`` is the partial
derivative of the cost function with respect to the contraction environment ``x``, and ``F``
is an algebraic function characterizing the convergence of the algorithm.
"""
implicit_gradient

function implicit_gradient(∂E∂x, ∂F∂x, ∂F∂A, y₀, alg::KrylovKit.LinearSolver)
    y, info = reallinsolve(∂F∂x, ∂E∂x, y₀, alg)
    if alg.verbosity > 0 && info.converged != 1
        @warn("Implicit gradient linear solve reached maximal number of iterations:", info)
    end

    return (-1) * ∂F∂A(y)
end

## C4vCTMRG gradient through implicit differentiation

function _rrule(
        gradmode::ImplicitGradient,
        config::RuleConfig,
        ::typeof(MPSKit.leading_boundary),
        envinit::CTMRGEnv,
        state,
        alg::C4vCTMRG,
    )
    _check_algorithm_combination(alg, gradmode)

    env, = leading_boundary(envinit, state, alg)

    # prepare iterating function corresponding to a single gauge-fixed CTMRG iteration
    alg_fixed = _set_fixed_truncation(alg) # fix spaces during differentiation
    alg_gauge = _scrambling_env_gauge(alg) # select appropriate gauge-fixing algorithm
    env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, alg_fixed)
    signs, corner_phases, edge_phases = compute_gauge_fix_gauge(env_conv, env, alg_gauge)
    env_fixed = fix_phases(env_conv, signs, corner_phases, edge_phases)

    # NOTE: explicitly keeping corner non-diagonal and (possibly) complex, for use in the backwards pass
    C, E = only(env_fixed.corners[NORTHWEST, :, :]), only(env_fixed.edges[NORTH, :, :])

    # gauge-fix projector accordingly
    U = _gauge_fix_c4v_projector(alg, signs, info)

    # get the projector nullspace
    UL = left_null(U)

    # instantiate the differentiable variables corresponding to the intermediate projector of the contraction algorithm
    u = zeros(scalartype(U), space(UL, numind(UL))' ← space(U, numind(U))')

    # prepare pullback of C4v CTMRG environment constructor (artefact of reusing asymmetric environment type for C4v symmetric contraction)
    _, c4v_env_vjp = rrule_via_ad(config, CTMRGEnv, C, E)

    # initialize the partial pullbacks of the characteristic equations
    F = generate_symmetric_characteristic_equation(C, E, U, UL)

    # get the partial pullback of the characteristic equations
    _, F_vjp = rrule_via_ad(config, F, state, C, E, u)
    vjp_env(x) = F_vjp(x)[3:end] # environment and isometry pullback
    vjp_state(x) = F_vjp(x)[2] # state pullback

    function leading_boundary_implicit_pullback((_Δenv, _Δinfo))
        Δenv, Δinfo = unthunk(_Δenv), unthunk(_Δinfo)

        # accumulate adjoints from all corners & edges through _c4v_env constructor
        _, ΔC, ΔE = c4v_env_vjp(Δenv)

        # apply Hermitian projectors to edge and corner cotangents
        ΔE = _project_hermitian(ΔE)
        ΔC = _project_hermitian(ΔC)

        # extract cotengent of the isometry variable and verify that it is trivial
        ΔU = isa(Δinfo.U, AbstractZero) ? zerovector(U) : unthunk(Δinfo.U)
        Δu = UL' * ΔU
        norm(Δu) < 1.0e4 * alg.tol || @warn "Nonzero Δu cotangent: norm(Δu)=$(norm(Δu))"

        # collect cotangents of characteristic equation
        Δy = (ΔC, ΔE, Δu)

        # evaluate the implicit gradient
        Δstate = implicit_gradient(Δy, vjp_env, vjp_state, Δy, gradmode.solver_alg)

        return NoTangent(), ZeroTangent(), Δstate, NoTangent()
    end

    # HACK: return gauge-fixed environment with potentially non-diagonal and complex corners
    # this ensures that backpropagation through any subsequent observable evaluations will
    # give non-diagonal and complex corner adjoints, which is required for root
    # differentiation to work here
    # NOTE: this very bad practice, since the return type here is manifestly different from
    # what you would get from a regular forward run without differentiation
    return (env_fixed, info), leading_boundary_implicit_pullback
end


## CTMRG gradient through implicit differentiation

# removing inverse roots of singular values from corners and edges
# i.e. what we actually do to get the modified corners and edges
function remove_inverse_roots(
        C::CornerTensors, E::EdgeTensors, S::CornerTensors
    )
    sqS = sdiag_pow.(S, 0.5)
    coordinates = eachcoordinate(C)
    nrows, ncols = size(C)[2:3]
    C′ = map(coordinates) do co
        co′ = _prev_coordinate(co, nrows, ncols)
        return sqS[co′...] * C[co...] * sqS[co...]
    end
    E′ = map(coordinates) do co
        co′ = _edge_sinv_indices(co, nrows, ncols)
        return absorb_left_right(
            E[co...], sqS[co′...], sqS[co...]
        )
    end
    return C′ ./ norm.(C′), E′ ./ norm.(E′)
end

# absorbing inverse roots of singular values into modified corners and edges to obtain
# regular corners and edges
# i.e. the thing we pretend we did right before the cost function evaluation
function absorb_inverse_roots(
        C::CornerTensors, E::EdgeTensors, S::CornerTensors
    )
    # absorb inverse roots
    is = map(inv, S)
    isqS = map(squareroot, is) # do this instead of sdiag_pow(S, -0.5) since dS may be non-diagonal
    coordinates = eachcoordinate(C)
    nrows, ncols = size(C)[2:3]
    C′ = map(coordinates) do co
        co′ = _prev_coordinate(co, nrows, ncols)
        return isqS[co′...] * C[co...] * isqS[co...]
    end
    E′ = map(coordinates) do co
        co′ = _edge_sinv_indices(co, nrows, ncols)
        return absorb_left_right(E[co...], isqS[co′...], isqS[co...])
    end
    return C′ ./ norm.(C′), E′ ./ norm.(E′)
end

# unit cell indices for absorbing square roots of S or S⁻¹ into corners and edges
function _edge_sinv_indices(coordinate, nrows, ncols)
    dir, r, c = coordinate
    r, c = if dir == NORTH
        r, _prev(c, ncols)
    elseif dir == EAST
        _prev(r, nrows), c
    elseif dir == SOUTH
        r, _next(c, ncols)
    elseif dir == WEST
        _next(r, nrows), c
    end
    return (dir, r, c)
end

# characteristic equation for asymmetric simultaneous CTMRG with generic unit cells and sparse SVD,
# i.e. where computation of the pullback requires solving a Sylvester equation for dU and dV
function PEPSKit._rrule(
        gradmode::ImplicitGradient,
        config::RuleConfig,
        ::typeof(MPSKit.leading_boundary),
        envinit::CTMRGEnv,
        state,
        alg::SimultaneousCTMRG{<:HalfInfiniteProjector}, # SequentialCTMRG doesn't return U, S, V (yet)
    )
    env, = leading_boundary(envinit, state, alg)

    # gauge-fix SVD isometries
    env_conv, info = ctmrg_iteration(InfiniteSquareNetwork(state), env, alg)
    signs, = compute_gauge_fix_gauge(env_conv, env, ScramblingEnvGauge())
    S = normalize.(info.S)
    U, V = fix_relative_phases(info.U, info.V, signs)

    # pretend the singular values matrices are just arbitrary complex tensors
    s = TensorMap.(S)
    if real(scalartype(state)) != scalartype(state)
        # complex network, complex things, required to make the derivatives work...
        s = complex.(s)
    end

    # remove the inverse square roots here to obtain modified corners and edges
    C̃, Ẽ = remove_inverse_roots(env.corners, env.edges, S)
    # which is fine as long as we pretend we did the absorption as the last step in the
    # forward pass, and then explicitly backpropagate through that in the pullback here
    _, absorb_inverse_roots_vjp = rrule_via_ad(config, absorb_inverse_roots, C̃, Ẽ, s)

    # get the projector nullspaces
    UL = left_null.(U)
    VR = right_null.(V)

    # instantiate the variables used in the characteristic equations
    u = map(zip(U, UL)) do (Uc, ULc)
        return zeros(scalartype(Uc), space(ULc, numind(ULc))' ← space(Uc, numind(Uc))')
    end
    v = map(zip(V, VR)) do (Vc, VRc)
        return zeros(scalartype(Vc), space(Vc, 1) ← space(VRc, 1))
    end
    is = sdiag_pow.(s, -1) # also treat them as general complex tensors

    # generate the characteristic equations
    F = generate_halfinfinite_characteristic_equation(is, U, V, UL, VR)

    # check if characteristic equations are actually satisfied
    FS = F(state, C̃, Ẽ, u, s, v)
    F_nrms = norm.(FS)
    any(F_nrms .> 1.0e2 * alg.tol) &&
        @warn "Characteristic equations not satisfied, still using the gradient: $F_nrms"

    # get the partial gradients of the characteristic equations
    _, F_vjp = rrule_via_ad(config, F, state, C̃, Ẽ, u, s, v) # full automatic pullback
    vjp_env(x) = F_vjp(x)[3:end] # environment and SVD pullback
    vjp_state(x) = F_vjp(x)[2] # state pullback

    function leading_boundary_characteristic_pullback((_Δenv, _Δinfo))
        # unpack incoming cotangents
        Δenv = unthunk(_Δenv)

        # apply pullback of the dummy inverse root absorption to get adjoints of modified
        # corners and edges and singular values
        _, ΔC̃, ΔẼ, Δs = absorb_inverse_roots_vjp((Δenv.corners, Δenv.edges))

        # U and V should have zero cotangents
        Δu = zerovector.(u)
        Δv = zerovector.(v)

        # collect cotangents of characteristic equation
        Δy = (ΔC̃, ΔẼ, Δu, Δs, Δv)

        # evaluate the implicit gradient
        Δstate = implicit_gradient(Δy, vjp_env, vjp_state, Δy, gradmode.solver_alg)

        return NoTangent(), ZeroTangent(), Δstate, NoTangent()
    end

    return (env, info), leading_boundary_characteristic_pullback
end

using Manifolds, Manopt

abstract type GradMode{F} end

iterscheme(::GradMode{F}) where {F} = F

"""
    struct GeomSum(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol,
                   verbosity=0, iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode for CTMRG using explicit evaluation of the geometric sum.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct GeomSum{F} <: GradMode{F}
    maxiter::Int
    tol::Real
    verbosity::Int
end
function GeomSum(;
    maxiter=Defaults.fpgrad_maxiter,
    tol=Defaults.fpgrad_tol,
    verbosity=0,
    iterscheme=Defaults.iterscheme,
)
    return GeomSum{iterscheme}(maxiter, tol, verbosity)
end

"""
    struct ManualIter(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol,
                      verbosity=0, iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode for CTMRG using manual iteration to solve the linear problem.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct ManualIter{F} <: GradMode{F}
    maxiter::Int
    tol::Real
    verbosity::Int
end
function ManualIter(;
    maxiter=Defaults.fpgrad_maxiter,
    tol=Defaults.fpgrad_tol,
    verbosity=0,
    iterscheme=Defaults.iterscheme,
)
    return ManualIter{iterscheme}(maxiter, tol, verbosity)
end

"""
    struct LinSolver(; solver=KrylovKit.GMRES(), iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode wrapper around `KrylovKit.LinearSolver` for solving the gradient linear
problem using iterative solvers.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct LinSolver{F} <: GradMode{F}
    solver::KrylovKit.LinearSolver
end
function LinSolver(;
    solver=KrylovKit.BiCGStab(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    iterscheme=Defaults.iterscheme,
)
    return LinSolver{iterscheme}(solver)
end

mutable struct RecordTruncationError
    recorded_values::Vector{Float64}
    RecordTruncationError() = new(Vector{Float64}())
end
function (r::RecordTruncationError)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    pec = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, pec.env_info.truncation_error, i)
end

mutable struct RecordConditionNumber
    recorded_values::Vector{Float64}
    RecordConditionNumber() = new(Vector{Float64}())
end
function (r::RecordConditionNumber)(
    p::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int
)
    pec = Manopt.get_cost_function(get_objective(p))
    return Manopt.record_or_reset!(r, pec.env_info.condition_number, i)
end

mutable struct RecordUnitCellGradientNorm
    recorded_values::Vector{Matrix{Float64}}
    RecordUnitCellGradientNorm() = new(Vector{Matrix{Float64}}())
end
function (r::RecordUnitCellGradientNorm)(
    p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    pec = Manopt.get_cost_function(get_objective(p))
    grad = pec.from_vec(get_gradient(s))
    return Manopt.record_or_reset!(r, norm.(grad.A), i)
end

"""
TODO

Algorithm struct that represent PEPS ground-state optimization using AD.
Set the algorithm to contract the infinite PEPS in `boundary_alg`;
currently only `CTMRGAlgorithm`s are supported. The `optimizer` computes the gradient directions
based on the CTMRG gradient and updates the PEPS parameters. In this optimization,
the CTMRG runs can be started on the converged environments of the previous optimizer
step by setting `reuse_env` to true. Otherwise a random environment is used at each
step. The CTMRG gradient itself is computed using the `gradient_alg` algorithm.
"""
struct PEPSOptimize{G}
    boundary_alg::CTMRGAlgorithm
    optim_alg::Function
    optim_kwargs::NamedTuple
    gradient_alg::G
    reuse_env::Bool
    # reuse_env_tol::Float64  # TODO: add option for reuse tolerance
    symmetrization::Union{Nothing,SymmetrizationStyle}

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
        boundary_alg::CTMRGAlgorithm,
        optim_alg,
        optim_kwargs,
        gradient_alg::G,
        reuse_env,
        symmetrization,
    ) where {G}
        if gradient_alg isa GradMode
            if boundary_alg isa SequentialCTMRG && iterscheme(gradient_alg) === :fixed
                throw(ArgumentError(":sequential and :fixed are not compatible"))
            end
        end
        return new{G}(
            boundary_alg, optim_alg, optim_kwargs, gradient_alg, reuse_env, symmetrization
        )
    end
end
function PEPSOptimize(;
    boundary_alg=Defaults.ctmrg_alg,
    optim_alg=Defaults.optim_alg,
    maxiter=Defaults.optim_maxiter,
    tol=Defaults.optim_tol,
    gradient_alg=Defaults.gradient_alg,
    reuse_env=Defaults.reuse_env,
    symmetrization=nothing,
)
    stopping_criterion = StopAfterIteration(maxiter) | StopWhenGradientNormLess(tol)
    optim_kwargs = (; stopping_criterion, record=Defaults.record_group)
    return PEPSOptimize(
        boundary_alg, optim_alg, optim_kwargs, gradient_alg, reuse_env, symmetrization
    )
end

mutable struct PEPSEnergyCost
    env::CTMRGEnv
    hamiltonian::LocalOperator
    alg::PEPSOptimize
    from_vec::Function
    env_info::NamedTuple
end

# TODO: split this up into f and grad_f
function (pec::PEPSEnergyCost)(peps_vec::Vector)
    peps = pec.from_vec(peps_vec)
    env₀ = reuse_env ? pec.env : CTMRGEnv(randn, scalartype(pec.env), env)

    # compute cost and gradient
    E, gs = withgradient(peps) do ψ
        env′, info = hook_pullback(
            leading_boundary,
            env₀,
            ψ,
            pec.alg.boundary_alg;
            alg_rrule=pec.alg.gradient_alg,
        )
        pec.env_info = info
        E = expectation_value(peps, pec.hamiltonian, env′)
        ignore_derivatives() do
            update!(pec.env, env′)  # Update environment in-place
            isapprox(imag(E), 0; atol=sqrt(eps(real(E)))) ||
                @warn "Expectation value is not real: $E."
        end
        return real(E)
    end
    g = only(gs)  # `withgradient` returns tuple of gradients `gs`

    # symmetrize gradient
    if !isnothing(pec.alg.symmetrization)
        g = symmetrize!(g, pec.alg.symmetrization)
    end

    return E, to_vec(g)[1]
end

# First retract and then symmetrize the resulting PEPS
# (ExponentialRetraction is the default retraction for Euclidean manifolds)
struct SymmetrizeExponentialRetraction <: AbstractRetractionMethod
    symmetrization::SymmetrizationStyle
    from_vec::Function
end

function Manifolds.retract(
    M::AbstractManifold, p, X, t::Number, sr::SymmetrizeExponentialRetraction
)
    q = retract(M, p, X, t, ExponentialRetraction())
    q_symm_peps = symmetrize!(sr.from_vec(q), sr.symmetrization)
    return to_vec(q_symm_peps)
end

"""
TODO
"""
function fixedpoint(
    peps₀::InfinitePEPS{T},
    H,
    alg::PEPSOptimize,
    env₀::CTMRGEnv=CTMRGEnv(peps₀, field(T)^20);
) where {T}
    if scalartype(env₀) <: Real
        env₀ = complex(env₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    # construct cost function struct
    peps₀_vec, from_vec = to_vec(peps₀)
    pec = PEPSEnergyCost(env₀, H, alg, from_vec, (;))
    fld = scalartype(peps₀) <: Real ? Manifolds.ℝ : Manifolds.ℂ
    cost = # TODO #ManifoldCostGradientObjective(pec)
    grad = # TODO

    # optimize
    M = Euclidean(; field=fld)
    retraction_method = if isnothing(alg.symmetrization)
        default_retraction_method(M)
    else
        SymmetrizeExponentialRetraction(alg.symmetrization, from_vec)
    end
    result = alg.optim_alg(
        M,
        cost,
        grad,
        peps₀_vec;
        alg.optim_kwargs...,
        return_state=true,
        retraction_method,
    )

    # extract final result
    peps_final = from_vec(get_solver_result(result))
    env_final = leading_boundary(pec.env, peps_final, alg.boundary_alg)
    E_final = expectation_value(peps_final, H, env_final)
    return peps_final, env_final, E_final, result
end

#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

function _rrule(
    gradmode::GradMode{:diffgauge},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::CTMRGAlgorithm,
)
    envs = leading_boundary(envinit, state, alg)

    function leading_boundary_diffgauge_pullback(Δenvs′)
        Δenvs = unthunk(Δenvs′)

        # find partial gradients of gauge_fixed single CTMRG iteration
        f(A, x) = gauge_fix(x, ctmrg_iteration(A, x, alg)[1])[1]
        _, env_vjp = rrule_via_ad(config, f, state, envs)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[3]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂envs, NoTangent()
    end

    return envs, leading_boundary_diffgauge_pullback
end

# Here f is differentiated from an pre-computed SVD with fixed U, S and V
function _rrule(
    gradmode::GradMode{:fixed},
    config::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    envinit,
    state,
    alg::SimultaneousCTMRG,
)
    @assert !isnothing(alg.projector_alg.svd_alg.rrule_alg)
    envs = leading_boundary(envinit, state, alg)
    envsconv, info = ctmrg_iteration(state, envs, alg)
    envs_fixed, signs = gauge_fix(envs, envsconv)

    # Fix SVD
    Ufix, Vfix = fix_relative_phases(info.U, info.V, signs)
    svd_alg_fixed = SVDAdjoint(;
        fwd_alg=FixedSVD(Ufix, info.S, Vfix), rrule_alg=alg.projector_alg.svd_alg.rrule_alg
    )
    alg_fixed = @set alg.projector_alg.svd_alg = svd_alg_fixed
    alg_fixed = @set alg_fixed.projector_alg.trscheme = notrunc()

    function leading_boundary_fixed_pullback(Δenvs′)
        Δenvs = unthunk(Δenvs′)

        f(A, x) = fix_global_phases(x, ctmrg_iteration(A, x, alg_fixed)[1])
        _, env_vjp = rrule_via_ad(config, f, state, envs_fixed)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[3]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂envs, NoTangent()
    end

    return envs_fixed, leading_boundary_fixed_pullback
end

@doc """
    fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y0, alg)

Compute the gradient of the cost function for CTMRG by solving the following equation:

dx = ∑ₙ (∂f∂x)ⁿ ∂f∂A dA = (1 - ∂f∂x)⁻¹ ∂f∂A dA

where `∂F∂x` is the gradient of the cost function with respect to the PEPS tensors, `∂f∂x`
is the partial gradient of the CTMRG iteration with respect to the environment tensors,
`∂f∂A` is the partial gradient of the CTMRG iteration with respect to the PEPS tensors, and
`y0` is the initial guess for the fixed-point iteration. The function returns the gradient
`dx` of the fixed-point iteration.
"""
fpgrad

# TODO: can we construct an implementation that does not need to evaluate the vjp
# twice if both ∂f∂A and ∂f∂x are needed?
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, _, alg::GeomSum)
    g = ∂F∂x
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

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::ManualIter)
    y = deepcopy(y₀)  # Do not mutate y₀
    dx = ∂f∂A(y)
    ϵ = 1.0
    for i in 1:(alg.maxiter)
        y′ = ∂F∂x + ∂f∂x(y)

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

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::LinSolver)
    y, info = linsolve(∂f∂x, ∂F∂x, y₀, alg.solver, 1, -1)
    if alg.solver.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

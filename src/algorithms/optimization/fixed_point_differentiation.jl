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
    envs, info = leading_boundary(envinit, state, alg)

    function leading_boundary_diffgauge_pullback((Δenvs′, Δinfo))
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

    return (envs, info), leading_boundary_diffgauge_pullback
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
    envs_conv, info = ctmrg_iteration(state, envs, alg)
    envs_fixed, signs = gauge_fix(envs, envs_conv)

    # Fix SVD
    U_fixed, V_fixed = fix_relative_phases(info.U, info.V, signs)
    svd_alg_fixed = SVDAdjoint(;
        fwd_alg=FixedSVD(U_fixed, info.S, V_fixed),
        rrule_alg=alg.projector_alg.svd_alg.rrule_alg,
    )
    alg_fixed = @set alg.projector_alg.svd_alg = svd_alg_fixed
    alg_fixed = @set alg_fixed.projector_alg.trscheme = notrunc()

    function leading_boundary_fixed_pullback((Δenvs′, Δinfo))
        Δenvs = unthunk(Δenvs′)

        f(A, x) = fix_global_phases(x, ctmrg_iteration(A, x, alg_fixed)[1])
        _, env_vjp = rrule_via_ad(config, f, state, envs_fixed)

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[2]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[3]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ZeroTangent(), ∂F∂envs, NoTangent()
    end

    return (envs_fixed, info), leading_boundary_fixed_pullback
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
    y, info = reallinsolve(∂f∂x, ∂F∂x, y₀, alg.solver, 1, -1)
    if alg.solver.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

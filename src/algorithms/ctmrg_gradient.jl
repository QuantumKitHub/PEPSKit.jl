#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

function ctmrg_gradient((peps, envs), H, alg::PEPSOptimize{NaiveAD})
    E, g = withgradient(peps) do ψ
        envs = leading_boundary(ψ, envs.boundary_alg, envs)
        alg.reuse_env && (envs = env′)
        return costfun(ψ, envs, H)
    end

    # AD returns namedtuple as gradient instead of InfinitePEPS
    ∂E∂A = g[1]
    if !(∂E∂A isa InfinitePEPS)
        # TODO: check if `reconstruct` works
        ∂E∂A = InfinitePEPS(∂E∂A.A)
    end
    @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A
end

function ctmrg_gradient((peps, envs), H, alg::PEPSOptimize{GeomSum})
    # find partial gradients of costfun
    env = leading_boundary(peps, alg.boundary_alg, envs)
    E, Egrad = withgradient(costfun, peps, env, H)
    ∂F∂A = InfinitePEPS(Egrad[1]...)
    ∂F∂x = CTMRGEnv(Egrad[2]...)

    # find partial gradients of single ctmrg iteration
    _, envvjp = pullback(peps, envs) do A, x
        return gauge_fix(x, ctmrg_iter(A, x, alg.boundary_alg)[1])
    end
    ∂f∂A(x) = InfinitePEPS(envvjp(x)[1]...)
    ∂f∂x(x) = CTMRGEnv(envvjp(x)[2]...)

    # evaluate the geometric sum
    ∂F∂envs = fpgrad(∂F∂x, ∂f∂x, ∂f∂A, ∂F∂x, alg.gradient_alg)

    return E, ∂F∂A + ∂F∂envs
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

abstract type GradMode end

"""
    NaiveAD <: GradMode

Gradient mode for CTMRG using AD.
"""
struct NaiveAD <: GradMode end

"""
    GeomSum <: GradMode

Gradient mode for CTMRG using explicit evaluation of the geometric sum.
"""
@kwdef struct GeomSum <: GradMode
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    verbosity::Int = 0
end

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
    return dx, ϵ
end

"""
    ManualIter <: GradMode

Gradient mode for CTMRG using manual iteration to solve the linear problem.
"""
@kwdef struct ManualIter <: GradMode
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    verbosity::Int = 0
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::ManualIter)
    y = deepcopy(y₀)  # Do not mutate y₀
    ϵ = 1.0
    for i in 1:(alg.maxiter)
        y′ = ∂F∂x + ∂f∂x(y)

        norma = norm(y.corners[NORTHWEST])
        ϵnew = norm(y′.corners[NORTHWEST] - y.corners[NORTHWEST]) / norma  # Normalize error to get comparable convergence tolerance
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 && @printf(
            "Gradient iter: %3d   ‖Cᵢ₊₁-Cᵢ‖/N: %.2e   Δ‖Cᵢ₊₁-Cᵢ‖/N: %.2e\n", i, ϵnew, Δϵ
        )
        y = y′
        ϵ = ϵnew

        ϵ < alg.tol && break
        if alg.verbosity > 0 && i == alg.maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Cᵢ₊₁-Cᵢ‖ = $ϵ"
        end
    end
    return ∂f∂A(y), ϵ
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::KrylovKit.LinearSolver)
    y, info = linsolve(∂f∂x, ∂F∂x, y₀, alg, 1, -1)
    if alg.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y), info
end

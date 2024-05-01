abstract type GradMode end

"""
    struct NaiveAD <: GradMode

Gradient mode for CTMRG using AD.
"""
struct NaiveAD <: GradMode end

"""
    struct GeomSum <: GradMode

Gradient mode for CTMRG using explicit evaluation of the geometric sum.
"""
@kwdef struct GeomSum <: GradMode
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    verbosity::Int = 0
end

"""
    struct ManualIter <: GradMode

Gradient mode for CTMRG using manual iteration to solve the linear problem.
"""
@kwdef struct ManualIter <: GradMode
    maxiter::Int = Defaults.fpgrad_maxiter
    tol::Real = Defaults.fpgrad_tol
    verbosity::Int = 0
end

"""
    PEPSOptimize{G}(; boundary_alg = CTMRG(), optimizer::OptimKit.OptimizationAlgorithm = LBFGS()
                    reuse_env::Bool = true, gradient_alg::G, verbosity::Int = 0)

Algorithm struct that represent PEPS ground-state optimization using AD.
Set the algorithm to contract the infinite PEPS in `boundary_alg`;
currently only `CTMRG` is supported. The `optimizer` computes the gradient directions
based on the CTMRG gradient and updates the PEPS parameters. In this optimization,
the CTMRG runs can be started on the converged environments of the previous optimizer
step by setting `reuse_env` to true. Otherwise a random environment is used at each
step. The CTMRG gradient itself is computed using the `gradient_alg` algorithm.
Different levels of output verbosity can be activated using `verbosity` (0, 1 or 2).
"""
@kwdef struct PEPSOptimize{G}
    boundary_alg::CTMRG = CTMRG()  # Algorithm to find boundary environment
    optimizer::OptimKit.OptimizationAlgorithm = LBFGS(
        4; maxiter=100, gradtol=1e-4, verbosity=2
    )
    reuse_env::Bool = true  # Reuse environment of previous optimization as initial guess for next
    gradient_alg::G = GeomSum() # Algorithm to solve gradient linear problem
    verbosity::Int = 0
end

"""
    fixedpoint(ψ₀::InfinitePEPS{T}, H, alg::PEPSOptimize, [env₀::CTMRGEnv]) where {T}
    
Optimize `ψ₀` with respect to the Hamiltonian `H` according to the parameters supplied
in `alg`. The initial environment `env₀` serves as an initial guess for the first CTMRG run.
By default, a random initial environment is used.
"""
function fixedpoint(
    ψ₀::InfinitePEPS{T}, H, alg::PEPSOptimize, env₀::CTMRGEnv=CTMRGEnv(ψ₀; Venv=field(T)^20)
) where {T}
    (peps, env), E, ∂E, info = optimize(
        (ψ₀, env₀),
        alg.optimizer;
        retract=my_retract,
        inner=my_inner,
    ) do (peps, envs)
        E, g = withgradient(peps) do ψ
            envs = hook_pullback(
                leading_boundary,
                ψ,
                alg.boundary_alg,
                envs;
                alg_rrule=alg.gradient_alg,
            )
            return costfun(ψ, envs, H)
        end
        # withgradient returns tuple of gradients `g`
        return E, only(g)
    end
    return (; peps, env, E, ∂E, info)
end

# Update PEPS unit cell in non-mutating way
# Note: Both x and η are InfinitePEPS during optimization
function my_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env = deepcopy(x[2])
    return (peps, env), η
end

# Take real valued part of dot product
my_inner(_, η₁, η₂) = real(dot(η₁, η₂))

#=
Evaluating the gradient of the cost function for CTMRG:
- The gradient of the cost function for CTMRG can be computed using automatic differentiation (AD) or explicit evaluation of the geometric sum.
- With AD, the gradient is computed by differentiating the cost function with respect to the PEPS tensors, including computing the environment tensors.
- With explicit evaluation of the geometric sum, the gradient is computed by differentiating the cost function with the environment kept fixed, and then manually adding the gradient contributions from the environments.
=#

function _rrule(
    gradmode::Union{GradMode,KrylovKit.LinearSolver},
    ::RuleConfig,
    ::typeof(MPSKit.leading_boundary),
    state,
    alg::CTMRG,
    envinit,
)
    envs = leading_boundary(state, alg, envinit)

    function leading_boundary_pullback(Δenvs′)
        Δenvs = unthunk(Δenvs′)

        # find partial gradients of gauge_fixed single CTMRG iteration
        # TODO: make this rrule_via_ad so it's zygote-agnostic
        _, env_vjp = pullback(state, envs) do A, x
            return gauge_fix(x, ctmrg_iter(A, x, alg)[1])
        end

        # evaluate the geometric sum
        ∂f∂A(x)::typeof(state) = env_vjp(x)[1]
        ∂f∂x(x)::typeof(envs) = env_vjp(x)[2]
        ∂F∂envs = fpgrad(Δenvs, ∂f∂x, ∂f∂A, Δenvs, gradmode)

        return NoTangent(), ∂F∂envs, NoTangent(), ZeroTangent()
    end

    return envs, leading_boundary_pullback
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
    return ∂f∂A(y)
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::KrylovKit.LinearSolver)
    y, info = linsolve(∂f∂x, ∂F∂x, y₀, alg, 1, -1)
    if alg.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

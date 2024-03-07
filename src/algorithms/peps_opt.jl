abstract type GradMode end

struct NaiveAD <: GradMode end
struct GeomSum <: GradMode end
struct ManualIter <: GradMode end
struct LinSolve <: GradMode end

# Algorithm struct containing parameters for PEPS optimization
@kwdef struct PEPSOptimize{G<:GradMode}
    boundary_alg::CTMRG = CTMRG()  # Algorithm to find boundary environment
    optimizer::OptimKit.OptimizationAlgorithm = LBFGS(
        4; maxiter=100, gradtol=1e-4, verbosity=2
    )
    reuse_env::Bool = true  # Reuse environment of previous optimization as initial guess for next
    fpgrad_tol::Float64 = Defaults.fpgrad_tol  # Convergence tolerance for gradient FP iteration
    fpgrad_maxiter::Int = Defaults.fpgrad_maxiter  # Maximal number  of FP iterations
    verbosity::Int = 0
end

# Find ground-state PEPS and energy
function fixedpoint(
    ψ₀::InfinitePEPS{T}, H, alg::PEPSOptimize, env₀::CTMRGEnv=CTMRGEnv(ψ₀; Venv=field(T)^20)
) where {T}
    (peps, env), E, ∂E, info = optimize(
        x -> ctmrg_gradient(x, H, alg),
        (ψ₀, env₀),
        alg.optimizer;
        retract=my_retract,
        inner=my_inner,
    )
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

# Function returning energy and CTMRG gradient for each optimization step
function ctmrg_gradient(x, H, alg::PEPSOptimize)
    peps, env = x
    E, g = withgradient(ψ -> _ctmrgcostfun!(ψ, env, H, alg), peps)
    ∂E∂A = g[1]  # Extract PEPS derivative from gradient tuple
    if !(typeof(∂E∂A) <: InfinitePEPS)  # NaiveAD returns NamedTuple as gradient instead of InfinitePEPS
        ∂E∂A = InfinitePEPS(∂E∂A.A)
    end
    @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A
end

# Helper function wrapping CTMRG run and cost function with custom adjoint
function _ctmrgcostfun!(peps, env::CTMRGEnv, H, alg::PEPSOptimize)
    env′ = leading_boundary(peps, alg.boundary_alg, env)
    alg.reuse_env && @diffset env = env′
    return costfun(peps, env′, H)
end

# Energy gradient backwards rule (does not apply to NaiveAD gradient mode)
function ChainRulesCore.rrule(
    ::typeof(_ctmrgcostfun!), peps, env::CTMRGEnv, H, alg::PEPSOptimize{G}
) where {G<:Union{GeomSum,ManualIter,LinSolve}}
    env = leading_boundary(peps, alg.boundary_alg, env)
    E, Egrad = withgradient(costfun, peps, env, H)
    ∂F∂A = InfinitePEPS(Egrad[1]...)
    ∂F∂x = CTMRGEnv(Egrad[2]...)
    _, envvjp = pullback(
        (A, x) -> gauge_fix(x, ctmrg_iter(A, x, alg.boundary_alg)[1]), peps, env
    )
    ∂f∂A(x) = InfinitePEPS(envvjp(x)[1]...)
    ∂f∂x(x) = CTMRGEnv(envvjp(x)[2]...)

    function costfun!_pullback(_)
        # TODO: Add interface to choose y₀ and possibly kwargs that are passed to fpgrad?
        # y₀ = CTMRGEnv(peps; Venv=space(env.edges[1])[1])  # This leads to slow convergence in LinSolve and gauge warnings
        dx, = fpgrad(∂F∂x, ∂f∂x, ∂f∂A, ∂F∂x, alg)
        return NoTangent(), ∂F∂A + dx, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return E, costfun!_pullback
end

# Compute energy and energy gradient, by explicitly evaluating geometric series
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, _, alg::PEPSOptimize{GeomSum})
    g = ∂F∂x
    dx = ∂f∂A(g)  # n=0 term: ∂F∂x ∂f∂A
    ϵ = 1.0
    for i in 1:(alg.fpgrad_maxiter)
        g = ∂f∂x(g)
        Σₙ = ∂f∂A(g)
        dx += Σₙ
        ϵnew = norm(Σₙ)  # TODO: normalize this error?
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 &&
            @printf("Gradient iter: %3d   ‖Σₙ‖: %.2e   Δ‖Σₙ‖: %.2e\n", i, ϵnew, Δϵ)
        ϵ = ϵnew

        ϵ < alg.fpgrad_tol && break
        if alg.verbosity > 0 && i == alg.fpgrad_maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Σₙ‖ = $ϵ"
        end
    end
    return dx, ϵ
end

# Manual iteration to solve gradient linear problem
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::PEPSOptimize{ManualIter})
    y = deepcopy(y₀)  # Do not mutate y₀
    ϵ = 1.0
    for i in 1:(alg.fpgrad_maxiter)
        y′ = ∂F∂x + ∂f∂x(y)

        norma = norm(y.corners[NORTHWEST])
        ϵnew = norm(y′.corners[NORTHWEST] - y.corners[NORTHWEST]) / norma  # Normalize error to get comparable convergence tolerance
        Δϵ = ϵ - ϵnew
        alg.verbosity > 1 && @printf(
            "Gradient iter: %3d   ‖Cᵢ₊₁-Cᵢ‖/N: %.2e   Δ‖Cᵢ₊₁-Cᵢ‖/N: %.2e\n", i, ϵnew, Δϵ
        )
        y = y′
        ϵ = ϵnew

        ϵ < alg.fpgrad_tol && break
        if alg.verbosity > 0 && i == alg.fpgrad_maxiter
            @warn "gradient fixed-point iteration reached maximal number of iterations at ‖Cᵢ₊₁-Cᵢ‖ = $ϵ"
        end
    end
    return ∂f∂A(y), ϵ
end

# Use KrylovKit.linsolve to solve gradient linear problem
function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, y₀, alg::PEPSOptimize{LinSolve})
    grad_op(env) = env - ∂f∂x(env)
    y, info = linsolve(
        grad_op, ∂F∂x, y₀; rtol=alg.fpgrad_tol, maxiter=alg.fpgrad_maxiter, krylovdim=20
    )

    if alg.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y), info
end
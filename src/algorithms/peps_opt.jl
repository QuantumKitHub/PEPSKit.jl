abstract type GradMode end

struct NaiveAD <: GradMode end
struct GeomSum <: GradMode end
struct ManualIter <: GradMode end
struct LinSolve <: GradMode end

# Algorithm struct containing parameters for PEPS optimization
# TODO: have an interface for general cost functions? (could merge energyfun and reuse_env)
@kwdef struct PEPSOptimize{G<:GradMode}
    optimizer::OptimKit.OptimizationAlgorithm = LBFGS(
        4; maxiter=100, gradtol=1e-4, verbosity=2
    )
    energyfun::Function = next_neighbor_energy  # Energy function returning real scalar
    reuse_env::Bool = true  # Reuse environment of previous optimization as initial guess for next
    fpgrad_tol::Float64 = Defaults.grad_tol  # Convergence tolerance for gradient FP iteration
    fpgrad_maxiter::Int = Defaults.grad_maxiter  # Maximal number  of FP iterations
    verbosity::Int = 0
end

# Find ground-state PEPS, environment and energy
function groundsearch(
    H, ctmalg::CTMRG, optalg::PEPSOptimize, ψinit::InfinitePEPS, envinit::CTMRGEnv
)
    (peps₀, env₀), E₀, ∂E, info = optimize(
        x -> ctmrg_gradient(x, H, ctmalg, optalg),
        (ψinit, envinit),
        optalg.optimizer;
        inner=my_inner,
        retract=my_retract,
    )
    return (; peps₀, env₀, E₀, ∂E, info)
end

# Function returning energy and CTMRG gradient for each optimization step
function ctmrg_gradient(x, H, ctmalg::CTMRG, optalg::PEPSOptimize)
    peps, env = x
    cfun = optalg.reuse_env ? costfun! : costfun
    E = cfun(peps, env, H, ctmalg, optalg)
    ∂E∂A = gradient(cfun, peps, env, H, ctmalg, optalg)[1]
    if !(typeof(∂E∂A) <: InfinitePEPS)  # NaiveAD returns NamedTuple as gradient instead of InfinitePEPS
        ∂E∂A = InfinitePEPS(∂E∂A.A)
    end
    @assert !isnan(norm(∂E∂A))
    return E, ∂E∂A
end

# Energy cost function with proper backwards rule depending only on final CTMRG fixed-point
# Mutates environment to reuse previous environments in optimization
function costfun!(peps, env, H, ctmalg::CTMRG, optalg::PEPSOptimize)
    env′ = leading_boundary(peps, ctmalg, env)
    @diffset env = env′
    return optalg.energyfun(peps, env′, H)
end

# Non-mutating version, recomputing environment from random initial guess in every optimization step
function costfun(peps, env, H, ctmalg::CTMRG, optalg::PEPSOptimize)
    env′ = deepcopy(env)  # Create copy to make non-mutating
    return costfun!(peps, env′, H, ctmalg, optalg)
end

# Energy gradient backwards rule (does not apply to NaiveAD gradient mode)
function ChainRulesCore.rrule(
    ::typeof(costfun!), peps, env, H, ctmalg::CTMRG, optalg::PEPSOptimize{G}
) where {G<:Union{GeomSum,ManualIter,LinSolve}}
    env = leading_boundary(peps, ctmalg, env)
    E, Egrad = withgradient(optalg.energyfun, peps, env, H)
    ∂F∂A = InfinitePEPS(Egrad[1]...)
    ∂F∂x = CTMRGEnv(Egrad[2]...)
    _, envvjp = pullback((A, x) -> gauge_fix(x, ctmrg_iter(A, x, ctmalg)[1]), peps, env)
    ∂f∂A(x) = InfinitePEPS(envvjp(x)[1]...)
    ∂f∂x(x) = CTMRGEnv(envvjp(x)[2]...)

    function costfun!_pullback(_)
        # TODO: Add interface to choose y₀ and possibly kwargs that are passed to fpgrad?
        # y₀ = CTMRGEnv(peps; Venv=space(env.edges[1])[1])  # This leads to slow convergence in LinSolve and gauge warnings
        dx, = fpgrad(∂F∂x, ∂f∂x, ∂f∂A, ∂F∂x, optalg)
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

# Contraction of CTMRGEnv and PEPS tensors with open physical bonds
function one_site_rho(peps::InfinitePEPS, env::CTMRGEnv{C,T}) where {C,T}
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        @tensor ρ[-1; -2] :=
            env.corners[NORTHWEST, r, c][1; 2] *
            env.edges[NORTH, r, c][2 3 4; 5] *
            env.corners[NORTHEAST, r, c][5; 6] *
            env.edges[EAST, r, c][6 7 8; 9] *
            env.corners[SOUTHEAST, r, c][9; 10] *
            env.edges[SOUTH, r, c][10 11 12; 13] *
            env.corners[SOUTHWEST, r, c][13; 14] *
            env.edges[WEST, r, c][14 15 16; 1] *
            peps[r, c][-1; 3 7 11 15] *
            conj(peps[r, c][-2; 4 8 12 16])
    end
end

# Horizontally extended contraction of CTMRGEnv and PEPS tensors with open physical bonds
function two_site_rho(peps::InfinitePEPS, env::CTMRGEnv{C,T}) where {C,T}
    return map(Iterators.product(axes(env.corners, 2), axes(env.corners, 3))) do (r, c)
        cnext = _next(c, size(peps, 2))
        @tensor ρ[-11 -20; -12 -18] :=
            env.corners[NORTHWEST, r, c][1; 3] *
            env.edges[NORTH, r, c][3 5 8; 13] *
            env.edges[NORTH, r, cnext][13 16 22; 23] *
            env.corners[NORTHEAST, r, cnext][23; 24] *
            env.edges[EAST, r, cnext][24 25 26; 27] *
            env.corners[SOUTHEAST, r, cnext][27; 28] *
            env.edges[SOUTH, r, cnext][28 17 21; 14] *
            env.edges[SOUTH, r, c][14 6 10; 4] *
            env.corners[SOUTHWEST, r, c][4; 2] *
            env.edges[WEST, r, c][2 7 9; 1] *
            peps[r, c][-12; 5 15 6 7] *
            conj(peps[r, c][-11; 8 19 10 9]) *
            peps[r, cnext][-18; 16 25 17 15] *
            conj(peps[r, cnext][-20; 22 26 21 19])
    end
end

# 1-site operator expectation values on unit cell
function MPSKit.expectation_value(
    peps::InfinitePEPS, env::CTMRGEnv, op::AbstractTensorMap{S,1,1}
) where {S<:ElementarySpace}
    result = similar(peps.A, eltype(op))
    ρ = one_site_rho(peps, env)

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        o = @tensor ρ[r, c][1; 2] * op[1; 2]
        n = @tensor ρ[r, c][1; 1]
        @diffset result[r, c] = o / n
    end

    return result
end

# 2-site operator expectation values on unit cell
function MPSKit.expectation_value(
    peps::InfinitePEPS, env::CTMRGEnv, op::AbstractTensorMap{S,2,2}
) where {S<:ElementarySpace}
    result = similar(peps.A, eltype(op))
    ρ = two_site_rho(peps, env)

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        o = @tensor ρ[r, c][1 2; 3 4] * op[1 2; 3 4]
        n = @tensor ρ[r, c][1 2; 1 2]
        @diffset result[r, c] = o / n
    end

    return result
end

# ⟨H⟩ from vertical and horizontal next-neighbor contributions
function next_neighbor_energy(
    peps::InfinitePEPS, env::CTMRGEnv, H::AbstractTensorMap{S,2,2}
) where {S<:ElementarySpace}
    Eh = sum(expectation_value(peps, env, H))
    Ev = sum(expectation_value(rotl90(peps), rotl90(env), H))
    return real(Eh + Ev)
end

# Update PEPS unit cell in non-mutating way
# Note: Both x (Y, X) and η are InfinitePEPS during optimization
function my_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env = deepcopy(x[2])
    return (peps, env), η
end

# Take real valued part of dot product
my_inner(_, η₁, η₂) = real(dot(η₁, η₂))

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

"""
    struct EigSolver(; solver=KrylovKit.Arnoldi(), iterscheme=Defaults.iterscheme) <: GradMode{iterscheme}

Gradient mode wrapper around `KrylovKit.KrylovAlgorithm` for solving the gradient linear
problem as an eigenvalue problem.

With `iterscheme` the style of CTMRG iteration which is being differentiated can be chosen.
If set to `:fixed`, the differentiated CTMRG iteration is assumed to have a pre-computed
SVD of the environments with a fixed set of gauges. Alternatively, if set to `:diffgauge`,
the differentiated iteration consists of a CTMRG iteration and a subsequent gauge fixing step,
such that `gauge_fix` will also be differentiated everytime a CTMRG derivative is computed.
"""
struct EigSolver{F} <: GradMode{F}
    solver::KrylovKit.KrylovAlgorithm
end
function EigSolver(;
    solver=KrylovKit.Arnoldi(; maxiter=Defaults.fpgrad_maxiter, tol=Defaults.fpgrad_tol),
    iterscheme=Defaults.iterscheme,
)
    return EigSolver{iterscheme}(solver)
end

"""
    PEPSOptimize{G}(; boundary_alg=Defaults.ctmrg_alg, optimizer::OptimKit.OptimizationAlgorithm=Defaults.optimizer
                    reuse_env::Bool=true, gradient_alg::G=Defaults.gradient_alg)

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
    optimizer::OptimKit.OptimizationAlgorithm
    reuse_env::Bool
    gradient_alg::G

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
        boundary_alg::CTMRGAlgorithm,
        optimizer,
        reuse_env,
        gradient_alg::G,
    ) where {G}
        if gradient_alg isa GradMode
            if boundary_alg isa SequentialCTMRG && iterscheme(gradient_alg) === :fixed
                throw(ArgumentError(":sequential and :fixed are not compatible"))
            end
        end
        return new{G}(boundary_alg, optimizer, reuse_env, gradient_alg)
    end
end
function PEPSOptimize(;
    boundary_alg=Defaults.ctmrg_alg,
    optimizer=Defaults.optimizer,
    reuse_env=Defaults.reuse_env,
    gradient_alg=Defaults.gradient_alg,
)
    return PEPSOptimize(boundary_alg, optimizer, reuse_env, gradient_alg)
end

"""
    fixedpoint(ψ₀::InfinitePEPS{T}, H, alg::PEPSOptimize, [env₀::CTMRGEnv];
               finalize!=OptimKit._finalize!, symmetrization=nothing) where {T}
    
Optimize `ψ₀` with respect to the Hamiltonian `H` according to the parameters supplied
in `alg`. The initial environment `env₀` serves as an initial guess for the first CTMRG run.
By default, a random initial environment is used.

The `finalize!` kwarg can be used to insert a function call after each optimization step
by utilizing the `finalize!` kwarg of `OptimKit.optimize`.
The function maps `(peps, envs), f, g = finalize!((peps, envs), f, g, numiter)`.
The `symmetrization` kwarg accepts `nothing` or a `SymmetrizationStyle`, in which case the
PEPS and PEPS gradient are symmetrized after each optimization iteration. Note that this
requires a symmmetric `ψ₀` and `env₀` to converge properly.

The function returns a `NamedTuple` which contains the following entries:
- `peps`: final `InfinitePEPS`
- `env`: `CTMRGEnv` corresponding to the final PEPS
- `E`: final energy
- `E_history`: convergence history of the energy function
- `grad`: final energy gradient
- `gradnorm_history`: convergence history of the energy gradient norms
- `numfg`: total number of calls to the energy function
"""
function fixedpoint(
    ψ₀::InfinitePEPS{F},
    H,
    alg::PEPSOptimize,
    env₀::CTMRGEnv=CTMRGEnv(ψ₀, field(F)^20);
    (finalize!)=OptimKit._finalize!,
    symmetrization=nothing,
) where {F}
    if isnothing(symmetrization)
        retract = peps_retract
    else
        retract, symm_finalize! = symmetrize_retract_and_finalize!(symmetrization)
        fin! = finalize!  # Previous finalize!
        finalize! = (x, f, g, numiter) -> fin!(symm_finalize!(x, f, g, numiter)..., numiter)
    end

    if scalartype(env₀) <: Real
        env₀ = complex(env₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    (peps, env), E, ∂E, numfg, convhistory = optimize(
        (ψ₀, env₀), alg.optimizer; retract, inner=real_inner, finalize!
    ) do (peps, envs)
        E, gs = withgradient(peps) do ψ
            envs´ = hook_pullback(
                leading_boundary,
                envs,
                ψ,
                alg.boundary_alg;
                alg_rrule=alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && update!(envs, envs´)
            end
            return costfun(ψ, envs´, H)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        return E, g
    end

    return (;
        peps,
        env,
        E,
        E_history=convhistory[:, 1],
        grad=∂E,
        gradnorm_history=convhistory[:, 2],
        numfg,
    )
end

# Update PEPS unit cell in non-mutating way
# Note: Both x and η are InfinitePEPS during optimization
function peps_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env = deepcopy(x[2])
    return (peps, env), η
end

# Take real valued part of dot product
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

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
    y, info = reallinsolve(∂f∂x, ∂F∂x, y₀, alg.solver, 1, -1)
    if alg.solver.verbosity > 0 && info.converged != 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end

    return ∂f∂A(y)
end

function fpgrad(∂F∂x, ∂f∂x, ∂f∂A, x₀, alg::EigSolver)
    function f(X)
        y = ∂f∂x(X[1])
        return (y + X[2] * ∂F∂x, X[2])
    end
    X₀ = (x₀, one(scalartype(x₀)))
    vals, vecs, info = eigsolve(f, X₀, 1, :LM, alg.solver)
    if alg.solver.verbosity > 0 && info.converged < 1
        @warn("gradient fixed-point iteration reached maximal number of iterations:", info)
    end
    if norm(vecs[1][2]) < 1e-8 # TODO: figure out what this actually means...
        @warn "fpgrad using Arnoldi failed: λ = $(vecs[1][3])"
    end    
    y = scale(vecs[1][1], 1 / vecs[1][2])

    return ∂f∂A(y)
end

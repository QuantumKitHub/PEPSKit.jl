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
    gradient_alg::G
    optimizer::OptimKit.OptimizationAlgorithm
    reuse_env::Bool
    symmetrization::Union{Nothing,SymmetrizationStyle}

    function PEPSOptimize(  # Inner constructor to prohibit illegal setting combinations
        boundary_alg::CTMRGAlgorithm,
        gradient_alg::G,
        optimizer,
        reuse_env,
        symmetrization,
    ) where {G}
        if gradient_alg isa GradMode
            if boundary_alg isa SequentialCTMRG && iterscheme(gradient_alg) === :fixed
                throw(ArgumentError(":sequential and :fixed are not compatible"))
            end
        end
        return new{G}(boundary_alg, gradient_alg, optimizer, reuse_env, symmetrization)
    end
end
function PEPSOptimize(;
    boundary_alg=Defaults.ctmrg_alg,
    gradient_alg=Defaults.gradient_alg,
    optimizer=Defaults.optimizer,
    reuse_env=Defaults.reuse_env,
    symmetrization=nothing,
)
    return PEPSOptimize(boundary_alg, gradient_alg, optimizer, reuse_env, symmetrization)
end

"""

    fixedpoint(operator, peps₀::InfinitePEPS{F}, [env₀::CTMRGEnv]; kwargs...)
    fixedpoint(operator, peps₀::InfinitePEPS{T}, alg::PEPSOptimize, [env₀::CTMRGEnv];
               finalize!=OptimKit._finalize!, symmetrization=nothing) where {T}
    
Optimize `peps₀` with respect to the `operator` according to the parameters supplied
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
    operator, peps₀::InfinitePEPS{F}, env₀::CTMRGEnv=CTMRGEnv(peps₀, field(F)^20); kwargs...
) where {F}
    alg = fixedpoint_selector(; kwargs...) # TODO: implement fixedpoint_selector
    return fixedpoint(operator, peps₀, env₀, alg)
end
function fixedpoint(
    operator,
    peps₀::InfinitePEPS,
    env₀::CTMRGEnv,
    alg::PEPSOptimize;
    (finalize!)=OptimKit._finalize!,
)
    if isnothing(alg.symmetrization)
        retract = peps_retract
    else
        retract, symm_finalize! = symmetrize_retract_and_finalize!(alg.symmetrization)
        fin! = finalize!  # Previous finalize!
        finalize! = (x, f, g, numiter) -> fin!(symm_finalize!(x, f, g, numiter)..., numiter)
    end

    if scalartype(env₀) <: Real && iterscheme(alg.gradient_alg) == :fixed
        env₀ = complex(env₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    (peps, env), E, ∂E, numfg, convhistory = optimize(
        (peps₀, env₀), alg.optimizer; retract, inner=real_inner, finalize!
    ) do (peps, envs)
        E, gs = withgradient(peps) do ψ
            envs´, = hook_pullback(
                leading_boundary,
                envs,
                ψ,
                alg.boundary_alg;
                alg_rrule=alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && update!(envs, envs´)
            end
            return cost_function(ψ, envs´, operator)
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

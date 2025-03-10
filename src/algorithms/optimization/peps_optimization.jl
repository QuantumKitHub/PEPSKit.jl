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
    fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv; kwargs...)
    fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv, alg::PEPSOptimize;
               finalize!=OptimKit._finalize!)
    
Find the fixed point of `operator` (i.e. the ground state) starting from `peps₀` according
to the optimization parameters supplied in `alg`. The initial environment `env₀` serves as
an initial guess for the first CTMRG run. By default, a random initial environment is used.

The `finalize!` kwarg can be used to insert a function call after each optimization step
by utilizing the `finalize!` kwarg of `OptimKit.optimize`.
The function maps `(peps, env), f, g = finalize!((peps, env), f, g, numiter)`.
The `symmetrization` kwarg accepts `nothing` or a `SymmetrizationStyle`, in which case the
PEPS and PEPS gradient are symmetrized after each optimization iteration. Note that this
requires a symmmetric `peps₀` and `env₀` to converge properly.

The function returns the final PEPS, CTMRG environment and cost value, as well as an
information `NamedTuple` which contains the following entries:
- `last_gradient`: last gradient of the cost function
- `fg_evaluations`: number of evaluations of the cost and gradient function
- `costs`: history of cost values
- `gradnorms`: history of gradient norms
- `truncation_errors`: history of truncation errors of the boundary algorithm
- `condition_numbers`: history of condition numbers of the CTMRG environments
- `gradnorms_unitcell`: history of gradient norms for each respective unit cell entry
- `times`: history of times each optimization step took
"""
function fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv; kwargs...)
    throw(error("method not yet implemented"))
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
    # setup retract and finalize! for symmetrization
    if isnothing(alg.symmetrization)
        retract = peps_retract
    else
        retract, symm_finalize! = symmetrize_retract_and_finalize!(alg.symmetrization)
        fin! = finalize!  # Previous finalize!
        finalize! = (x, f, g, numiter) -> fin!(symm_finalize!(x, f, g, numiter)..., numiter)
    end

    # check realness compatibility
    if scalartype(env₀) <: Real && iterscheme(alg.gradient_alg) == :fixed
        env₀ = complex(env₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps₀)), real(scalartype(env₀)))
    truncation_errors = Vector{T}()
    condition_numbers = Vector{T}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # normalize the initial guess
    peps₀ = peps_normalize(peps₀)

    # optimize operator cost function
    (peps_final, env_final), cost, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀),
        alg.optimizer;
        retract,
        inner=real_inner,
        finalize!,
        (transport!)=(peps_transport!),
    ) do (peps, env)
        start_time = time_ns()
        E, gs = withgradient(peps) do ψ
            env′, info = hook_pullback(
                leading_boundary,
                env,
                ψ,
                alg.boundary_alg;
                alg_rrule=alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && update!(env, env′)
                push!(truncation_errors, info.truncation_error)
                push!(condition_numbers, info.condition_number)
            end
            return cost_function(ψ, env′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        push!(gradnorms_unitcell, norm.(g.A))
        push!(times, (time_ns() - start_time) * 1e-9)
        return E, g
    end

    info = (
        last_gradient=∂cost,
        fg_evaluations=numfg,
        costs=convergence_history[:, 1],
        gradnorms=convergence_history[:, 2],
        truncation_errors,
        condition_numbers,
        gradnorms_unitcell,
        times,
    )
    return peps_final, env_final, cost, info
end

"""
    peps_normalize(A::InfinitePEPS)

Normalize the individual tensors in the unit cell of an `InfinitePEPS` such that they each
have unit Euclidean norm.
"""
function peps_normalize(A::InfinitePEPS)
    normalized_tensors = map(unitcell(A)) do t
        return t / norm(t)
    end
    return InfinitePEPS(normalized_tensors)
end

"""
    peps_retract(x, η, α)

Performs a norm-preserving retraction of an infinite PEPS `A = x[1]` along `η` with step
size `α`, giving a new PEPS `A´`,
```math
A' \\leftarrow \\cos \\left( α \\frac{||η||}{||A||} \\right) A + \\sin \\left( α \\frac{||η||}{||A||} \\right) ||A|| \\frac{η}{||η||},
```
and corresponding directional derivative `ξ`,
```math
ξ = \\cos \\left( α \\frac{||η||}{||A||} \\right) η - \\sin \\left( α \\frac{||η||}{||A||} \\right) ||η|| \\frac{A}{||A||},
```
such that ``\\langle A', ξ \\rangle = 0`` and ``||A'|| = ||A||``.
"""
function peps_retract(x, η, α)
    peps = x[1]
    norms_peps = norm.(peps.A)
    norms_η = norm.(η.A)

    peps´ = similar(x[1])
    peps´.A .=
        cos.(α .* norms_η ./ norms_peps) .* peps.A .+
        sin.(α .* norms_η ./ norms_peps) .* norms_peps .* η.A ./ norms_η

    env = deepcopy(x[2])

    ξ = similar(η)
    ξ.A .=
        cos.(α .* norms_η ./ norms_peps) .* η.A .-
        sin.(α .* norms_η ./ norms_peps) .* norms_η .* peps.A ./ norms_peps

    return (peps´, env), ξ
end

"""
    peps_transport!(ξ, x, η, α, x′)

Transports a direction at `A = x[1]` to a valid direction at `A´ = x´[1]` corresponding to
the norm-preserving retraction of `A` along `η` with step size `α`. In particular, starting
from a direction `η` of the form
```math
ξ = \\left\\langle \\frac{η}{||η||}, ξ \\right\\rangle \\frac{η}{||η||} + Δξ
```
where ``\\langle Δξ, A \\rangle = \\langle Δξ, η \\rangle = 0``, it returns
```math
ξ(α) = \\left\\langle \\frac{η}{||η||}, ξ \\right \\rangle \\left( \\cos \\left( α \\frac{||η||}{||A||} \\right) \\frac{η}{||η||} - \\sin( \\left( α \\frac{||η||}{||A||} \\right) \\frac{A}{||A||} \\right) + Δξ
```
such that ``||ξ(α)|| = ||ξ||, \\langle A', ξ(α) \\rangle = 0``.
"""
function peps_transport!(ξ, x, η, α, x´)
    peps = x[1]
    norms_peps = norm.(peps.A)

    norms_η = norm.(η.A)
    normalized_η = η.A ./ norms_η
    overlaps_η_ξ = inner.(normalized_η, ξ.A)

    # isolate the orthogonal component
    Δξ = ξ.A .- overlaps_η_ξ .* normalized_η

    # keep orthogonal component fixed, modify the rest by the proper directional derivative
    ξ.A .=
        overlaps_η_ξ .* (
            cos.(α .* norms_η ./ norms_peps) .* normalized_η .-
            sin.(α .* norms_η ./ norms_peps) .* peps.A ./ norms_peps
        ) .+ Δξ

    return ξ
end

# Take real valued part of dot product
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))

"""
    symmetrize_retract_and_finalize!(symm::SymmetrizationStyle)

Return the `retract` and `finalize!` function for symmetrizing the `peps` and `grad` tensors.
"""
function symmetrize_retract_and_finalize!(symm::SymmetrizationStyle)
    finf = function symmetrize_finalize!((peps, env), E, grad, _)
        grad_symm = symmetrize!(grad, symm)
        return (peps, env), E, grad_symm
    end
    retf = function symmetrize_retract((peps, env), η, α)
        peps_symm = deepcopy(peps)
        peps_symm.A .+= η.A .* α
        env′ = deepcopy(env)
        symmetrize!(peps_symm, symm)
        return (peps_symm, env′), η
    end
    return retf, finf
end

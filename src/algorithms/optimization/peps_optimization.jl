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
    fixedpoint(operator, peps₀::InfinitePEPS, env₀::CTMRGEnv; kwargs...) # TODO
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
    χenv = maximum(env₀.corners) do corner # extract maximal environment dimension
        return dim(space(corner, 1))
    end
    alg, finalize! = select_fixedpoint_algorithm(χenv; kwargs...)
    return fixedpoint(operator, peps₀, env₀, alg; finalize!)
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

    # optimize operator cost function
    (peps_final, env_final), cost, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env₀), alg.optimizer; retract, inner=real_inner, finalize!
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

    info = (;
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
    function select_fixedpoint_algorithm(
        χenv::Int;
        tol=Defaults.optimizer_tol,
        verbosity=1,
        boundary_alg,
        gradient_alg,
        optimization_alg,
    )

Parse optimization keyword arguments on to the corresponding algorithm structs and return
a final `PEPSOptimize` to be used in `fixedpoint`. For a description of the keyword
arguments, see [`fixedpoint`](@ref).
"""
function select_fixedpoint_algorithm(
    χenv::Int;
    tol=Defaults.optimizer_tol, # top-level tolerance
    verbosity=1, # top-level verbosity
    boundary_alg=nothing,
    gradient_alg=nothing,
    optimization_alg=nothing,
)
    # top-level verbosity
    if verbosity ≤ 0 # disable output
        boundary_verbosity = -1
        gradient_verbosity = -1
        optimizer_verbosity = -1
    elseif verbosity == 1 # output only optimization steps and degeneracy warnings
        boundary_verbosity = -1
        gradient_verbosity = -1
        optimizer_verbosity = 3
    elseif verbosity == 2 # output optimization and boundary information
        boundary_verbosity = 2
        gradient_verbosity = -1
        optimizer_verbosity = 3
    elseif verbosity == 3 # verbose debug output
        boundary_verbosity = 3
        gradient_verbosity = 3
        optimizer_verbosity = 3
    end

    # parse boundary algorithm
    boundary_algorithm = if boundary_alg isa Union{SimultaneousCTMRG,SequentialCTMRG}
        boundary_alg
    elseif boundary_alg isa NamedTuple
        boundary_kwargs = (;
            alg=Defaults.ctmrg_alg_type,
            tol=1e-4tol,
            maxiter=Defaults.ctmrg_miniter,
            miniter=Defaults.ctmrg_maxiter,
            verbosity=boundary_verbosity,
            trscheme=Defaults.trscheme,
            svd_alg=Defaults.svd_fwd_alg,
            svd_rrule_alg=typeof(Defaults.svd_rrule_alg),
            projector_alg=Defaults.projector_alg_type,
            boundary_alg..., # replaces all specified kwargs
        )
        select_leading_boundary_algorithm(χenv; boundary_kwargs...)
    else
        throw(ArgumentError("unknown boundary algorithm: $boundary_alg"))
    end

    # parse fixed-point gradient algorithm
    gradient_algorithm = if gradient_alg isa GradMode
        gradient_alg
    elseif gradient_alg isa NamedTuple
        gradient_kwargs = (;
            tol=1e-2tol,
            maxiter=Defaults.gradient_alg_maxiter,
            alg=typeof(Defaults.gradient_alg),
            verbosity=gradient_verbosity,
            iterscheme=Defaults.gradient_alg_iterscheme,
            gradient_alg..., # replaces all specified kwargs
        )
        if gradient_kwargs.alg <: Union{GeomSum,ManIter}
            gradient_alg_type(;
                tol=gradient_kwargs.tol,
                maxiter=gradient_kwargs.maxiter,
                verbosity=gradient_kwargs.verbosity,
                iterscheme=gradient_kwargs.iterscheme,
            )
        elseif gradient_kwargs.alg <: LinSolver
            solver = Defaults.gradient_linsolver.solver
            @reset solver.maxiter = gradient_kwargs.maxiter
            @reset solver.tol = gradient_kwargs.tol
            @reset solver.verbosity = gradient_kwargs.verbosity
            LinSolver(; solver, iterscheme=gradient_kwargs.iterscheme)
        elseif gradient_kwargs.alg <: EigSolver
            solver = Defaults.gradient_eigsolver.solver
            @reset solver.maxiter = gradient_kwargs.maxiter
            @reset solver.tol = gradient_kwargs.tol
            @reset solver.verbosity = gradient_kwargs.verbosity
            EigSolver(; solver, iterscheme=gradient_kwargs.iterscheme)
        end
    else
        throw(ArgumentError("unknown gradient algorithm: $gradient_alg"))
    end

    # construct final PEPSOptimize optimization algorithm
    optimization_algorithm = if optimizer_alg isa OptimKit.OptimizationAlgorithm
        optimization_alg
    elseif optimization_algorithm isa NamedTuple
        optimization_kwargs = (;
            tol=tol,
            maxiter=Defaults.optimizer_maxiter,
            lbfgs_memory=Defaults.lbfgs_memory,
            reuse_env=Defaults.reuse_env,
            symmetrization=nothing,
            (finalize!)=OptimKit._finalize!,
            optimization_alg..., # replaces all specified kwargs
        )
        optimizer = LBFGS(
            lbfgs_memory;
            gradtol=optimization_kwargs.tol,
            maxiter=optimization_kwargs.maxiter,
            verbosity=optimizer_verbosity,
        )
        PEPSOptimize(
            boundary_algorithm, gradient_algorithm, optimizer, reuse_env, symmetrization
        )
    else
        throw(ArgumentError("unknown optimization algorithm: $optimization_alg"))
    end

    return optimization_algorithm, finalize!
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

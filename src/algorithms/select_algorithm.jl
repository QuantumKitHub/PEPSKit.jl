"""
    select_algorithm(::typeof(fixedpoint), env₀::CTMRGEnv; kwargs...)

Parse optimization keyword arguments on to the corresponding algorithm structs and return
a final `PEPSOptimize` to be used in `fixedpoint`. For a description of the keyword
arguments, see [`fixedpoint`](@ref).
"""
function select_algorithm(
    ::typeof(fixedpoint),
    env₀::CTMRGEnv;
    tol=Defaults.optimizer_tol, # top-level tolerance
    verbosity=2, # top-level verbosity
    boundary_alg=(;),
    gradient_alg=(;),
    optimization_alg=(;),
    (finalize!)=OptimKit._finalize!,
)

    # top-level verbosity
    if verbosity ≤ 0 # disable output
        boundary_verbosity = -1
        gradient_verbosity = -1
        optimizer_verbosity = -1
    elseif verbosity == 1 # output only optimization steps and degeneracy warnings
        boundary_verbosity = -1
        gradient_verbosity = 1
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
    boundary_algorithm = if boundary_alg isa CTMRGAlgorithm
        boundary_alg
    elseif boundary_alg isa NamedTuple
        select_algorithm(
            leading_boundary,
            env₀;
            tol=1e-4tol,
            verbosity=boundary_verbosity,
            svd_alg=(; rrule_alg=(; tol=1e-3tol)),
            boundary_alg...,
        )
    else
        throw(ArgumentError("unknown boundary algorithm: $boundary_alg"))
    end

    # parse fixed-point gradient algorithm
    gradient_algorithm = if gradient_alg isa GradMode
        gradient_alg
    elseif gradient_alg isa NamedTuple
        select_algorithm(GradMode; gradient_kwargs...)
    else
        throw(ArgumentError("unknown gradient algorithm: $gradient_alg"))
    end

    # construct final PEPSOptimize optimization algorithm
    optimization_algorithm = if optimization_alg isa PEPSOptimize
        optimization_alg
    elseif optimization_alg isa NamedTuple
        optimization_kwargs = (;
            alg=Defaults.optimizer_alg,
            tol=tol,
            maxiter=Defaults.optimizer_maxiter,
            lbfgs_memory=Defaults.lbfgs_memory,
            reuse_env=Defaults.reuse_env,
            symmetrization=nothing,
            optimization_alg..., # replaces all specified kwargs
        )
        optimizer = LBFGS(
            optimization_kwargs.lbfgs_memory;
            gradtol=optimization_kwargs.tol,
            maxiter=optimization_kwargs.maxiter,
            verbosity=optimizer_verbosity,
        )
        PEPSOptimize(
            boundary_algorithm,
            gradient_algorithm,
            optimizer,
            optimization_kwargs.reuse_env,
            optimization_kwargs.symmetrization,
        )
    else
        throw(ArgumentError("unknown optimization algorithm: $optimization_alg"))
    end

    return optimization_algorithm, finalize!
end

"""
    select_algorithm(::typeof(leading_boundary), env₀::CTMRGEnv; kwargs...) -> CTMRGAlgorithm

Parse and standardize CTMRG keyword arguments, and bundle them into a `CTMRGAlgorithm` struct,
which is passed on to [`leading_boundary`](@ref). See [`leading_boundary`](@ref) for a
description of all keyword arguments.
"""
function select_algorithm(
    ::typeof(leading_boundary),
    env₀::CTMRGEnv;
    alg=Defaults.ctmrg_alg,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    trscheme=(; alg=Defaults.trscheme),
    svd_alg=(;),
    projector_alg=Defaults.projector_alg, # only allows for Symbol/Type{ProjectorAlgorithm} to expose projector kwargs
)
    # extract maximal environment dimensions
    χenv = maximum(env₀.corners) do corner
        return dim(space(corner, 1))
    end
    krylovdim = round(Int, Defaults.krylovdim_factor * χenv)

    # replace symbol with projector alg type
    alg_type = if alg isa Symbol
        projector_symbols[alg]
    else
        alg
    end

    # parse SVD forward & rrule algorithm 
    svd_algorithm = if svd_alg isa SVDAdjoint
        svd_alg
    elseif svd_alg isa NamedTuple
        alg′ = select_algorithm(
            SVDAdjoint; rrule_alg=(; tol=1e1tol, verbosity=verbosity - 2), svd_alg...
        )
        if typeof(alg′.rrule_alg) <: Union{<:GMRES,<:Arnoldi}
            @reset alg′.rrule_alg.krylovdim = krylovdim
        end
    else
        throw(ArgumentError("unknown SVD algorithm: $svd_alg"))
    end

    # parse CTMRG projector algorithm
    projector_algorithm = select_algorithm(
        ProjectorAlgorithm; alg=projector_alg, svd_alg=svd_algorithm, trscheme, verbosity
    )

    return alg_type(tol, maxiter, miniter, verbosity, projector_algorithm)
end

function select_algorithm(
    ::Type{ProjectorAlgorithm};
    alg=Defaults.projector_alg,
    svd_alg=(;),
    trscheme=(;),
    verbosity=Defaults.projector_verbosity,
)
    # replace symbol with projector alg type
    alg_type = if alg isa Symbol
        projector_symbols[alg]
    else
        alg
    end

    # parse SVD forward & rrule algorithm
    svd_algorithm = if svd_alg isa SVDAdjoint
        svd_alg
    elseif svd_alg isa NamedTuple
        select_algorithm(SVDAdjoint; svd_alg...)
    else
        throw(ArgumentError("unknown SVD algorithm: $svd_alg"))
    end

    # parse truncation scheme
    truncation_scheme = if trscheme isa TruncationScheme
        trscheme
    elseif trscheme isa NamedTuple
        select_algorithm(TruncationScheme; trscheme...)
    else
        throw(ArgumentError("unknown truncation scheme: $trscheme"))
    end

    return alg_type(svd_algorithm, truncation_scheme, verbosity)
end

function select_algorithm(
    ::Type{GradMode};
    alg=Defaults.gradient_alg,
    tol=Defaults.gradient_tol,
    maxiter=Defaults.gradient_maxiter,
    verbosity=Defaults.gradient_verbosity,
    iterscheme=Defaults.gradient_iterscheme,
    solver_alg=(;),
)
    # replace symbol with GradMode alg type
    alg_type = if alg isa Symbol
        gradmode_symbols[alg]
    else
        alg
    end

    # parse GradMode algorithm
    gradient_algorithm = if alg_type <: Union{GeomSum,ManualIter}
        alg_type(; tol, maxiter, verbosity, iterscheme)
    elseif alg_type <: Union{<:LinSolver,<:EigSolver}
        solver = if solver_alg isa NamedTuple # determine linear/eigen solver algorithm
            solver_kwargs = (;
                alg=Defaults.gradient_solver, tol, maxiter, verbosity, solver_alg...
            )

            solver_type = if alg <: LinSolver # replace symbol with solver alg type
                if solver_kwargs.alg isa Symbol
                    linsolver_solver_symbols[solver_kwargs.alg]
                else
                    solver_kwargs.alg
                end
            elseif alg <: EigSolver
                if solver_kwargs.alg isa Symbol
                    eigsolver_solver_symbols[solver_kwargs.alg]
                else
                    solver_kwargs.alg
                end
                solver_kwargs = (; # use default eager for EigSolver
                    eager=Defaults.gradient_eigsolver_eager,
                    solver_kwargs...,
                )
            end

            solver_kwargs = Base.structdiff(solver_kwargs, (; alg)) # remove `alg` keyword argument
            solver_type(; solver_kwargs...)
        else
            solver_alg
        end

        alg_type(; solver, iterscheme)
    else
        throw(ArgumentError("unknown gradient algorithm: $alg"))
    end

    return gradient_algorithm
end

function select_algorithm(
    ::Type{TensorKit.TruncationScheme}; alg=Defaults.trscheme, kwargs...
)
    alg_type = alg isa Symbol ? truncation_scheme_symbols[alg] : alg # replace Symbol with TruncationScheme type
    args = map(k -> last(kwargs[k]), keys(kwargs)) # extract only values of supplied kwargs (empty Tuple, if kwargs is empty)
    return alg_type(args...)
end

function select_algorithm(
    ::Type{SVDAdjoint}; fwd_alg=(;), rrule_alg=(;), broadening=nothing
)
    # parse forward SVD algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (; alg=Defaults.svd_fwd_alg, fwd_alg...) # overwrite with specified kwargs
        fwd_type = if fwd_kwargs.alg isa Symbol # replace symbol with alg type
            svd_fwd_symbols[fwd_kwargs.alg]
        else
            fwd_kwargs.alg
        end
        fwd_type(fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule SVD algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg=Defaults.svd_rrule_alg,
            verbosity=Defaults.svd_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs
        rrule_type = if rrule_kwargs.alg isa Symbol # replace symbol with alg type
            svd_rrule_symbols[rrule_kwargs.alg]
        else
            rrule_kwargs.alg
        end
        rrule_type(rrule_kwargs...)
    else
        rrule_alg
    end

    return SVDAdjoint(fwd_algorithm, rrule_algorithm, broadening)
end

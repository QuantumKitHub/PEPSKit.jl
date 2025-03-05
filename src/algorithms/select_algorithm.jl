"""
    select_algorithm(func_or_alg, args...; kwargs...) -> Algorithm

Parse arguments and keyword arguments to the algorithm struct corresponding to
`func_or_alg` and return an algorithm instance. To that end, we use a general interface
where all keyword arguments that can be algorithm themselves can be specified using

* `alg::Algorithm`: an instance of the algorithm struct or
* `(; alg::Union{Symbol,AlgorithmType}, alg_kwargs...)`: a `NamedTuple` where the algorithm is specified by a symbol or the type of the algorithm struct, and the algorithm keyword arguments 

A full description of the keyword argument can be found in the respective function or
algorithm struct docstrings.
"""
function select_algorithm end

function select_algorithm(
    ::typeof(fixedpoint),
    env₀::CTMRGEnv;
    tol=Defaults.optimizer_tol, # top-level tolerance
    verbosity=2, # top-level verbosity
    boundary_alg=(;),
    kwargs...,
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

    # adjust CTMRG tols and verbosity
    boundary_algorithm = if boundary_alg isa CTMRGAlgorithm
        boundary_alg
    elseif boundary_alg isa NamedTuple
        svd_alg = if haskey(boundary_alg, :svd_alg)
            if boundary_alg.svd_alg isa SVDAdjoint
                boundary_alg.svd_alg
            elseif boundary_alg.svd_alg isa NamedTuple
                select_algorithm(SVDAdjoint; rrule_alg=(; tol=1e-3tol), svd_alg...)
            else
                throw(ArgumentError("unknown SVD algorithm: $(boundary_alg.svd_alg)"))
            end
        else
            (; rrule_alg(; tol=1e-3tol))
        end

        select_algorithm(
            leading_boundary,
            env₀;
            tol=1e-4tol,
            verbosity=boundary_verbosity,
            svd_alg,
            boundary_alg...,
        )
    else
        throw(ArgumentError("unknown boundary algorithm: $boundary_alg"))
    end

    # adjust gradient verbosity
    gradient_algorithm = if gradient_alg isa GradMode
        gradient_alg
    elseif gradient_alg isa NamedTuple
        select_algorithm(
            GradMode; tol=1e-2tol, verbosity=gradient_verbosity, gradient_kwargs...
        )
    else
        throw(ArgumentError("unknown gradient algorithm: $gradient_alg"))
    end

    # adjust optimizer tol and verbosity
    optimizer_algorithm = if optimizer_alg isa OptimKit.OptimizationAlgorithm
        optimization_alg
    elseif optimizer_alg isa NamedTuple
        select_algorithm(
            OptimKit.OptimizationAlgorithm;
            tol,
            verbosity=optimizer_verbosity,
            optimizer_alg...,
        )
    else
        throw(ArgumentError("unknown optimization algorithm: $optimizer_alg"))
    end

    return select_algorithm(
        PEPSOptimize;
        boundary_alg=boundary_algorithm,
        gradient_alg=gradient_algorithm,
        optimizer_alg=optimizer_algorithm,
        kwargs...,
    )
end

function select_algorithm(
    ::Type{PEPSOptimize},
    env₀::CTMRGEnv;
    boundary_alg=(;),
    gradient_alg=(;),
    optimizer_alg=(;),
    reuse_env=Defaults.reuse_env,
    symmetrization=nothing,
)
    # parse boundary algorithm
    boundary_algorithm = if boundary_alg isa CTMRGAlgorithm
        boundary_alg
    elseif boundary_alg isa NamedTuple
        select_algorithm(leading_boundary, env₀; boundary_alg...)
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

    # parse optimizer algorithm
    optimizer_algorithm = if optimizer_alg isa OptimKit.OptimizationAlgorithm
        optimization_alg
    elseif optimizer_alg isa NamedTuple
        select_algorithm(OptimKit.OptimizationAlgorithm; optimizer_alg...)
    else
        throw(ArgumentError("unknown optimization algorithm: $optimizer_alg"))
    end

    return PEPSOptimize(
        boundary_algorithm,
        gradient_algorithm,
        optimizer_algorithm,
        reuse_env,
        symmetrization,
    )
end

function select_algorithm(
    ::Type{OptimKit.OptimizationAlgorithm};
    alg=Defaults.optimizer_alg,
    tol=Defaults.optimizer_tol,
    maxiter=Defaults.optimizer_maxiter,
    verbosity=Defaults.optimizer_verbosity,
    lbfgs_memory=Defaults.lbfgs_memory,
    # TODO: add linesearch, ... to kwargs and defaults?
)
    # replace symbol with projector alg type
    alg_type = if alg isa Symbol
        if alg == :gradientdescent
            GradientDescent
        elseif alg == :conjugategradient
            ConjugateGradient
        elseif alg == :lbfgs
            (; kwargs...) -> LBFGS(lbfgs_memory; kwargs...)
        else
            throw(ArgumentError("unknown optimizer algorithm: $alg"))
        end
    else
        alg
    end

    optimizer = alg_type(; gradtol=tol, maxiter, verbosity)
    return PEPSOptimize(
        boundary_algorithm,
        gradient_algorithm,
        optimizer,
        optimization_kwargs.reuse_env,
        optimization_kwargs.symmetrization,
    )
end

function select_algorithm(
    ::typeof(leading_boundary),
    env₀::CTMRGEnv;
    alg=Defaults.ctmrg_alg,
    tol=Defaults.ctmrg_tol,
    verbosity=Defaults.ctmrg_verbosity,
    svd_alg=(;),
    kwargs...,
)
    # adjust SVD rrule settings to CTMRG tolerance, verbosity and environment dimension
    svd_algorithm = if svd_alg isa SVDAdjoint
        svd_alg
    elseif svd_alg isa NamedTuple
        alg′ = select_algorithm(
            SVDAdjoint; rrule_alg=(; tol=1e1tol, verbosity=verbosity - 2), svd_alg...
        )
        if typeof(alg′.rrule_alg) <: Union{<:GMRES,<:Arnoldi}
            # extract maximal environment dimensions
            χenv = maximum(env₀.corners) do corner
                return dim(space(corner, 1))
            end
            krylovdim = round(Int, Defaults.krylovdim_factor * χenv)
            @reset alg′.rrule_alg.krylovdim = krylovdim
        end
    else
        throw(ArgumentError("unknown SVD algorithm: $svd_alg"))
    end

    return select_algorithm(
        CTMRGAlgorithm; alg, tol, verbosity, svd_alg=svd_algorithm, kwargs...
    )
end

function select_algorithm(
    ::Type{CTMRGAlgorithm};
    alg=Defaults.ctmrg_alg,
    tol=Defaults.ctmrg_tol,
    maxiter=Defaults.ctmrg_maxiter,
    miniter=Defaults.ctmrg_miniter,
    verbosity=Defaults.ctmrg_verbosity,
    trscheme=(; alg=Defaults.trscheme),
    svd_alg=(;),
    projector_alg=Defaults.projector_alg, # only allows for Symbol/Type{ProjectorAlgorithm} to expose projector kwargs
)
    # replace symbol with projector alg type
    alg_type = if alg isa Symbol
        if alg == :simultaneous
            SimultaneousCTMRG
        elseif alg == :sequential
            SequentialCTMRG
        else
            throw(ArgumentError("unknown CTMRG algorithm: $alg"))
        end
    else
        alg
    end

    # parse CTMRG projector algorithm
    projector_algorithm = select_algorithm(
        ProjectorAlgorithm; alg=projector_alg, svd_alg, trscheme, verbosity
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
        if alg == :halfinfinite
            HalfInfiniteProjector
        elseif alg == :fullinfinite
            FullInfiniteProjector
        else
            throw(ArgumentError("unknown projector algorithm: $alg"))
        end
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
        if alg == :geomsum
            GeomSum
        elseif alg == :manualiter
            ManualIter
        elseif alg == :linsolver
            LinSolver
        elseif alg == :eigsolver
            EigSolver
        else
            throw(ArgumentError("unknown GradMode algorithm: $alg"))
        end
    else
        alg
    end

    # parse GradMode algorithm
    gradient_algorithm = if alg_type <: Union{GeomSum,ManualIter}
        alg_type{iterscheme}(tol, maxiter, verbosity)
    elseif alg_type <: Union{<:LinSolver,<:EigSolver}
        solver = if solver_alg isa NamedTuple # determine linear/eigen solver algorithm
            solver_kwargs = (;
                alg=Defaults.gradient_solver, tol, maxiter, verbosity, solver_alg...
            )

            solver_type = if alg <: LinSolver # replace symbol with solver alg type
                if solver_kwargs.alg isa Symbol
                    if solver_kwargs.alg == :gmres
                        GMRES
                    elseif solver_kwargs.alg == :bicgstab
                        BiCGStab
                    else
                        throw(ArgumentError("unknown LinSolver solver: $(solver_kwargs.alg)"))
                    end
                else
                    solver_kwargs.alg
                end
            elseif alg <: EigSolver
                if solver_kwargs.alg isa Symbol
                    if solver_kwargs.alg == :arnoldi
                        Arnoldi
                    else
                        throw(ArgumentError("unknown EigSolver solver: $(solver_kwargs.alg)"))
                    end
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

        alg_type{iterscheme}(solver)
    else
        throw(ArgumentError("unknown gradient algorithm: $alg"))
    end

    return gradient_algorithm
end

function select_algorithm(
    ::Type{TensorKit.TruncationScheme}; alg=Defaults.trscheme, kwargs...
)
    alg_type = if alg isa Symbol # replace Symbol with TruncationScheme type
        if alg == :fixedspace
            FixedSpaceTruncation
        elseif alg == :notrunc
            TensorKit.NoTruncation
        elseif alg == :truncerr
            TensorKit.TruncationError
        elseif alg == :truncspace
            TensorKit.TruncationSpace
        elseif alg == :truncbelow
            TensorKit.TruncationCutoff
        else
            throw(ArgumentError("unknown truncation scheme: $alg"))
        end
    else
        alg
    end

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
            if fwd_kwargs.alg == :sdd
                TensorKit.SDD
            elseif fwd_kwargs.alg == :svd
                TensorKit.SVD
            elseif fwd_kwargs.alg == :iterative
                IterSVD
            else
                throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
            end
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
            if rrule_kwargs.alg == :gmres
                GMRES
            elseif rrule_kwargs.alg == :bicgstab
                BiCGStab
            elseif rrule_kwargs.alg == :arnoldi
                Arnoldi
            else
                throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
            end
        else
            rrule_kwargs.alg
        end
        rrule_type(rrule_kwargs...)
    else
        rrule_alg
    end

    return SVDAdjoint(fwd_algorithm, rrule_algorithm, broadening)
end

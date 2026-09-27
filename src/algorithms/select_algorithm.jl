_alg_or_nt(::Type{T}, alg::NamedTuple) where {T} = T(; alg...)
_alg_or_nt(::Type{T}, alg::A) where {T, A <: T} = alg
_alg_or_nt(::Type{T}, alg::DynamicTol{<:T}) where {T} = alg
_alg_or_nt(::Type, ::Nothing) = nothing
_alg_or_nt(T, alg) = throw(ArgumentError("unkown $T: $alg"))

"""
    parent_alg(alg)

Unwrap an algorithm from a `MPSKit.DynamicTols.DynamicTol` wrapper, if
present, returning the algorithm it wraps. Falls back to returning `alg` unchanged for any
other input, including `nothing`.
"""
parent_alg(alg) = alg
parent_alg(alg::DynamicTol) = parent_alg(alg.alg)

"""
    _dynamic_tol_or_alg(alg; dynamic_tols::Bool, tol_min::Real, tol_max::Real, tol_factor::Real)

Wrap `alg` in a `MPSKit.DynamicTols.DynamicTol` with the given
tolerance-scaling settings if `dynamic_tols` is `true`, otherwise return `alg` unchanged.
"""
function _dynamic_tol_or_alg(alg; dynamic_tols::Bool, tol_min::Real, tol_max::Real, tol_factor::Real)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

const DYNAMIC_TOL_KWARGS = (; dynamic_tols = nothing, tol_min = nothing, tol_max = nothing, tol_factor = nothing)

"""
    _pop_dynamic_tol_kwargs(kwargs::NamedTuple) -> dynamic_tol_kwargs, remaining_kwargs

Split off the `dynamic_tols`/`tol_min`/`tol_max`/`tol_factor` entries from `kwargs`,
returning them separately (to be passed on to [`_dynamic_tol_or_alg`](@ref)) from the
remaining keyword arguments (to be passed on to the algorithm constructor).
"""
function _pop_dynamic_tol_kwargs(kwargs::NamedTuple)
    dynamic_tol_kwargs = (;
        dynamic_tols = kwargs.dynamic_tols,
        tol_min = kwargs.tol_min,
        tol_max = kwargs.tol_max,
        tol_factor = kwargs.tol_factor,
    )
    return dynamic_tol_kwargs, Base.structdiff(kwargs, DYNAMIC_TOL_KWARGS)
end

"""
    select_algorithm(func_or_alg, args...; kwargs...) -> Algorithm

Parse arguments and keyword arguments to the algorithm struct corresponding to
`func_or_alg` and return an algorithm instance. To that end, we use a general interface
where all keyword arguments that can be algorithm themselves can be specified using

* `alg::Algorithm` : an instance of the algorithm struct or
* `(; alg::Symbol, alg_kwargs...)` : a `NamedTuple` where the algorithm is specified by a `Symbol` and the algorithm keyword arguments 

A full description of the keyword argument can be found in the respective function or
algorithm struct docstrings.
"""
function select_algorithm end

function select_algorithm(
        ::typeof(fixedpoint),
        env₀;
        tol = Defaults.optimizer_tol, # top-level tolerance
        verbosity = 3, # top-level verbosity
        boundary_alg = (;), gradient_alg = (;), optimizer_alg = (;), precondition_alg = (;),
        symmetrization = nothing, kwargs...,
    )
    # adjust CTMRG tols and verbosity
    if boundary_alg isa NamedTuple
        defaults = (;
            verbosity = verbosity ≤ 3 ? -1 : 3, tol = 1.0e-4tol,
            dynamic_tols = Defaults.ctmrg_dynamic_tols,
            tol_min = Defaults.ctmrg_tol_min,
            tol_max = Defaults.ctmrg_tol_max,
            tol_factor = Defaults.ctmrg_tol_factor,
        )
        boundary_kwargs = merge(defaults, boundary_alg)
        dynamic_tol_kwargs, boundary_kwargs = _pop_dynamic_tol_kwargs(boundary_kwargs)
        boundary_alg = select_algorithm(leading_boundary, env₀; boundary_kwargs...)
        boundary_alg = _dynamic_tol_or_alg(boundary_alg; dynamic_tol_kwargs...)
    end

    # defaults specific to fully symmetric contraction algorithms
    if parent_alg(boundary_alg) isa Union{C4vCTMRG, SymmetricBoundaryMPS}
        # symmetrize state and gradient
        if isnothing(symmetrization)
            symmetrization = RotateReflect()
        end
    end

    # adjust gradient verbosity and construct the gradient algorithm
    if gradient_alg isa NamedTuple
        # TODO: check this:
        defaults = (;
            verbosity = verbosity ≤ 3 ? -1 : 3, tol = 1.0e-2tol,
            dynamic_tols = Defaults.gradient_dynamic_tols,
            tol_min = Defaults.gradient_tol_min,
            tol_max = Defaults.gradient_tol_max,
            tol_factor = Defaults.gradient_tol_factor,
        )
        gradient_kwargs = merge(defaults, gradient_alg)
        dynamic_tol_kwargs, gradient_kwargs = _pop_dynamic_tol_kwargs(gradient_kwargs)
        gradient_alg = GradientAlgorithm(; gradient_kwargs...)
        gradient_alg = _dynamic_tol_or_alg(gradient_alg; dynamic_tol_kwargs...)
    end

    # adjust preconditioner verbosity and construct the preconditioner algorithm
    if precondition_alg isa NamedTuple
        defaults = (;
            verbosity = verbosity ≤ 3 ? -1 : 3,
            dynamic_tols = Defaults.precondition_dynamic_tols,
            tol_min = Defaults.precondition_tol_min,
            tol_max = Defaults.precondition_tol_max,
            tol_factor = Defaults.precondition_tol_factor,
        )
        precondition_kwargs = merge(defaults, precondition_alg)
        dynamic_tol_kwargs, precondition_kwargs = _pop_dynamic_tol_kwargs(precondition_kwargs)
        precondition_alg = PreconditionAlgorithm(; precondition_kwargs...)
        precondition_alg = _dynamic_tol_or_alg(precondition_alg; dynamic_tol_kwargs...)
    end

    # adjust optimizer tol and verbosity
    if optimizer_alg isa NamedTuple
        defaults = (; tol, verbosity)
        optimizer_alg = merge(defaults, optimizer_alg)
    end

    return PEPSOptimize(;
        boundary_alg, gradient_alg, optimizer_alg, precondition_alg,
        symmetrization, kwargs...,
    )
end

function select_algorithm(
        ::typeof(leading_boundary), ::SymmetricBoundaryMPSEnv;
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        verbosity = Defaults.ctmrg_verbosity,
        mps_alg = (;),
    )
    return SymmetricBoundaryMPS(; tol, maxiter, verbosity, mps_alg)
end

function select_algorithm(
        ::typeof(leading_boundary),
        env₀::CTMRGEnv;
        alg = Defaults.ctmrg_alg,
        tol = Defaults.ctmrg_tol,
        verbosity = Defaults.ctmrg_verbosity,
        decomposition_alg = (;),
        kwargs...,
    )
    # adjust SVD rrule settings to CTMRG tolerance, verbosity and environment dimension
    if decomposition_alg isa NamedTuple &&
            haskey(decomposition_alg, :rrule_alg) &&
            decomposition_alg.rrule_alg isa NamedTuple
        χenv = maximum(env₀.corners) do corner
            return dim(space(corner, 1))
        end
        # TODO: this should be scaled for each sector separately I think
        krylovdim = max(
            Defaults.svd_rrule_min_krylovdim, round(Int, Defaults.krylovdim_factor * χenv)
        )
        rrule_alg = (; tol = 1.0e1tol, verbosity = verbosity - 2, krylovdim, decomposition_alg.rrule_alg...)
        decomposition_alg = (; rrule_alg, decomposition_alg...)
    end

    return CTMRGAlgorithm(; alg, tol, verbosity, decomposition_alg, kwargs...)
end

_alg_or_nt(::Type{T}, alg::NamedTuple) where {T} = T(; alg...)
_alg_or_nt(::Type{T}, alg::A) where {T,A<:T} = alg
_alg_or_nt(T, alg) = throw(ArgumentError("unkown $T: $alg"))

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
    env₀::CTMRGEnv;
    tol=Defaults.optimizer_tol, # top-level tolerance
    verbosity=2, # top-level verbosity
    boundary_alg=(;),
    gradient_alg=(;),
    optimizer_alg=(;),
    kwargs...,
)
    # adjust CTMRG tols and verbosity
    if boundary_alg isa NamedTuple
        defaults = (; verbosity=verbosity ≤ 1 ? -1 : verbosity, tol=1e-4tol)
        boundary_kwargs = merge(defaults, boundary_alg)
        boundary_alg = select_algorithm(leading_boundary, env₀; boundary_kwargs...)
    end

    # adjust gradient verbosity
    if gradient_alg isa NamedTuple
        # TODO: check this:
        defaults = (; verbosity=verbosity ≤ 2 ? -1 : 3, tol=1e-2tol)
        gradient_alg = merge(defaults, gradient_alg)
    end

    # adjust optimizer tol and verbosity
    if optimizer_alg isa NamedTuple
        defaults = (; tol, verbosity=verbosity < 1 ? -1 : 3)
        optimizer_alg = merge(defaults, optimizer_alg)
    end

    return PEPSOptimize(; boundary_alg, gradient_alg, optimizer_alg, kwargs...)
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
    if svd_alg isa NamedTuple &&
        haskey(svd_alg, :rrule_alg) &&
        svd_alg.rrule_alg isa NamedTuple
        χenv = maximum(env₀.corners) do corner
            return dim(space(corner, 1))
        end
        # TODO: this should be scaled for each sector separately I think
        krylovdim = max(
            Defaults.svd_rrule_min_krylovdim, round(Int, Defaults.krylovdim_factor * χenv)
        )
        rrule_alg = (; tol=1e1tol, verbosity=verbosity - 2, krylovdim, svd_alg.rrule_alg...)
        svd_alg = (; rrule_alg, svd_alg...)
    end
    svd_algorithm = SVDAdjoint(; svd_alg...)

    return CTMRGAlgorithm(; alg, tol, verbosity, svd_alg=svd_algorithm, kwargs...)
end

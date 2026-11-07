"""
$(TYPEDEF)

Abstract super type for all algorithms that contract an infinite square network by computing
a boundary fixed point, such as CTMRG and boundary MPS algorithms.

This is the type of the `boundary_alg` field of [`PEPSOptimize`](@ref), and hence the
supertype every algorithm which can be used to contract a network during a variational
optimization must belong to.
"""
abstract type BoundaryAlgorithm end

"""
    _tol(alg::BoundaryAlgorithm)

Effective convergence tolerance of a boundary contraction algorithm. Defaults to the `tol`
field; algorithms which keep their tolerance elsewhere (such as boundary MPS algorithms,
where it lives on the wrapped MPS optimization algorithm) must overload this.
"""
_tol(alg::BoundaryAlgorithm) = alg.tol

"""
    BoundaryAlgorithm(; alg=:$(Defaults.boundary_alg), kwargs...)

Keyword argument parser returning the appropriate [`BoundaryAlgorithm`](@ref) struct, where
`alg` can be one of the following:

* `:SimultaneousCTMRG`, `:SequentialCTMRG`, `:C4vCTMRG` : see [`CTMRGAlgorithm`](@ref)
* `:BoundaryMPS`, `:C4vBoundaryMPS` : see [`BoundaryMPSAlgorithm`](@ref)

All remaining keyword arguments are passed on to the corresponding parser. In
[`leading_boundary`](@ref) and [`fixedpoint`](@ref), the type of the initial environment
selects between the two families, and `alg` selects the algorithm within that family.
"""
function BoundaryAlgorithm(; alg = Defaults.boundary_alg, kwargs...)
    # CTMRG variants go through their own parser, which handles the projector and
    # decomposition keyword arguments
    haskey(CTMRG_SYMBOLS, alg) && return CTMRGAlgorithm(; alg, kwargs...)
    haskey(BOUNDARY_MPS_SYMBOLS, alg) && return BoundaryMPSAlgorithm(; alg, kwargs...)
    throw(ArgumentError("unknown boundary algorithm: $alg"))
end

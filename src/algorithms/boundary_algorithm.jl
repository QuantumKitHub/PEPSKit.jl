"""
$(TYPEDEF)

Abstract super type for all algorithms that contract an infinite square network by computing
a boundary fixed point, such as CTMRG and boundary MPS algorithms.

This is the type of the `boundary_alg` field of [`PEPSOptimize`](@ref), and hence the
supertype every algorithm which can be used to contract a network during a variational
optimization must belong to.
"""
abstract type BoundaryAlgorithm end

const BOUNDARY_ALGORITHM_SYMBOLS = IdDict{Symbol, Type{<:BoundaryAlgorithm}}()

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
`alg` selects the boundary contraction *family*:

* `:SimultaneousCTMRG`, `:SequentialCTMRG`, `:C4vCTMRG` : dispatch to [`CTMRGAlgorithm`](@ref)
* `:SymmetricBoundaryMPS` : dispatch to [`SymmetricBoundaryMPS`](@ref)

All remaining keyword arguments are forwarded to the corresponding parser. Note that the
underlying MPS optimization algorithm of a boundary MPS contraction is *not* selected here,
but through that parser's own `mps_alg` keyword, since several boundary MPS families can be
driven by the same MPS algorithm.
"""
function BoundaryAlgorithm(; alg = Defaults.boundary_alg, kwargs...)
    # CTMRG variants go through their own parser, which handles the projector and
    # decomposition keyword arguments
    haskey(CTMRG_SYMBOLS, alg) && return CTMRGAlgorithm(; alg, kwargs...)
    haskey(BOUNDARY_ALGORITHM_SYMBOLS, alg) ||
        throw(ArgumentError("unknown boundary algorithm: $alg"))
    return BOUNDARY_ALGORITHM_SYMBOLS[alg](; kwargs...)
end

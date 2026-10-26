"""
Select the contraction method used to evaluate local expectation values.
"""
abstract type ContractionAlgorithm end

"""
    ExactContract()

Contract each local expectation-value patch exactly, without boundary-MPS truncation.
This is the default algorithm for [`expectation_value`](@ref).
"""
struct ExactContract <: ContractionAlgorithm end

"""
    BoundaryMPSContract(; trunc = nothing, maxiter = 1, direction = :auto)
    BoundaryMPSContract(zipup, dmrg; direction = :auto)

Approximately contract single-layer PEPO expectation values using boundary-MPS zipup sweeps.
Each term is normalized in its own enclosing patch.
Dense terms are reordered into a column-wise snake and decomposed into MPOs without truncation.
Explicit [`MPOTerm`](@ref) factors retain their order and are connected by non-self-intersecting nearest-neighbor paths.
Tensor-product terms are routed with trivial MPO bonds.
Routing tries horizontal-first and then vertical-first shortest connections; more complicated explicit MPO routes raise a not-implemented error.
Single-site terms are evaluated exactly.
The operator decomposition is exact; `trunc` controls only boundary-MPS truncation.
By default (`trunc = nothing`), the rank is limited to the largest CTMRG boundary dimension.
After each zipup step, `maxiter` single-site DMRG sweeps refine the result; use zero to disable refinement.
Alternatively, supply the zipup and DMRG algorithms directly, using `dmrg = nothing` to disable refinement.
The fields `zipup` and `dmrg` store these algorithms; `zipup = nothing` selects zipup truncation from the environment at evaluation time.

With `direction = :auto`, wide and square patches are contracted north to south, and tall patches east to west.
Use `direction = :rows` or `:columns` to select an orientation explicitly.

Pass this algorithm as `expectation_value(ρ, O, env, BoundaryMPSContract(; kwargs...))`.
"""
struct BoundaryMPSContract{Z, D} <: ContractionAlgorithm
    zipup::Z
    dmrg::D
    direction::Symbol

    function BoundaryMPSContract(zipup, dmrg; direction::Symbol = :auto)
        direction in (:auto, :rows, :columns) ||
            throw(ArgumentError("invalid sweep direction: $direction"))
        return new{typeof(zipup), typeof(dmrg)}(zipup, dmrg, direction)
    end
end

function BoundaryMPSContract(; trunc = nothing, maxiter::Int = 1, direction::Symbol = :auto)
    maxiter >= 0 || throw(ArgumentError("maxiter should be nonnegative"))
    zipup = isnothing(trunc) ? nothing : Zipup(; trunc)
    dmrg = iszero(maxiter) ? nothing : DMRG(; maxiter, verbosity = 0)
    return BoundaryMPSContract(zipup, dmrg; direction)
end

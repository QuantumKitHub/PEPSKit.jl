"""
$(TYPEDEF)

Projector algorithm implementing the `eigh` decomposition of a Hermitian enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vEighProjector(; kwargs...)

Construct the C₄ᵥ `eigh`-based projector algorithm based on the following keyword arguments:

* `decomposition_alg::Union{<:EighAdjoint,NamedTuple}=EighAdjoint()` : `eigh` algorithm including the reverse rule. See [`EighAdjoint`](@ref).
* `trunc::Union{TruncationStrategy,NamedTuple}=(; alg::Symbol=:$(Defaults.trunc))` : Truncation strategy for the projector computation, which controls the resulting virtual spaces. Here, `alg` can be one of the following:
    - `:FixedSpaceTruncation` : Keep virtual spaces fixed during projection
    - `:notrunc` : No singular values are truncated and the performed SVDs are exact
    - `:truncerror` : Additionally supply error threshold `η`; truncate to the maximal virtual dimension of `η`
    - `:truncrank` : Additionally supply truncation dimension `η`; truncate such that the 2-norm of the truncated values is smaller than `η`
    - `:truncspace` : Additionally supply truncation space `η`; truncate according to the supplied vector space
    - `:trunctol` : Additionally supply singular value cutoff `η`; truncate such that every retained singular value is larger than `η`
* `verbosity::Int=$(Defaults.projector_verbosity)` : Projector output verbosity which can be:
    0. Suppress output information
    1. Print singular value degeneracy warnings
"""
struct C4vEighProjector{S <: EighAdjoint, T} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    verbosity::Int
end
function C4vEighProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :C4vEighProjector, kwargs...)
end
PROJECTOR_SYMBOLS[:C4vEighProjector] = C4vEighProjector

"""
Compute the normalized and Hermitian-symmetrized C₄ᵥ enlarged corner.
```
    C-←-E-←-
    |   |   
    E---A---
    |   |
```
"""
function c4v_enlarge(network, env, ::C4vEighProjector)
    enlarged_corner = TensorMap(EnlargedCorner(network, env, (NORTHWEST, 1, 1)))
    enlarged_corner = project_hermitian(enlarged_corner)
    return enlarged_corner / norm(enlarged_corner)
end

"""
Compute the C₄ᵥ projector from `eigh` decomposing the Hermitian `enlarged_corner`.
Return the projector and decomposition diagnostics used to renormalize the corner.
"""
function c4v_projector!(enlarged_corner, alg::C4vEighProjector)
    alg = _set_decomposition_truncation(alg, truncation_strategy(alg, enlarged_corner))
    eigh_alg = decomposition_algorithm(alg)

    D, V, truncation_error = eigh_trunc!(enlarged_corner, eigh_alg)

    # Check for degenerate eigenvalues
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(D)
            vals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(D))
            @warn("degenerate eigenvalues detected: ", vals)
        end
    end

    return V, (; contraction_metrics = (; truncation_error), D, V)
end

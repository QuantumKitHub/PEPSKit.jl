"""
$(TYPEDEF)

Projector algorithm implementing the `qr` decomposition of a column-enlarged corner.

## Fields

$(TYPEDFIELDS)

## Constructors

    C4vQRProjector(; kwargs...)

Construct the C₄ᵥ `qr`-based projector algorithm based on the following keyword arguments:

* `decomposition_alg=QRAdjoint()` : `left_orth` algorithm including the reverse rule. See [`QRAdjoint`](@ref).
"""
struct C4vQRProjector{S} <: ProjectorAlgorithm
    # TODO: support all `left_orth` algorithms
    decomposition_alg::S
end
function C4vQRProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :C4vQRProjector, kwargs...)
end
PROJECTOR_SYMBOLS[:C4vQRProjector] = C4vQRProjector

decomposition_algorithm(alg::C4vQRProjector) = alg.decomposition_alg

# no truncation
_set_truncation(alg::C4vQRProjector, ::TruncationStrategy) = alg
_set_decomposition_truncation(alg::C4vQRProjector, ::TruncationStrategy) = alg

"""
Compute the column-enlarged northwest corner for C₄ᵥ QR-CTMRG.
```
    C-←-E-←-
    ↓   |   
```
"""
function c4v_enlarge(network, env, ::C4vQRProjector)
    return TensorMap(ColumnEnlargedCorner(env, (NORTHWEST, 1, 1)))
end

"""
Compute the C₄ᵥ projector by decomposing the column-enlarged corner with `left_orth`.
```
                   R--←--
                   ↓
    C-←-E-←-  =  [~Q~]
    ↓   |        ↓   |
```
"""
function c4v_projector!(enlarged_corner, alg::C4vQRProjector)
    Q, R = left_orth!(enlarged_corner, decomposition_algorithm(alg))
    # TODO: what's a meaningful way to compute a truncation error/condition number in this scheme?
    return Q, (; contraction_metrics = (;), Q, R)
end

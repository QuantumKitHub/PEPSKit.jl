module PEPSKitGPUArraysExt

using GPUArrays
using PEPSKit
using LinearAlgebra
using TensorKit
using TensorKit: MatrixAlgebraKit as MAK
import PEPSKit: _column_vectors, _materialize_block, _indexable_diagview

PEPSKit._materialize_block(m::Union{Adjoint{T, A}, Transpose{T, A}}) where {T, A <: AnyGPUArray} = copyto!(similar(m, size(m)), m)
function PEPSKit._column_vectors(::Type{A}, m::AbstractMatrix) where {A <: AnyGPUArray}
    m̃ = PEPSKit._materialize_block(m)
    return A[m̃[:, j] for j in axes(m̃, 2)]
end
PEPSKit._indexable_diagview(m::AnyGPUArray) = collect(MAK.diagview(m))

# Hook into the collection-level convergence API. Deliberately restricted to the generic
# CTMRG algorithms: `C4vCTMRG` overrides `corner_spectrum` to `eigh_vals` (its corners are
# diagonal), so a blanket override here would silently switch it back to `svd_vals`.
# Restricted to `BlasFloat` since that is what the batched GPU SVD drivers support.
const GPUTensorMap = TensorMap{<:LinearAlgebra.BlasFloat, <:Any, <:Any, <:Any, <:AbstractGPUArray}

function PEPSKit.corner_spectra(
        Cs::AbstractArray{<:GPUTensorMap},
        ::Union{PEPSKit.SequentialCTMRG, PEPSKit.SimultaneousCTMRG},
    )
    return _batched_spectra(Cs)
end
function PEPSKit.edge_spectra(
        Ts::AbstractArray{<:GPUTensorMap},
        ::Union{PEPSKit.SequentialCTMRG, PEPSKit.SimultaneousCTMRG},
    )
    return _batched_spectra(Ts)
end

# `calc_convergence` decomposes every corner and every edge of the environment
# for regular CTMRG, which is expensive, at *least* 8 separate `svd_vals` calls.
# Across *multiple tensors* the situation is much better than within *one*,
# because the corners have to all share a space,
# so for a given sector their blocks have identical sizes and batch with no padding at all.
function _batched_spectra(ts::AbstractArray{<:AbstractTensorMap})
    items = [(i, c) for i in eachindex(ts) for c in blocksectors(ts[i])]
    isempty(items) && return map(svd_vals, ts)
    # `batched_svd_vals!` destroys its inputs, but `ts` is the live environment, and computing
    # the convergence spectra must not damage the environment it is measuring.
    # Copying whole tensors costs one copy per tensor instead of one per block.
    ts′ = map(copy, ts)
    As = [block(ts′[i], c) for (i, c) in items]
    alg = MAK.default_algorithm(MAK.batched_svd_vals!, typeof(As))
    Ss = map(t -> MAK.initialize_output(MAK.svd_vals!, t, alg), ts)
    MAK.batched_svd_vals!(As, [block(Ss[i], c) for (i, c) in items], alg)
    return Ss
end

end

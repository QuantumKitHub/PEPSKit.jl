module PEPSKitGPUArraysExt

using GPUArrays
using PEPSKit, MatrixAlgebraKit
using LinearAlgebra
import PEPSKit: _column_vectors, _materialize_block, _indexable_diagview

PEPSKit._materialize_block(m::Union{Adjoint{T, A}, Transpose{T, A}}) where {T, A <: AnyGPUArray} = copyto!(similar(m, size(m)), m)
function PEPSKit._column_vectors(::Type{A}, m::AbstractMatrix) where {A <: AnyGPUArray}
    m̃ = PEPSKit._materialize_block(m)
    return A[m̃[:, j] for j in axes(m̃, 2)]
end
PEPSKit._indexable_diagview(m::AnyGPUArray) = collect(MatrixAlgebraKit.diagview(m))

end

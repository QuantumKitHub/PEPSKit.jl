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

# Reduce into a 0-dimensional device array instead of returning a host scalar. `sdiag_pow` only
# feeds this into a broadcast, and a 0-dim array broadcasts as a scalar, so the value never has to
# come back to the CPU. Returning a scalar here would force a device sync on every call, and
# `sdiag_pow` runs once per bond per weight absorption in simple update.
function PEPSKit._maxabs(data::AnyGPUArray)
    T = real(eltype(data))
    acc = similar(data, T, ())
    fill!(acc, zero(T))
    Base.mapreducedim!(abs, max, acc, data)
    return acc
end

end

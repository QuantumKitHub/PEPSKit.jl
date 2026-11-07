module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays.KernelAbstractions: @kernel, @index, @Const, get_backend
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

const GPUTensorMap = TensorMap{<:LinearAlgebra.BlasFloat, <:Any, <:Any, <:Any, <:AbstractGPUArray}
function PEPSKit.corner_spectrum(C::GPUTensorMap, ::PEPSKit.C4vCTMRG)
    S = eigh_vals(C)
    x = parent(S)
    isempty(x) && return S
    m = reduce(_maxabs_signed, x; dims = 1, init = zero(eltype(x)))
    starts = sort!(Int32[first(r) for r in values(S.structure)])
    push!(starts, length(x) + 1)
    offsets = copyto!(similar(x, Int32, length(starts)), starts)
    _flip_spectrum_kernel!(get_backend(x))(x, offsets, m; ndrange = length(x))
    return S
end
_maxabs_signed(a, b) = abs(a) > abs(b) ? a : abs(b) > abs(a) ? b : max(a, b)

# if `m[1] < 0`, negate `x` and reverse it within each block; `offsets` are the sorted block start
# positions followed by `length(x) + 1`
@kernel function _flip_spectrum_kernel!(x, @Const(offsets), @Const(m))
    i = @index(Global, Linear)
    @inbounds if m[1] < zero(eltype(m))
        lo, hi = 1, length(offsets) - 1 # last block starting at or before `i`
        while lo < hi
            mid = (lo + hi + 1) >> 1
            offsets[mid] <= i ? (lo = mid) : (hi = mid - 1)
        end
        j = offsets[lo] + offsets[lo + 1] - 1 - i # mirror image of `i` within its block
        if i < j
            xi, xj = x[i], x[j]
            x[i] = -xj
            x[j] = -xi
        elseif i == j
            x[i] = -x[i]
        end
    end
end

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

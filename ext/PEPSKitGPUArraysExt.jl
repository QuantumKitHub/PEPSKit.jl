module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays: AnyGPUArray, AllocCache
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

# Each caller (such as `su_iter`) gets a pair of caches. This makes sense to do on a per-caller basis
# because what is being cached varies between algorithms.
# For each caller we can also store several caches. For CTMRG and SU,
# we currently store one but we retain the ability to store a "stack" if needed in the future.
# TODO could this be better handled by LRU.jl or MemoizationKit.jl?
const ALLOC_CACHES = Dict{Tuple{Symbol, Int}, Vector{AllocCache}}()
const ALLOC_CACHES_LOCK = ReentrantLock()

function _caches(site::Symbol, depth::Int)
    return Base.@lock ALLOC_CACHES_LOCK begin
        get!(() -> [AllocCache() for _ in 1:depth], ALLOC_CACHES, (site, depth))
    end
end

function PEPSKit._with_alloc_cache(f, ::Type{<:AnyGPUArray}, site::Symbol, iter::Int, depth::Int)
    cache = @inbounds _caches(site, depth)[mod1(iter + 1, depth)]
    return GPUArrays.@cached cache f()
end

PEPSKit._uncache(x, ::Type{<:AnyGPUArray}) = deepcopy(x)

function PEPSKit.free_alloc_caches!(::Type{<:AnyGPUArray}, caller::Symbol)
    Base.@lock ALLOC_CACHES_LOCK begin
        # collect first: freeing mutates ALLOC_CACHES
        stale = [key for key in keys(ALLOC_CACHES) if first(key) === caller]
        for key in stale
            for cache in ALLOC_CACHES[key]
                GPUArrays.unsafe_free!(cache)
            end
            delete!(ALLOC_CACHES, key)
        end
    end
    return nothing
end

function PEPSKit.free_alloc_caches!(::Type{<:AnyGPUArray})
    Base.@lock ALLOC_CACHES_LOCK begin
        for caches in values(ALLOC_CACHES), cache in caches
            GPUArrays.unsafe_free!(cache)
        end
        empty!(ALLOC_CACHES)
    end
    return nothing
end

end

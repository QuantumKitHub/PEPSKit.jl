module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays: AnyGPUArray, AllocCache
using PEPSKit

# Each caller (such as `su_iter`) gets a pair of caches. This makes sense to do on a per-caller basis
# because what is being cached varies between algorithms.
# For each caller we also store several caches, for SimultaneousCTMRG and SU,
# one for even iterations and one for odd, and for SequentialCTMRG, 5 (one "round" plus one extra)
# This has to be done because we can't reuse a cache from iteration `i`
# until iteration `i+n` is completely finished and its result handed off.
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

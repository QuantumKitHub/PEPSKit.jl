module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays: AnyGPUArray, AllocCache
using PEPSKit

# Each caller (such as `su_iter`) gets a pair of caches. This makes sense to do on a per-caller basis 
# because what is being cached varies between algorithms.
# For each caller we also store *two* caches, one for even iterations and one for odd.
# This has to be done because we can't reuse a cache from iteration `i`
# until iteration `i+1` is completely finished and its result handed off to iteration `i+2`. 
const ALLOC_CACHES = Dict{Symbol, NTuple{2, AllocCache}}()
const ALLOC_CACHES_LOCK = ReentrantLock()

function _caches(site::Symbol)
    return Base.@lock ALLOC_CACHES_LOCK begin
        get!(() -> (AllocCache(), AllocCache()), ALLOC_CACHES, site)
    end
end

function PEPSKit._with_alloc_cache(f, ::Type{<:AnyGPUArray}, site::Symbol, iter::Int)
    cache = @inbounds _caches(site)[mod1(iter + 1, 2)]
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

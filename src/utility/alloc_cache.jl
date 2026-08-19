"""
    with_alloc_cache(f, storage, caller::Symbol, iter::Int) -> f()

Run one iteration `f()` of an iterative algorithm using memory cache.

**This is a no-op unless `GPUArrays` is loaded _and_ `storage` is a GPU array.**

There seems to be no benefit to caching for CPU-side memory, but for the device,
using a warm memory pool avoids everything being blocked while the GPU device driver
allocates.

`caller` ids the calling algo (`:su`, `:ctmrg`) so that algorithms allocating different
buffer sizes avoid stepping on each others' caches. Not every algorithm needs an allocation cache,
only the ones with repeated iterations.

Belief propagation seems to not benefit from the caching as much, so it's currently unused there.

`iter` selects between two alternating caches per `caller`. A buffer allocated during iteration
`i` can't become reusable until iteration `i+2`, by which point the output of iteration `i` has
been used by iteration `i+1`. With only one cache, it would be possible to overwrite the state
at `i+1` while it's still being used.

Caching is skipped while Zygote is differentiating, because the reverse-mode tape holds references
to intermediates, and recycling those could silently corrupt gradients.
"""
function with_alloc_cache(f, storage::Type, caller::Symbol, iter::Int)
    Zygote.isderiving() && return f()
    return _with_alloc_cache(f, storage, caller, iter)
end
_with_alloc_cache(f, ::Type, ::Symbol, ::Int) = f()

"""
    free_alloc_caches!(storage)

Release memory held by the allocation caches for storage type `storage`. Does nothing 
unless the `PEPSKitGPUArraysExt` extension is loaded and `storage` is a GPU array type.
It's worth calling this when the bond dimension changes, because the cache keys depend on
buffer size.
"""
free_alloc_caches!(::Type) = nothing

"""
    with_alloc_cache(f, storage, caller::Symbol, iter::Int) -> f()

Run one iteration `f()` of an iterative algorithm using a memory cache.

**This is a no-op unless `GPUArrays` is loaded _and_ `storage` is a GPU array.**

There seems to be no benefit to caching for CPU-side memory, but for the device,
using a warm memory pool which recycles memory avoids everything being blocked
while the GPU device driver allocates.

`caller` ids the calling algo (`:su`, `:ctmrg`) so that algorithms allocating different
buffer sizes avoid stepping on each others' caches. Not every algorithm needs an allocation cache,
only the ones with repeated iterations.

Belief propagation seems to not benefit from the caching as much, so it's currently unused there.

`iter` selects between two alternating caches per `caller`. A buffer allocated during iteration
`i` can't become reusable until later iterations have completely used and discarded its output.
For `:SimultaneousCTMRG` and simple update, that occurs after 2 iterations, while for `:SequentialCTMRG`,
it occurs after 5. With too few caches, it would be possible to overwrite the state
at later iterations while it's still being used.

Caching is skipped while `Zygote.jl` is differentiating, because the reverse-mode tape holds references
to intermediates, and recycling those could silently corrupt gradients.
"""
function with_alloc_cache(f, storage::Type, caller::Symbol, iter::Int, depth::Int = 2)
    Zygote.isderiving() && return f()
    return _with_alloc_cache(f, storage, caller, iter, depth)
end

"""
    uncache(x, storage::Type) -> x

Copy `x` out of any allocation-cache region, so that it stays live and is not overwriten
once the enclosing [`with_alloc_cache`](@ref) block has exited. This is necessary to
ensure `leading_boundary` and other functions which call back into AD handle cached
memory correctly.

Buffers allocated inside a cache block are handed back to the pool when the block exits,
and the next cached call may hand them out again, which *silently overwrites* a result the
caller is still holding. Anything that escapes such a block must be copied out.

**This is a no-op unless `GPUArrays` is loaded _and_ `storage` is a GPU array**, and also
while Zygote is differentiating, since caching is skipped in both of those cases.
"""
function uncache(x, storage::Type)
    Zygote.isderiving() && return x
    return _uncache(x, storage)
end
_uncache(x, ::Type) = x

"""
    alloc_cache_depth(alg)

How many iterations a buffer must go unused before it may be recycled.
For SimultaneousCTMRG and SU, 2 is enough, but not necessarily for other algorithms.
"""
alloc_cache_depth(alg) = 2
_with_alloc_cache(f, ::Type, ::Symbol, ::Int, ::Int) = f()

"""
    free_alloc_caches!(storage)
    free_alloc_caches!(storage, caller::Symbol)

Release memory held by the allocation caches for storage type `storage`, either for every
`caller` (if none is provided) or only for the given one. Does nothing unless the
`PEPSKitGPUArraysExt` extension is loaded and `storage` is a GPU array type.
This should be called when the bond dimension changes, because the cache keys depend on
buffer size and the caches can't be reused when the bond dimension has changed.

!!! warning
    This releases the underlying device memory, so it is only safe to call when nothing
    still references a buffer that was allocated inside a [`with_alloc_cache`](@ref) block.
    Results that escape such a block must have been copied out with [`uncache`](@ref)
    first, otherwise freeing the cache leaves them undefined.
"""
free_alloc_caches!(::Type) = nothing
free_alloc_caches!(::Type, ::Symbol) = nothing

# last buffer-shape signature seen per caller, used to detect when cached buffer sizes
# have gone stale because a bond dimension changed
const ALLOC_CACHE_SIGNATURES = Dict{Tuple{Symbol, Symbol}, UInt}()
const ALLOC_CACHE_SIGNATURES_LOCK = ReentrantLock()

"""
    free_stale_alloc_caches!(storage, caller::Symbol, phase::Symbol, signature::UInt)

Release `caller`'s allocation caches when `signature` differs from the one seen at the same
`phase` of the previous call, and record `signature` as the current one for that phase.

The caches are keyed by buffer size, so that when the size changes the old, unusable caches
can be freed and new ones allocated, corresponding to the new size. This avoids the cache
size growing without bound.

`phase` distinguishes the points a caller checks from. For example, CTMRG grows the
corner and edge spaces as it converges, so its incoming and outgoing environments differ
whenever the truncation is *not* fixed-space. Recording both under one key makes the stored
signature alternate between them, so every check reports stale and the pool is freed on every
call. Comparing each phase only against itself keeps the pool across repeated calls while
still invalidating it when the spaces genuinely change.

Like [`free_alloc_caches!`](@ref) this is a no-op without a GPU storage type, and it's
skipped while `Zygote.jl` is differentiating, since caching is disabled there anyway.
"""
function free_stale_alloc_caches!(storage::Type, caller::Symbol, phase::Symbol, signature::UInt)
    Zygote.isderiving() && return nothing
    stale = Base.@lock ALLOC_CACHE_SIGNATURES_LOCK begin
        key = (caller, phase)
        previous = get(ALLOC_CACHE_SIGNATURES, key, nothing)
        ALLOC_CACHE_SIGNATURES[key] = signature
        !isnothing(previous) && previous != signature
    end
    stale && free_alloc_caches!(storage, caller)
    return nothing
end

"""
    alloc_cache_signature(tensors) -> UInt

Hash the spaces of `tensors`, for use as a [`free_stale_alloc_caches!`](@ref) signature.
Two states whose tensors live in the same spaces allocate the same buffer sizes.
"""
alloc_cache_signature(tensors) = hash(map(space, tensors))

module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays: AnyGPUArray, AllocCache
using PEPSKit
using TensorKit
using TensorKit: MatrixAlgebraKit as MAK

const BATCH_THRESHOLD = 4

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


const Factorizations = TensorKit.Factorizations

"""
    _batched_spectra_alg(proto) -> alg or nothing

Default batched SVD algorithm this backend offers, or `nothing` if it has none.
"""
function _batched_spectra_alg(proto)
    # TODO BAD MAKE THIS A MAK CALL
    alg = try
        MAK.default_svd_algorithm(typeof(similar(proto, 0, 0, 0)))
    catch
        return nothing
    end
    return alg isa Factorizations.BatchedSVDAlgorithm ? alg : nothing
end

# `calc_convergence` decomposes every corner and every edge of the environment
# for regular CTMRG, which is expensive, at *least* 8 separate `svd_vals` calls.
# Across *multiple tensors* the situation is much better than within *one*,
# because the corners have to all share a space,
# so for a given sector their blocks have identical sizes and can be batched with no
# padding at all.
function _batch_svd_vals!(ts, Ss, items, (m, n), pad::Bool, alg)
    b1 = block(ts[first(items)[1]], first(items)[2])
    A = similar(b1, m, n, length(items))
    pad && fill!(A, zero(eltype(A)))
    for (j, (i, c)) in enumerate(items)
        b = block(ts[i], c)
        copyto!(view(A, axes(b, 1), axes(b, 2), j), b)
    end
    o1 = block(Ss[first(items)[1]], first(items)[2])
    Sb = similar(o1, min(m, n), length(items))
    MAK.svd_vals!(A, Sb, alg)
    for (j, (i, c)) in enumerate(items)
        o = block(Ss[i], c)
        copyto!(o, view(Sb, axes(o, 1), j))
    end
    return nothing
end

# Hook into the collection-level convergence API. Deliberately restricted to the generic
# CTMRG algorithms: `C4vCTMRG` overrides `corner_spectrum` to `eigh_vals` (its corners are
# diagonal), so a blanket override here would silently switch it back to `svd_vals`.
function PEPSKit.corner_spectra(
        Cs::AbstractArray{<:AbstractTensorMap},
        ::Union{PEPSKit.SequentialCTMRG, PEPSKit.SimultaneousCTMRG},
    )
    return _batched_spectra(Cs)
end
function PEPSKit.edge_spectra(
        Ts::AbstractArray{<:AbstractTensorMap},
        ::Union{PEPSKit.SequentialCTMRG, PEPSKit.SimultaneousCTMRG},
    )
    return _batched_spectra(Ts)
end

function _batched_spectra(ts::AbstractArray{T}) where {T <: AbstractTensorMap}
    # TODO BAD FIND A BETTER DISPATCH HERE
    (isempty(ts) || !(TensorKit.storagetype(T) <: AnyGPUArray)) && return map(svd_vals, ts)
    proto = nothing
    for i in eachindex(ts), c in blocksectors(ts[i])
        proto = block(ts[i], c)
        break
    end
    isnothing(proto) && return map(svd_vals, ts)
    alg = _batched_spectra_alg(proto)
    isnothing(alg) && return map(svd_vals, ts)
    tall = Factorizations.batched_requires_tall(alg)
    lim = Factorizations.max_batched_blocksize(alg, TensorKit.storagetype(T))

    Ss = map(
        t -> MAK.initialize_output(
            MAK.svd_vals!, t,
            MAK.default_algorithm(
                MAK.svd_vals!, typeof(t)
            )
        ), ts
    )

    # Group by block size because the decomposition doesn't care which sector a block came
    # from, so blocks of equal size can batch together even *across* sectors and tensors.
    # Each entry records the (tensor index, sector) it came from so the spectrum can be written back.
    I = eltype(eachindex(ts))
    C = sectortype(eltype(ts))
    groups = Dict{Tuple{Int, Int}, Vector{Tuple{I, C}}}()
    for i in eachindex(ts), c in blocksectors(ts[i])
        push!(get!(() -> Tuple{I, C}[], groups, size(block(ts[i], c))), (i, c))
    end

    # Groups that the solver can't work with, due to too few blocks, wider than
    # tall for an algo that needs m >= n, or larger than the solver's block limit,
    # join the padded batch below.
    small = Tuple{I, C}[]
    for ((m, n), items) in groups
        if length(items) >= BATCH_THRESHOLD && (!tall || m >= n) && max(m, n) <= lim
            _batch_svd_vals!(ts, Ss, items, (m, n), false, alg)
        else
            append!(small, items)
        end
    end

    # Groups too small to batch on their own are merged into one padded batch: zero padding
    # leaves a block's leading min(m, n) singular values untouched.
    mm = maximum(((i, c),) -> size(block(ts[i], c), 1), small; init = 0)
    nn = maximum(((i, c),) -> size(block(ts[i], c), 2), small; init = 0)
    # pad to a square only when the algo demands m >= n
    padded = tall ? (max(mm, nn), max(mm, nn)) : (mm, nn)
    if length(small) >= BATCH_THRESHOLD && maximum(padded) <= lim
        _batch_svd_vals!(ts, Ss, small, padded, true, alg)
    else
        for (i, c) in small
            b = block(ts[i], c)
            MAK.svd_vals!(b, block(Ss[i], c), MAK.default_svd_algorithm(typeof(b)))
        end
    end
    return Ss
end

end

module PEPSKitGPUArraysExt

using GPUArrays
using GPUArrays: AnyGPUArray, AllocCache
using PEPSKit
using TensorKit
using TensorKit: MatrixAlgebraKit as MAK

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

# Reduce into a 0-dimensional device array instead of returning a host scalar. `sdiag_pow` only
# feeds this into a broadcast, and a 0-dim array broadcasts as a scalar, so the value never has to
# come back to the host. Returning a number here would force a device sync on every call, and
# `sdiag_pow` runs once per bond per weight absorption in simple update.
function PEPSKit._maxabs(data::AnyGPUArray)
    T = real(eltype(data))
    acc = similar(data, T, ())
    fill!(acc, zero(T))
    Base.mapreducedim!(abs, max, acc, data)
    return acc
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


# Batched truncated SVD of a whole cluster's internal bonds to avoid multiple small kernel launches.
function PEPSKit.bond_svds(
        ::Type{<:AnyGPUArray}, rls::AbstractVector, truncs::AbstractVector
    )
    isempty(rls) && return map(_ -> nothing, rls)
    # The different GPU libaries offer different batching algos,
    # make sure we have one that actually works.
    alg = MAK.default_algorithm(MAK.batched_svd_compact!, eltype(rls))
    Fs = map(rl -> MAK.initialize_output(MAK.svd_compact!, rl, alg), rls)
    balg = _cluster_batched_alg(rls)
    if isnothing(balg)
        for (rl, F) in zip(rls, Fs)
            MAK.svd_compact!(rl, F, alg)
        end
    else
        # Pool every (bond, sector) block into one ragged batch. MatrixAlgebraKit batches
        # blocks of equal size together even across different bonds, since the decomposition
        # does not care which bond a block came from, and zero-pads the leftovers.
        items = [(i, c) for i in eachindex(rls) for c in blocksectors(rls[i])]
        As = [block(rls[i], c) for (i, c) in items]
        Us = [block(Fs[i][1], c) for (i, c) in items]
        Ss = [TensorKit.diagview(block(Fs[i][2], c)) for (i, c) in items]
        Vᴴs = [block(Fs[i][3], c) for (i, c) in items]
        MAK.batched_svd_compact!(As, (Us, Ss, Vᴴs), balg)
    end
    return map(Fs, truncs) do F, trunc
        (U, S, Vᴴ) = F
        USVᴴtrunc, ind = MAK.truncate(MAK.svd_trunc!, (U, S, Vᴴ), trunc)
        ϵ = MAK.truncation_error!(TensorKit.diagview(S), ind)
        return (USVᴴtrunc..., ϵ)
    end
end

"""
    CLUSTER_BATCHED_SVD[]

Whether simple update batches the SVDs of a cluster's internal bonds into one call.
Off by default.
"""
const CLUSTER_BATCHED_SVD = Ref(false)

# Which batched algorithm the backend offers for the cluster's blocks, or `nothing`.
function _cluster_batched_alg(rls::AbstractVector)
    CLUSTER_BATCHED_SVD[] || return nothing
    for i in eachindex(rls), c in blocksectors(rls[i])
        return _batched_spectra_alg(block(rls[i], c))
    end
    return nothing
end

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
    return alg isa MAK.AbstractAlgorithm ? alg : nothing
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

# `calc_convergence` decomposes every corner and every edge of the environment
# for regular CTMRG, which is expensive, at *least* 8 separate `svd_vals` calls.
# Across *multiple tensors* the situation is much better than within *one*,
# because the corners have to all share a space,
# so for a given sector their blocks have identical sizes and batch with no padding at all.
function _batched_spectra(ts::AbstractArray{T}) where {T <: AbstractTensorMap}
    # TODO BAD FIND A BETTER DISPATCH HERE
    (isempty(ts) || !(TensorKit.storagetype(T) <: AnyGPUArray)) && return map(svd_vals, ts)
    items = [(i, c) for i in eachindex(ts) for c in blocksectors(ts[i])]
    isempty(items) && return map(svd_vals, ts)
    alg = _batched_spectra_alg(block(ts[first(items)[1]], first(items)[2]))
    isnothing(alg) && return map(svd_vals, ts)

    Ss = map(
        t -> MAK.initialize_output(
            MAK.svd_vals!, t,
            MAK.default_algorithm(
                MAK.svd_vals!, typeof(t)
            )
        ), ts
    )
    # `batched_svd_vals!` destroys the blocks it has to decompose one at a time, but `ts` is the
    # live environment, and computing the convergence spectra must not damage the environment
    # it is measuring. Copying whole tensors costs one copy per tensor instead of one per block.
    ts′ = map(copy, ts)
    As = [block(ts′[i], c) for (i, c) in items]
    MAK.batched_svd_vals!(As, [block(Ss[i], c) for (i, c) in items], alg)
    return Ss
end

end

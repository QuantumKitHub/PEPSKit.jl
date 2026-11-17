"""Abstract runtime cache threaded between projector iterations."""
abstract type ProjectorCache end

"""Empty runtime cache used by projector algorithms without recycled data."""
struct EmptyProjectorCache <: ProjectorCache end

"""Allow convergence unless an adaptive projector still needs a charge-allocation check."""
_projector_convergence_ready(cache, alg) = true

"""
Runtime rangefinder cache for SI-CTMRG iterations.
`X` embeds the trial space into the full right space.
"""
struct SubspaceIterationCache{X} <: ProjectorCache
    iteration::Int
    X::X
end

"""Copy the SI tensor grid while preserving cache metadata."""
function Base.copy(cache::SubspaceIterationCache)
    return SubspaceIterationCache(cache.iteration, copy(cache.X))
end

"""Initialize an empty runtime projector cache for a fresh boundary contraction."""
initialize_projector_cache(network, env, alg::CTMRGAlgorithm) = EmptyProjectorCache()

"""Thread an unchanged empty projector cache through a three-argument CTMRG iteration."""
function ctmrg_iteration(network, env, alg::CTMRGAlgorithm, cache::EmptyProjectorCache)
    env′, info = ctmrg_iteration(network, env, alg)
    return env′, info, cache
end

"""Report whether a cached SI rangefinder matches the current map spaces and scalar type."""
function _compatible_rangefinder(X, L, R, subspace::ElementarySpace)
    return !isnothing(X) && scalartype(X) == scalartype(L) && space(X) == (domain(R) ← subspace)
end

"""Check grid layout and scalar type."""
function _compatible_subspace_cache(cache, env)
    return size(cache.X) == size(env) && scalartype(first(cache.X)) == scalartype(env)
end

"""Rotate a directional tensor grid without mutating arrays needed by backpropagation."""
_rotate_subspace_grid(grid) = reverse(permutedims(circshift(grid, (-1, 0, 0)), (1, 3, 2)); dims = 2)

"""Rotate the directional SI cache grid counter-clockwise."""
function Base.rotl90(cache::SubspaceIterationCache)
    return SubspaceIterationCache(cache.iteration, _rotate_subspace_grid(cache.X))
end

"""RMS sine of trial-space changes, computed without subtracting nearly equal norms."""
function _subspace_error(old, new)
    size(old) == size(new) || return Inf
    return maximum(zip(old, new)) do (a, b)
        space(a) == space(b) || return Inf
        norm(b - a * (a' * b)) / sqrt(dim(domain(b)))
    end
end
@non_differentiable _subspace_error(args...)

function _update_subspace_cache(cache, X)
    return SubspaceIterationCache(cache.iteration + 1, X), _subspace_error(cache.X, X)
end

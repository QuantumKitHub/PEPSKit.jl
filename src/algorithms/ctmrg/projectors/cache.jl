"""Abstract runtime cache threaded between projector iterations."""
abstract type ProjectorCache end

"""Empty runtime cache used by projector algorithms without recycled data."""
struct EmptyProjectorCache <: ProjectorCache end

"""Allow convergence unless an adaptive projector still needs a charge-allocation check."""
_projector_convergence_ready(cache, alg) = true

"""
Runtime rangefinder cache for SI-CTMRG iterations.
Both `X` and `Y` embed the trial space into the full right and left spaces, respectively.
"""
struct SubspaceIterationCache{X, Y} <: ProjectorCache
    iteration::Int
    X::X
    Y::Y
end

"""Copy all SI tensor grids while preserving cache metadata."""
function Base.copy(cache::SubspaceIterationCache)
    return SubspaceIterationCache(
        cache.iteration,
        map(_copy_subspace_grid, (cache.X, cache.Y))...,
    )
end

"""Initialize an empty runtime projector cache for a fresh boundary contraction."""
initialize_projector_cache(network, env, alg::CTMRGAlgorithm) = EmptyProjectorCache()

"""Thread an unchanged empty projector cache through a three-argument CTMRG iteration."""
function ctmrg_iteration(network, env, alg::CTMRGAlgorithm, cache::EmptyProjectorCache)
    env′, info = ctmrg_iteration(network, env, alg)
    return env′, info, cache
end

"""Report whether cached SI rangefinders match the current map spaces and scalar type."""
function _compatible_rangefinders(
        X, Y, halfinf_left, halfinf_right,
        subspace::ElementarySpace, require_left::Bool = true,
    )
    isnothing(X) && return false
    right = scalartype(X) == scalartype(halfinf_left) &&
        space(X) == (domain(halfinf_right) ← subspace)
    left = !require_left || (
        !isnothing(Y) &&
            scalartype(Y) == scalartype(halfinf_left) &&
            space(Y) == (codomain(halfinf_left) ← subspace)
    )
    return right && left
end

"""Copy an optional cache grid while leaving absent state unallocated."""
_copy_subspace_grid(grid) = isnothing(grid) ? nothing : copy(grid)

"""Index a trial grid, returning no left basis when that grid is absent."""
_subspace_grid_entry(grid, indices...) = isnothing(grid) ? nothing : grid[indices...]

"""Check grid layout, scalar type, and state required by the configured update."""
function _compatible_subspace_cache(cache, env, alg)
    return all(g -> isnothing(g) || size(g) == size(env), (cache.X, cache.Y)) &&
        scalartype(first(cache.X)) == scalartype(env) &&
        (!_retain_subspace_left(alg) || !isnothing(cache.Y))
end

"""Rotate a directional tensor grid without mutating arrays needed by backpropagation."""
_rotate_subspace_grid(grid) = isnothing(grid) ? nothing : reverse(permutedims(circshift(grid, (-1, 0, 0)), (1, 3, 2)); dims = 2)

"""Rotate all directional SI cache arrays counter-clockwise."""
function Base.rotl90(cache::SubspaceIterationCache)
    X′, Y′ = map(_rotate_subspace_grid, (cache.X, cache.Y))
    return SubspaceIterationCache(cache.iteration, X′, Y′)
end

"""RMS sine of trial-space changes, computed without subtracting nearly equal norms."""
_subspace_error(::Nothing, ::Nothing) = 0.0
function _subspace_error(old, new)
    size(old) == size(new) || return Inf
    return maximum(zip(old, new)) do (a, b)
        space(a) == space(b) || return Inf
        norm(b - a * (a' * b)) / sqrt(dim(domain(b)))
    end
end
@non_differentiable _subspace_error(args...)

function _update_subspace_cache(cache, X, Y)
    error = max(_subspace_error(cache.X, X), _subspace_error(cache.Y, Y))
    return SubspaceIterationCache(cache.iteration + 1, X, Y), error
end

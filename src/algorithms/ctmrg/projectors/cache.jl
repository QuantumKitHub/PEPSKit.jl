"""Abstract runtime cache threaded between projector iterations."""
abstract type ProjectorCache end

"""Empty runtime cache used by projector algorithms without recycled data."""
struct EmptyProjectorCache <: ProjectorCache end

"""
Runtime rangefinder and dominant-subspace cache for SI-CTMRG iterations.
Both `X` and `Y` embed the trial space into the full right and left spaces, respectively.
"""
struct SubspaceIterationCache{X, Y, U, V} <: ProjectorCache
    iteration::Int
    recycle::Bool
    X::X
    Y::Y
    U::U
    V::V
end

"""Copy all SI tensor grids while preserving cache metadata."""
function Base.copy(cache::SubspaceIterationCache)
    return SubspaceIterationCache(
        cache.iteration, cache.recycle,
        map(_copy_subspace_grid, (cache.X, cache.Y, cache.U, cache.V))...,
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
    return all(g -> isnothing(g) || size(g) == size(env), (cache.X, cache.Y, cache.U, cache.V)) &&
        scalartype(first(cache.X)) == scalartype(env) &&
        (!_retain_subspace_left(alg) || !isnothing(cache.Y)) &&
        (alg.subspace_tol == 0 || (!isnothing(cache.U) && !isnothing(cache.V)))
end

"""Update monitoring frames and reevaluate next-sweep range freezing."""
function _update_subspace_cache(cache, X, Y, U, V, alg, freezable)
    monitoring = alg.subspace_tol > 0
    error = monitoring && cache.iteration > 0 ? _subspace_error(cache.U, cache.V, U, V) : Inf
    iteration = cache.iteration + 1
    recycle = freezable && iteration >= alg.min_subspace_iters && error < alg.subspace_tol
    next = SubspaceIterationCache(
        iteration, recycle, X,
        _retain_subspace_left(alg) ? Y : nothing, monitoring ? U : nothing, monitoring ? V : nothing
    )
    return next, error
end

@doc raw"""
Measure the mean squared sine of the principal angles between two dominant subspaces.

For orthonormal bases ``Q_{\mathrm{old}}`` and ``Q_{\mathrm{new}}`` of dimension ``\chi``, the error is
```math
\epsilon_Q = \frac{1}{\chi} \sum_{i=1}^{\chi} \sin^2(\theta_i) = 1 - \frac{\lVert Q_{\mathrm{old}}^\dagger Q_{\mathrm{new}} \rVert_F^2}{\chi}.
```
The returned error is the maximum of the left and right dominant-subspace errors.
"""
function _subspace_error(
        U_old::AbstractTensorMap, V_old::AbstractTensorMap,
        U_new::AbstractTensorMap, V_new::AbstractTensorMap,
    )
    compatible = scalartype(U_old) == scalartype(U_new) == scalartype(V_old) == scalartype(V_new) &&
        space(U_old) == space(U_new) && space(V_old) == space(V_new)
    compatible || return Inf
    error_U = 1 - norm(U_old' * U_new)^2 / dim(domain(U_new))
    error_V = 1 - norm(V_old * V_new')^2 / dim(codomain(V_new))
    return max(zero(error_U), error_U, error_V)
end

"""Return the largest dominant-subspace error over a projector grid."""
function _subspace_error(U_old, V_old, U_new, V_new)
    any(isnothing, (U_old, V_old, U_new, V_new)) && return Inf
    size(U_old) == size(V_old) == size(U_new) == size(V_new) || return Inf
    return maximum(
        _subspace_error(U_old[i], V_old[i], U_new[i], V_new[i]) for i in eachindex(U_new)
    )
end

@non_differentiable _subspace_error(args...)

"""Rotate a directional tensor grid without mutating arrays needed by backpropagation."""
_rotate_subspace_grid(grid) = isnothing(grid) ? nothing : reverse(permutedims(circshift(grid, (-1, 0, 0)), (1, 3, 2)); dims = 2)

"""Rotate all directional SI cache arrays counter-clockwise."""
function Base.rotl90(cache::SubspaceIterationCache)
    X′, Y′, U′, V′ = map(_rotate_subspace_grid, (cache.X, cache.Y, cache.U, cache.V))
    return SubspaceIterationCache(cache.iteration, cache.recycle, X′, Y′, U′, V′)
end

"""Abstract runtime cache threaded between projector iterations."""
abstract type ProjectorCache end

"""Empty runtime cache used by projector algorithms without recycled data."""
struct EmptyProjectorCache <: ProjectorCache end

"""Runtime rangefinder and dominant-subspace cache for SI-CTMRG iterations."""
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
        copy(cache.X), copy(cache.Y), copy(cache.U), copy(cache.V),
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
        X, Y, halfinf_left::AbstractTensorMap, halfinf_right::AbstractTensorMap,
        subspace::ElementarySpace,
    )
    isnothing(X) && return false
    isnothing(Y) && return false
    return scalartype(X) == scalartype(halfinf_left) &&
        scalartype(Y) == scalartype(halfinf_left) &&
        space(X) == (domain(halfinf_right) ← subspace) &&
        space(Y) == (subspace ← codomain(halfinf_left))
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
    size(U_old) == size(V_old) == size(U_new) == size(V_new) || return Inf
    return maximum(
        _subspace_error(U_old[i], V_old[i], U_new[i], V_new[i]) for i in eachindex(U_new)
    )
end

"""Rotate all directional SI cache arrays counter-clockwise."""
function Base.rotl90(cache::SubspaceIterationCache)
    X′ = similar(cache.X, size(cache.X, 1), size(cache.X, 3), size(cache.X, 2))
    Y′ = similar(cache.Y, size(cache.Y, 1), size(cache.Y, 3), size(cache.Y, 2))
    U′ = similar(cache.U, size(cache.U, 1), size(cache.U, 3), size(cache.U, 2))
    V′ = similar(cache.V, size(cache.V, 1), size(cache.V, 3), size(cache.V, 2))
    for dir in axes(cache.X, 1)
        dir′ = _prev(dir, 4)
        X′[dir′, :, :] = rotl90(cache.X[dir, :, :])
        Y′[dir′, :, :] = rotl90(cache.Y[dir, :, :])
        U′[dir′, :, :] = rotl90(cache.U[dir, :, :])
        V′[dir′, :, :] = rotl90(cache.V[dir, :, :])
    end
    return SubspaceIterationCache(cache.iteration, cache.recycle, X′, Y′, U′, V′)
end

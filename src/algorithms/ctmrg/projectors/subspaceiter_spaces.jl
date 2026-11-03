"""Identify trial spaces whose charge allocation is selected from a sampled spectrum."""
_adaptive_subspace(alg::SubspaceIterationProjector) = alg.subspace isa Integer

"""Probe shared sectors up to the integer budget, preserving required fixed-space multiplicities."""
function _subspace_probe_space(available, alg, trunc)
    budget = Int(alg.subspace)
    required = trunc isa TruncationSpace ? trunc.space : zero(available)
    infimum(required, available) == required || throw(ArgumentError("The environment cannot contain the target truncation space."))
    dim(required) <= budget || throw(ArgumentError("The trial dimension must contain the target truncation space."))
    K = typeof(available)(s => min(dim(available, s), floor(Int, budget / dim(s))) for s in sectors(available))
    dim(K) > 0 || throw(ArgumentError("The trial budget cannot fit an available symmetry multiplet."))
    return K
end

"""Refresh shared charges periodically."""
_subspace_charge_refresh(cache, alg) = _adaptive_subspace(alg) && (cache.iteration == 0 || (alg.charge_refresh > 0 && cache.iteration % alg.charge_refresh == 0))

"""Confirm convergence on a charge-refresh sweep when periodic exploration is enabled."""
function _projector_convergence_ready(cache::SubspaceIterationCache, alg::SubspaceIterationProjector)
    return !_adaptive_subspace(alg) || alg.charge_refresh == 0 || (cache.iteration - 1) % alg.charge_refresh == 0
end

"""Reuse a valid allocation or probe every shared charge up to the total trial budget per sector."""
function _adaptive_subspace_space(L, R, alg, ranges)
    available = infimum(fuse(codomain(L)), fuse(domain(R)))
    budget = Int(alg.subspace)
    K = isnothing(ranges.X) ? zero(available) : fuse(domain(ranges.X))
    valid = _compatible_rangefinders(ranges.X, ranges.Y, L, R, K, _retain_subspace_left(alg)) &&
        infimum(K, available) == K && dim(K) <= budget
    if ranges.trunc isa TruncationSpace
        valid &= infimum(K, ranges.trunc.space) == ranges.trunc.space
    end
    refresh = !valid || get(ranges, :refresh, true)
    if refresh
        K = _subspace_probe_space(available, alg, ranges.trunc)
    end
    return K, refresh
end

"""Preserve compatible cached columns and add orthogonal Gaussian directions in enlarged sectors."""
function _resize_subspace_basis(X, output, K, alg, ::Type{T}) where {T}
    if isnothing(X) || scalartype(X) != T || codomain(X) != output
        return ignore_derivatives(() -> first(left_orth!(randn(alg.rng, T, output ← K), alg.orth_alg)))
    end
    fuse(domain(X)) == K && return X
    old, keep, added, extra = ignore_derivatives() do
        common = infimum(fuse(domain(X)), K)
        extra = K ⊖ common
        isometry(T, domain(X) ← common), isometry(T, K ← common), isometry(T, K ← extra), extra
    end
    U = X * old
    dim(extra) == 0 && return U * keep'
    Z = ignore_derivatives(() -> randn(alg.rng, T, output ← extra))
    Q, _ = left_orth!(Z - U * (U' * Z), alg.orth_alg)
    return U * keep' + Q * added'
end

"""Select reduced singular directions with fixed sector counts or a global truncation strategy."""
function _subspace_truncation_isometry(s, trunc, ::Type{T}) where {T}
    W = space(s, 1)
    trunc isa TruncationSpace && return isometry(T, W ← trunc.space)
    indices = MatrixAlgebraKit.findtruncated_svd(diagview(s), trunc)
    target = TensorKit.Factorizations.truncate_space(W, indices)
    dim(target) > 0 || throw(ArgumentError("Truncation retained no symmetry multiplet."))
    return TensorKit.Factorizations.truncate_domain!(zeros(T, W ← target), id(T, W), indices)
end

"""Keep the strongest sampled directions within the budget, reserving fixed target multiplicities."""
function _adaptive_subspace_isometry(s, alg, trunc, ::Type{T}) where {T}
    scores = trunc isa TruncationSpace ? copy(s) : s
    if trunc isa TruncationSpace
        for c in sectors(trunc.space)
            diagview(block(scores, c))[1:dim(trunc.space, c)] .= Inf
        end
    end
    return _subspace_truncation_isometry(scores, truncrank(Int(alg.subspace)), T)
end

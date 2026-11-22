"""
    SubspaceIteration(; subspace, n_subspace_iter=1, charge_refresh=10,
                      charge_probe=nothing, rotate_basis=true, rng=Random.default_rng())

Warm-started alternating subspace iteration for `SVDAdjoint(fwd_alg=...)`.
`subspace` is an elementary trial space or a positive total dimension budget.
Integer budgets discover charges initially and every `charge_refresh` decompositions;
`charge_probe` optionally limits added multiplicities per sector on later refreshes.
`n_subspace_iter` counts extra stabilized power steps (default: one).
CTMRG keeps a separate warm start for each cut and creates fresh caches for each contraction.
The default SVD reverse rule is GMRES, independent of the trial bases.
"""
struct SubspaceIteration{E, Q, R, C}
    subspace::E
    n_subspace_iter::Int
    charge_refresh::Int
    charge_probe::Union{Nothing, Int}
    rotate_basis::Bool
    orth_alg::Q
    rng::R
    cache::C
end

function SubspaceIteration(;
        subspace, n_subspace_iter::Int = 1, charge_refresh::Int = 10,
        charge_probe = nothing, rotate_basis::Bool = true,
        orth_alg = Householder(; positive = true), rng = Random.default_rng(),
    )
    subspace isa ElementarySpace || (subspace isa Integer && !(subspace isa Bool) && 0 < subspace <= typemax(Int)) ||
        throw(ArgumentError("subspace must be an elementary space or a positive integer dimension."))
    n_subspace_iter ≥ 0 || throw(ArgumentError("n_subspace_iter must be nonnegative."))
    charge_refresh ≥ 0 || throw(ArgumentError("charge_refresh must be nonnegative."))
    isnothing(charge_probe) || (charge_probe isa Int && charge_probe > 0) ||
        throw(ArgumentError("charge_probe must be nothing or a positive Int."))
    return SubspaceIteration(subspace, n_subspace_iter, charge_refresh, charge_probe, rotate_basis, orth_alg, rng, Ref{Any}(nothing))
end

SVD_FWD_SYMBOLS[:SubspaceIteration] = SubspaceIteration
_default_svd_rrule_alg(::SubspaceIteration) = :GMRES

# Ordinary matrix products represent tensor contractions once the internal dual legs are twisted.
struct FactorizedMap{F}
    factors::F
end
TensorKit.domain(f::FactorizedMap) = domain(last(f.factors))
TensorKit.codomain(f::FactorizedMap) = codomain(first(f.factors))
TensorKit.scalartype(f::FactorizedMap) = scalartype(first(f.factors))
TensorKit.storagetype(f::FactorizedMap) = storagetype(first(f.factors))
TensorKit.spacetype(f::FactorizedMap) = spacetype(first(f.factors))
TensorKit.TensorMap(f::FactorizedMap) = foldl(*, f.factors)
TensorKit.block(f::FactorizedMap, c) = FactorizedMap(map(a -> block(a, c), f.factors))
Base.copy(f::FactorizedMap) = f
Base.adjoint(f::FactorizedMap) = FactorizedMap(reverse(map(adjoint, f.factors)))
Base.:*(f::FactorizedMap, x::Union{AbstractTensorMap, AbstractVecOrMat}) = foldr(*, f.factors; init = x)
Base.:*(x::Union{AbstractTensorMap, AbstractVecOrMat}, f::FactorizedMap) = foldl(*, f.factors; init = x)
(f::FactorizedMap)(x, ::Val{false}) = f * x
(f::FactorizedMap)(x, ::Val{true}) = f' * x

_map_template(a::AbstractTensorMap) = a
_map_template(a::FactorizedMap) = first(a.factors)
_map_factors(a::AbstractTensorMap) = (a,)
_map_factors(a::FactorizedMap) = a.factors
function _factorized_environment(a, b)
    left, right = _map_factors(a), _map_factors(b)
    twisted = twistdual(first(right), codomainind(first(right)))
    return FactorizedMap((left..., twisted, Base.tail(right)...))
end

"""Use exact left bases in complete sectors instead of redundant QR factorizations."""
function _subspace_left_range(M, X, K, alg)
    W = fuse(codomain(M))
    sampled = typeof(K)(s => dim(K, s) for s in sectors(K) if dim(K, s) < dim(W, s))
    dim(sampled) == 0 && return isometry(scalartype(X), codomain(M) ← K), false
    E = sampled == K ? nothing : isometry(scalartype(X), K ← sampled)
    Q, _ = left_orth!(M * (isnothing(E) ? X : X * E), alg.orth_alg)
    isnothing(E) && return Q, true
    exact = typeof(K)(s => dim(K, s) for s in sectors(K) if dim(sampled, s) == 0)
    Q0, E0 = isometry(scalartype(X), codomain(M) ← exact), isometry(scalartype(X), K ← exact)
    return Q0 * E0' + Q * E', true
end

"""Compute a truncated SVD, recycling all trial directions rather than just the retained ones."""
function MatrixAlgebraKit.svd_trunc!(M, wrapped::TruncatedAlgorithm{<:SubspaceIteration})
    alg, trunc = wrapped.alg, wrapped.trunc
    _check_subspace_truncation(alg, trunc)
    available = infimum(fuse(codomain(M)), fuse(domain(M)))
    alg.subspace isa Integer || sectortype(alg.subspace) === sectortype(available) ||
        throw(ArgumentError("The map and subspace must use the same symmetry sectors."))
    T, A = scalartype(M), storagetype(M)
    Xtype = TensorMap{T, spacetype(M), length(domain(M)), 1, A}
    old = alg.cache[]
    iteration = isnothing(old) ? 0 : old.iteration::Int
    previous = isnothing(old) || !(old.X isa Xtype) ? nothing : old.X::Xtype
    K = isnothing(previous) ? zero(available) : fuse(domain(previous))
    compatible = !isnothing(previous) && scalartype(previous) == scalartype(M) && codomain(previous) == domain(M) && infimum(K, available) == K
    required = trunc isa NoTruncation ? available : trunc isa TruncationSpace ? trunc.space : zero(available)
    compatible &= infimum(required, K) == required
    adaptive = alg.subspace isa Integer
    refresh = adaptive && (!compatible || (alg.charge_refresh > 0 && iteration % alg.charge_refresh == 0))
    K = adaptive ? (refresh ? _subspace_probe_space(available, alg, trunc, compatible ? K : nothing) : K) : infimum(alg.subspace, available)
    infimum(required, K) == required || throw(ArgumentError("Subspace must contain the target truncation space."))
    X = _resize_subspace_basis(previous, domain(M), K, alg, scalartype(M))
    Q, sampled = _subspace_left_range(M, X, K, alg)
    for _ in 1:alg.n_subspace_iter
        sampled || break
        X, _ = left_orth!(M' * Q, alg.orth_alg)
        Q, _ = _subspace_left_range(M, X, K, alg)
    end
    B = Q' * M
    core, Z = sampled ? right_orth!(B, alg.orth_alg) : (B, nothing)
    # The small SVD keeps oversampling directions for the next decomposition.
    u, s, v = svd_compact!(core)
    E = _subspace_truncation_isometry(s, trunc, scalartype(M))
    U, S, V = Q * (u * E), DiagonalTensorMap(real(E' * s * E)), E' * v
    !isnothing(Z) && (V = V * Z)
    error = norm(s - E * S * E')
    X = isnothing(Z) ? copy(v') : (refresh || alg.rotate_basis ? Z' * v' : copy(Z'))
    refresh && (X = X * _adaptive_subspace_isometry(s, alg, trunc, scalartype(M)))
    subspace_error = isnothing(previous) || space(previous) != space(X) ? Inf : norm(X - previous * (previous' * X)) / sqrt(dim(domain(X)))
    alg.cache[] = (; X, Q, iteration = iteration + 1, subspace_error)
    Utype = TensorMap{T, spacetype(M), length(codomain(M)), 1, A}
    Vtype = TensorMap{T, spacetype(M), 1, length(domain(M)), A}
    Stype = DiagonalTensorMap{real(T), spacetype(M), similarstoragetype(_map_template(M), real(T))}
    return U::Utype, S::Stype, V::Vtype, error::real(T)
end
function MatrixAlgebraKit.svd_trunc_no_error!(M, alg::TruncatedAlgorithm{<:SubspaceIteration})
    return svd_trunc!(M, alg)[1:3]
end

function _check_subspace_truncation(alg, trunc)
    trunc isa Union{NoTruncation, TruncationSpace, MatrixAlgebraKit.TruncationByOrder} ||
        throw(ArgumentError("SubspaceIteration supports notrunc, truncspace, or truncrank."))
    if trunc isa Union{TruncationSpace, MatrixAlgebraKit.TruncationByOrder}
        trunc.by === abs && trunc.rev || throw(ArgumentError("SubspaceIteration retains the largest singular values."))
    end
    if trunc isa MatrixAlgebraKit.TruncationByOrder
        trunc.howmany > 0 || throw(ArgumentError("The retained rank must be positive."))
        alg.subspace isa Integer && alg.subspace < trunc.howmany && throw(ArgumentError("The trial dimension must be at least the retained rank."))
    end
    return nothing
end

"""Probe shared sectors broadly or add bounded multiplicities to a compatible allocation."""
function _subspace_probe_space(available, alg, trunc, current = nothing)
    budget = Int(alg.subspace)
    required = trunc isa NoTruncation ? available : trunc isa TruncationSpace ? trunc.space : zero(available)
    infimum(required, available) == required || throw(ArgumentError("The environment cannot contain the target truncation space."))
    dim(required) <= budget || throw(ArgumentError("The trial dimension must contain the target truncation space."))
    K = typeof(available)(
        s => begin
            cap = min(dim(available, s), floor(Int, budget / dim(s)))
            if isnothing(current) || isnothing(alg.charge_probe)
                cap
            else
                n = dim(current, s)
                max(dim(required, s), n + min(alg.charge_probe, cap - n))
            end
        end for s in sectors(available)
    )
    dim(K) > 0 || throw(ArgumentError("The trial budget cannot fit an available symmetry multiplet."))
    return K
end

"""Preserve compatible cached columns and add orthogonal Gaussian directions in enlarged sectors."""
function _resize_subspace_basis(X, output, K, alg, ::Type{T}) where {T}
    if isnothing(X) || scalartype(X) != T || codomain(X) != output
        return first(left_orth!(randn(alg.rng, T, output ← K), alg.orth_alg))
    end
    fuse(domain(X)) == K && return X
    old, keep, added, extra = let
        common = infimum(fuse(domain(X)), K)
        extra = K ⊖ common
        isometry(T, domain(X) ← common), isometry(T, K ← common), left_null(isometry(T, K ← common)), extra
    end
    U = X * old
    dim(extra) == 0 && return U * keep'
    Z = randn(alg.rng, T, output ← extra)
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

"""
Apply MatrixAlgebraKit's SVD phase convention with a differentiable correction.

The forward SVD fixes this gauge by default, but its pullback assumes gauge-invariant cotangents.
Keep the phase correction in the AD graph when lifting the singular vectors.
"""
function _subspace_svd_phase_correction(U::AbstractTensorMap, V::AbstractTensorMap, basis = U)
    layout = ignore_derivatives(() -> collect(values(TensorKit.blockstructure(space(basis)))))
    phase_blocks = map(layout) do (dims, inds)
        u = reshape(view(basis.data, inds), dims)
        pivots = ignore_derivatives(() -> [last(findmax(abs, col)) for col in eachcol(u)])
        return vec(diagm([sign(u[pivots[j], j]) for j in axes(u, 2)]))
    end
    phases = TensorMap{scalartype(U)}(reduce(vcat, phase_blocks), domain(U) ← domain(U))
    return (U * phases')::typeof(U), (phases * V)::typeof(V)
end


# Solve the complementary SVD response for all retained directions together.
# MatrixAlgebraKit handles the retained-space contribution and its gauge conventions.
function _subspace_svd_pullback_data(f, u, s, v, du, ds, dv, alg::GMRES)
    h, a, bx, by = MatrixAlgebraKit.check_and_prepare_svd_cotangents(
        u, s, v, du, Diagonal(ds), dv, length(s);
        degeneracy_atol = alg.tol, gauge_atol = _get_pullback_gauge_tol((du, ds, dv), alg.verbosity),
    )
    si = Diagonal(inv.(s))
    x0 = isnothing(bx) ? zero(u) : bx * si
    y0 = isnothing(by) ? zero(v') : by' * si
    project(q, z) = z - q * (q' * z)
    normal(y) = project(u, f(y, Val(false)))
    adjoint(x) = project(v', f(x, Val(true)))
    # Eliminate Y from X S - A⊥ Y = ΔU⊥ and Y S - A⊥' X = ΔV⊥.
    rhs = x0 + normal(y0) * si
    alg = @set alg.tol = alg.tol * max(norm(rhs), eps(real(eltype(s))))
    x, info = linsolve(rhs, zero(rhs), alg) do x
        x - normal(adjoint(x) * si) * si
    end
    info.converged == 0 && alg.verbosity > 0 &&
        @warn "SVD cotangent linear problem did not converge" info.normres
    y = y0 + adjoint(x) * si
    g = (h + a) / 2
    return x + u * g, y + v' * g'
end

# A factorized SVD uses thin pullbacks, without forming the full environment.
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(svd_trunc!),
        M::FactorizedMap, alg::SVDAdjoint{F, R},
    ) where {F, R <: Union{Arnoldi, GMRES, BiCGStab}}
    U, S, V, error = svd_trunc(M, alg)
    scale = norm(S)
    reverse_alg = alg.rrule_alg
    smallest = minimum(((_, b),) -> minimum(diag(b)), blocks(S)) / scale
    reverse_alg = @set reverse_alg.tol = clamp(reverse_alg.tol, eps(scalartype(S))^(3 / 4), 1.0e-2 * smallest)
    getdiag = reverse_alg isa GMRES ? diagview : _indexable_diagview
    function pullback(Δ)
        ΔU, ΔS, ΔV, = unthunk.(Δ)
        X, Y = zero(U), zero(V')
        for (c, s) in blocks(S)
            u, v = block(U, c), block(V, c)
            du, ds, dv = block(ΔU, c), block(ΔS, c), block(ΔV, c)
            values = getdiag(s) / scale
            dvalues = ds isa AbstractZero ? zero(values) : getdiag(ds) * scale
            m = block(M, c)
            f = (x, adj) -> m(x, adj) / scale
            if reverse_alg isa GMRES
                x, y = _subspace_svd_pullback_data(f, u, values, v, du, dvalues, dv, reverse_alg)
                copyto!(block(X, c), x / scale)
                copyto!(block(Y, c), y / scale)
                continue
            end
            left = _column_vectors(storagetype(U), u)
            right = _column_vectors(storagetype(V), v')
            dleft = du isa AbstractZero ? fill(ZeroTangent(), length(values)) : _column_vectors(storagetype(U), du)
            dright = dv isa AbstractZero ? fill(ZeroTangent(), length(values)) : _column_vectors(storagetype(V), dv')
            info = KrylovKit.ConvergenceInfo(length(values), nothing, nothing, -1, -1)
            primal = GKL(; tol = reverse_alg.tol, verbosity = reverse_alg.verbosity)
            xs, ys = KrylovKitCRCExt.compute_svdsolve_pullback_data(
                dvalues, dleft, dright, values, left, right, info,
                f, :LR, primal, reverse_alg,
            )
            copyto!(block(X, c), stack(xs) / scale)
            copyto!(block(Y, c), stack(ys) / scale)
        end
        dnormal, _ = _factorized_pullback(M.factors, V', X)
        dadjoint, _ = _factorized_pullback(adjoint(M).factors, U, Y)
        factors = map(+, dnormal, reverse(map(adjoint, dadjoint)))
        return NoTangent(), Tangent{typeof(M)}(; factors), NoTangent()
    end
    return (U, S, V, error), pullback
end

# Expert-mode TruncPullback can reuse the dense Sylvester implementation.
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(svd_trunc!),
        M::FactorizedMap, alg::SVDAdjoint{F, R},
    ) where {F, R <: TruncPullback}
    U, S, V, error = svd_trunc(M, alg)
    dense, materialize_pullback = rrule_via_ad(config, TensorMap, M)
    function pullback(Δ)
        dUSV = unthunk.(Δ)
        dM = svd_trunc_pullback!(
            zero(dense), dense, (U, S, V), dUSV;
            gauge_atol = _get_pullback_gauge_tol(dUSV, alg.rrule_alg.verbosity), degeneracy_atol = alg.rrule_alg.degeneracy_atol
        )
        return NoTangent(), materialize_pullback(dM)[2], NoTangent()
    end
    return (U, S, V, error), pullback
end

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(svd_trunc_no_error!),
        M::FactorizedMap, alg::SVDAdjoint,
    )
    result, pullback = rrule(config, svd_trunc!, M, alg)
    return result[1:3], Δ -> pullback((Δ..., ZeroTangent()))
end


# Pull back thin applications without materializing the product or differentiating tuple folds.
function _factorized_pullback(factors, X, ΔY)
    rights = reverse((X, accumulate((x, a) -> a * x, reverse(Base.tail(factors)); init = X)...))
    gradients = ()
    for j in eachindex(factors)
        gradients = (gradients..., ProjectTo(factors[j])(ΔY * rights[j]'))
        ΔY = factors[j]' * ΔY
    end
    return gradients, ProjectTo(X)(ΔY)
end
function ChainRulesCore.rrule(::typeof(*), M::FactorizedMap, X::AbstractTensorMap)
    function pullback(ΔY)
        factors, ΔX = _factorized_pullback(M.factors, X, unthunk(ΔY))
        return NoTangent(), Tangent{typeof(M)}(; factors), ΔX
    end
    return M * X, pullback
end
function ChainRulesCore.rrule(::typeof(*), X::AbstractTensorMap, M::FactorizedMap)
    function pullback(ΔY)
        gradients, ΔX = _factorized_pullback(adjoint(M).factors, X', unthunk(ΔY)')
        factors = reverse(map(adjoint, gradients))
        return NoTangent(), ΔX', Tangent{typeof(M)}(; factors)
    end
    return X * M, pullback
end

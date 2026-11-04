"""
$(TYPEDEF)

Recycled subspace approximation of the full-infinite CTMRG environment.
Supply `subspace` as an elementary space with explicit charge multiplicities, or as a positive integer specifying the total trial dimension.
An integer budget selects its charge distribution from the sampled singular spectrum, preserving complete multiplets.
`trunc` independently selects the retained environment: [`FixedSpaceTruncation`](@ref), `truncspace`, or `truncrank`.
With fixed target sectors, the trial allocation reserves their required multiplicities.
`charge_probe=nothing` uses broad exploration; a positive integer limits added multiplicities per sector on compatible-cache refreshes.
Initialization and incompatible-cache recovery remain broad.
Integer allocations are explored initially and every `charge_refresh=$(Defaults.charge_refresh)` sweeps.
Set `charge_refresh=0` to disable periodic exploration; incompatible cached spaces still trigger exploration.
Refreshes select trial directions in reduced SVD coordinates even when ordinary basis rotations are disabled.
See [Subspace-iteration CTMRG projectors](@ref subspace_iteration) for update equations and convergence.
Both trial bases embed the reduced space into the full environment spaces: ``X^{\\dagger} X = Y^{\\dagger} Y = I`` and ``ρ = Y^{\\dagger} M X``.

The `update` selector chooses `:independent` (default, the paper's two-factorization independent update), `:alternating` (alternating QR/LQ with a reused reduced factor), or `:alternating_refined` (three factorizations with a final left-basis refresh).
`n_subspace_iter=q` adds `q` independent updates, or `q` stabilized right-basis power steps before the mandatory alternating update.
`rotate_basis=true` expresses the complete trial bases in the reduced SVD coordinates, retaining all oversampling directions.

[`FixedPointGradient`](@ref) differentiates the joint environment/rangefinder fixed point, holding only the charge allocation fixed.
Use `gradient_alg=nothing` in [`PEPSOptimize`](@ref) to differentiate through all sweeps and recycled bases.

## Fields

$(TYPEDFIELDS)

## Constructors

    SubspaceIterationProjector(; subspace, kwargs...)

Defaults are `update=:independent`, `n_subspace_iter=0`, `rotate_basis=true`, `rng=Random.default_rng()`, `decomposition_alg=SVDAdjoint()`, `orth_alg=OrthAdjoint()`, and `trunc=FixedSpaceTruncation()`.
"""
struct SubspaceIterationProjector{M, S <: SVDAdjoint, T, Q <: OrthAdjoint, E <: Union{ElementarySpace, Integer}, R} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    orth_alg::Q
    subspace::E
    charge_refresh::Int
    charge_probe::Union{Nothing, Int}
    verbosity::Int
    update::Symbol
    n_subspace_iter::Int
    rotate_basis::Bool
    rng::R
end

function SubspaceIterationProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :SubspaceIterationProjector, trunc = FixedSpaceTruncation(), kwargs...)
end

"""Specialize the range kernel on the selected update without additional public algorithm types."""
function SubspaceIterationProjector(svd, trunc, orth, space, refresh, probe, verbosity, update::Symbol, q::Int, rotate::Bool, rng)
    return SubspaceIterationProjector{update, typeof(svd), typeof(trunc), typeof(orth), typeof(space), typeof(rng)}(
        svd, trunc, orth, space, refresh, probe, verbosity, update, q, rotate, rng
    )
end

PROJECTOR_SYMBOLS[:SubspaceIterationProjector] = SubspaceIterationProjector

_retain_subspace_left(alg::SubspaceIterationProjector{M}) where {M} = M == :independent

"""Apply a half environment through its enlarged corners without forming their product."""
struct SIHalfEnvironment{L, R, S}
    left::L
    right::R
    scale::S
end

"""Normalize an implicit half without copying its enlarged corners."""
SIHalfEnvironment(left, right) = SIHalfEnvironment(left, right, inv(norm(left) * norm(right)))

TensorKit.domain(h::SIHalfEnvironment) = domain(h.right)
TensorKit.codomain(h::SIHalfEnvironment) = codomain(h.left)
TensorKit.scalartype(h::SIHalfEnvironment) = scalartype(h.left)
Base.adjoint(h::SIHalfEnvironment) = SIHalfEnvironment(h.right', h.left', h.scale)
Base.:*(h::SIHalfEnvironment, x::AbstractTensorMap) = scale!!(h.left * (h.right * x), h.scale)
Base.:*(x::AbstractTensorMap, h::SIHalfEnvironment) = scale!!((x * h.left) * h.right, h.scale)
Base.:*(h::SIHalfEnvironment{<:Any, <:Any, Nothing}, x::AbstractTensorMap) = h.left * (h.right * x)
Base.:*(x::AbstractTensorMap, h::SIHalfEnvironment{<:Any, <:Any, Nothing}) = (x * h.left) * h.right

"""Normalize unscaled corner halves while preserving explicitly supplied maps."""
_normalize_subspace_half(h) = h
_normalize_subspace_half(h::SIHalfEnvironment{<:Any, <:Any, Nothing}) = SIHalfEnvironment(h.left, h.right)

"""Check compatible network, environment, and explicitly specified trial sectors."""
function _check_subspace_iteration_input(network, env, alg::SubspaceIterationProjector)
    sector = sectortype(spacetype(network))
    sectortype(spacetype(env)) === sector &&
        (!(alg.subspace isa ElementarySpace) || sectortype(alg.subspace) === sector) ||
        throw(ArgumentError("Network, environment, and subspace must use the same symmetry sectors."))
    return nothing
end

"""Select trial multiplicities from retained charges or an explicit space, clipped to both map dimensions."""
function _subspace_iteration_space(left, right, alg, trunc)
    available = infimum(fuse(left), fuse(right))
    _adaptive_subspace(alg) && return _subspace_probe_space(available, alg, trunc)
    K = infimum(alg.subspace, available)
    !(trunc isa TruncationSpace) || infimum(trunc.space, K) == trunc.space ||
        throw(ArgumentError("Subspace must contain the target truncation space."))
    return K
end

"""Initialize right and required left isometries from the trial space into the full spaces."""
function rangefinder_initialize(::Type{T}, left::ProductSpace, right::ProductSpace, K::ElementarySpace, alg::SubspaceIterationProjector{M}) where {T, M}
    X, _ = left_orth!(randn(alg.rng, T, right ← K), alg.orth_alg)
    Y = if M == :independent
        last(right_orth!(randn(alg.rng, T, K ← left), alg.orth_alg))'
    else
        nothing
    end
    return X, Y
end

"""Initialize trial bases from factorized half-environment spaces."""
function rangefinder_initialize(L, R, K::ElementarySpace, alg::SubspaceIterationProjector)
    return rangefinder_initialize(scalartype(L), codomain(L), domain(R), K, alg)
end

"""Initialize one directional cache entry without forming enlarged corners."""
function _initialize_subspace_iteration_entry(network::InfiniteSquareNetwork, env::CTMRGEnv, coordinate::NTuple{3, Int}, alg::SubspaceIterationProjector)
    coordinates, trunc = _enlarged_corner_layout(coordinate, env, alg, Val(4))
    left = codomain(EnlargedCorner(network, env, first(coordinates)))
    right = domain(EnlargedCorner(network, env, last(coordinates)))
    K = _subspace_iteration_space(left, right, alg, trunc)
    T = scalartype(network)
    X, Y = rangefinder_initialize(T, left, right, K, alg)
    return (; X, Y)
end

"""Initialize only the trial grids required by the selected update."""
function initialize_subspace_iteration_cache(network::InfiniteSquareNetwork, env::CTMRGEnv, alg::SubspaceIterationProjector)
    entries = map(c -> _initialize_subspace_iteration_entry(network, env, c, alg), eachcoordinate(env, 1:4))
    Y = _retain_subspace_left(alg) ? map(x -> x.Y, entries) : nothing
    return SubspaceIterationCache(0, map(x -> x.X, entries), Y)
end

@non_differentiable initialize_subspace_iteration_cache(args...)

"""Construct a sampled left range and exact bases for complete left sectors."""
function _subspace_alternating_range(L, R, X, K, alg)
    W = ignore_derivatives(() -> fuse(codomain(L)))
    sampled = ignore_derivatives(() -> Vect[sectortype(K)](s => dim(K, s) for s in sectors(K) if dim(K, s) < dim(W, s)))
    dim(sampled) == 0 && return ignore_derivatives(() -> isometry(scalartype(L), codomain(L) ← K)), false
    E = sampled == K ? nothing : ignore_derivatives(() -> isometry(scalartype(X), K ← sampled))
    Xsampled = isnothing(E) ? X : X * E
    Q, _ = left_orth!(L * (R * Xsampled), alg.orth_alg)
    isnothing(E) && return Q, true
    Q0, E0 = ignore_derivatives() do
        exact = Vect[sectortype(K)](s => dim(K, s) for s in sectors(K) if dim(sampled, s) == 0)
        isometry(scalartype(L), codomain(L) ← exact), isometry(scalartype(L), K ← exact)
    end
    return Q0 * E0' + Q * E', true
end

"""
Update the trial bases and construct `core`, the reduced matrix ``ρ = Y^{\\dagger} L R X``.
Use independent normal-map updates or alternating QR steps with a final LQ or refined QR factor reuse.
"""
function rangefinder_update(X::AbstractTensorMap, Y, L, R, K, alg::SubspaceIterationProjector{M}) where {M}
    if M == :independent
        if ignore_derivatives(() -> fuse(domain(X)) == fuse(domain(R)) == fuse(codomain(L)))
            X, Y = ignore_derivatives() do
                isometry(storagetype(X), domain(R) ← K), isometry(storagetype(X), codomain(L) ← K)
            end
        else
            for _ in 0:alg.n_subspace_iter
                MX = L * (R * X)
                Xnew = R' * (L' * MX)
                # Adjoint LQ evaluates leftorth(M M'Y) in the cached row layout.
                YadjM = (Y' * L) * R
                _, Yrow = right_orth!((YadjM * R') * L', alg.orth_alg)
                Y = Yrow'
                X, _ = left_orth!(Xnew, alg.orth_alg)
            end
        end
        right = R * X
        left = Y' * L
        core = left * right
    else
        # Initialize Q, then perform q right/left update pairs.
        Q, sampled = if M == :alternating
            _subspace_alternating_range(L, R, X, K, alg)
        else
            Q, _ = left_orth!(L * (R * X), alg.orth_alg)
            Q, true
        end
        for _ in 1:alg.n_subspace_iter
            sampled || break
            X, _ = left_orth!(R' * (L' * Q), alg.orth_alg)
            Q, _ = if M == :alternating
                _subspace_alternating_range(L, R, X, K, alg)
            else
                left_orth!(L * (R * X), alg.orth_alg)
            end
        end
        if M == :alternating
            # The LQ of Q'M is the adjoint of the final QR; its triangular factor is ρ.
            Y = Q
            left = Y' * L
            B = left * R
            core, Z = sampled ? right_orth!(B, alg.orth_alg) : (B, nothing)
            X = isnothing(Z) ? nothing : Z'
            right = nothing
        else
            X, _ = left_orth!(R' * (L' * Q), alg.orth_alg)
            right = R * X
            Y, core = left_orth!(L * right, alg.orth_alg)
            left = Y' * L
        end
    end
    return (; X, Y, left, right, core)
end

"""
Apply MatrixAlgebraKit's SVD phase convention with a differentiable correction.

The forward SVD fixes this gauge by default, but its pullback assumes gauge-invariant cotangents.
Keep the phase correction in the AD graph to support phase-sensitive singular-vector outputs.
"""
function _subspace_svd_phase_correction(U::AbstractTensorMap, V::AbstractTensorMap)
    layout = ignore_derivatives(() -> collect(values(TensorKit.blockstructure(space(U)))))
    phase_blocks = map(layout) do (dims, inds)
        u = reshape(view(U.data, inds), dims)
        pivots = ignore_derivatives(() -> [last(findmax(abs, col)) for col in eachcol(u)])
        return vec(diagm([sign(u[pivots[j], j]) for j in axes(u, 2)]))
    end
    phases = TensorMap{scalartype(U)}(reduce(vcat, phase_blocks), domain(U) ← domain(U))
    return U * phases', phases * V
end

"""
Factor `core`, the reduced matrix ``ρ = Y^{\\dagger} M X``, retaining complete singular directions when recycling needs them.
The `:alternating` update normalizes this matrix before its SVD.
"""
function _subspace_reduced_svd(core, trunc, alg, complete::Bool)
    svd_alg = decomposition_algorithm(alg)
    fwd = svd_alg.fwd_alg
    fwd = fwd isa TruncatedAlgorithm ? fwd.alg : fwd
    # TODO: Unify alternating core normalization with the other updates' half-environment scaling;
    # verify projector scaling, convergence, and gradients before changing the conventions.
    rho = alg.update == :alternating ? core / norm(core) : core
    if complete
        svd_alg = @set svd_alg.fwd_alg = TruncatedAlgorithm(fwd, notrunc())
        u, s, v = svd_trunc_no_error!(rho, svd_alg)
        u, v = _subspace_svd_phase_correction(u, v)
        E = ignore_derivatives(() -> _subspace_truncation_isometry(s, trunc, scalartype(core)))
        uχ, sχ, vχ = u * E, DiagonalTensorMap(E' * s * E), E' * v
        error = ignore_derivatives(() -> norm(s - E * sχ * E') / norm(sχ))
        return (; u, s, v, uχ, sχ, vχ, error)
    end
    svd_alg = @set svd_alg.fwd_alg = TruncatedAlgorithm(fwd, trunc)
    u, s, v, error = svd_trunc!(rho, svd_alg)
    u, v = _subspace_svd_phase_correction(u, v)
    error = ignore_derivatives(() -> error / norm(s))
    return (; u, s, v, uχ = u, sχ = s, vχ = v, error)
end

"""
Compute shared projectors, singular frames, and the minimal next-sweep trial state.
The internal `core` is the reduced matrix ``ρ = Y^{\\dagger} M X`` with ``M = L R``, obtained by contraction or a reused QR/LQ factor.
"""
function _subspace_iteration_decomposition(L, R, alg::SubspaceIterationProjector{M}, ranges) where {M}
    trunc = ranges.trunc
    K, refresh = ignore_derivatives() do
        return _adaptive_subspace(alg) ? _adaptive_subspace_space(L, R, alg, ranges) :
            (_subspace_iteration_space(codomain(L), domain(R), alg, trunc), false)
    end
    compatible = _compatible_rangefinders(ranges.X, ranges.Y, L, R, K, _retain_subspace_left(alg))
    X, Y = if refresh
        X = _resize_subspace_basis(ranges.X, domain(R), K, alg, scalartype(L))
        Y = _retain_subspace_left(alg) ? copy(_resize_subspace_basis(ranges.Y, codomain(L), K, alg, scalartype(L))')' : nothing
        X, Y
    else
        compatible ? (ranges.X, ranges.Y) : ignore_derivatives(() -> rangefinder_initialize(L, R, K, alg))
    end
    if M != :alternating
        L, R = _normalize_subspace_half(L), _normalize_subspace_half(R)
    end
    state = rangefinder_update(X, Y, L, R, K, alg)
    full = _subspace_reduced_svd(state.core, trunc, alg, refresh || alg.rotate_basis || isnothing(state.X))
    U = state.Y * full.uχ
    V = isnothing(state.X) ? full.vχ : full.vχ * state.X'
    isqS = sdiag_pow(full.sχ, -0.5)
    P_left = isnothing(state.right) ? R * (V' * isqS) : state.right * (full.vχ' * isqS)
    P_right = (isqS * full.uχ') * state.left
    rotate = refresh || alg.rotate_basis
    X = isnothing(state.X) ? copy(full.v') : (rotate ? state.X * full.v' : copy(state.X))
    # Keep the left cache as a lazy adjoint of the existing row-oriented contraction.
    Y = _retain_subspace_left(alg) ? (rotate ? full.u' * state.Y' : copy(state.Y'))' : nothing
    if refresh
        E = ignore_derivatives(() -> _adaptive_subspace_isometry(full.s, alg, trunc, scalartype(state.core)))
        X = X * E
        Y = isnothing(Y) ? nothing : (E' * Y')'
    end
    return (; U, S = full.sχ, V, X, Y, truncation_error = full.error, projectors = (P_left, P_right))
end

"""Build factorized half environments using the established scaling for each update."""
function compute_projector(corners, alg::SubspaceIterationProjector, ranges)
    L = SIHalfEnvironment(corners[1], corners[2], nothing)
    R = SIHalfEnvironment(corners[3], corners[4], nothing)
    info = _subspace_iteration_decomposition(L, R, alg, ranges)
    return info.projectors, info
end

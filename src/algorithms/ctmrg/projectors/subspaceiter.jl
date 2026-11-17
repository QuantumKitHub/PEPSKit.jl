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

Alternating QR/LQ updates reuse the final triangular factor as the reduced environment.
`n_subspace_iter=q` adds `q` stabilized right-basis power steps before the final alternating update (default: `q=1`).
`rotate_basis=true` expresses the cached right trial basis in the reduced SVD coordinates, retaining all oversampling directions.

[`FixedPointGradient`](@ref) differentiates the joint environment/rangefinder fixed point, holding only the charge allocation fixed.
Use `gradient_alg=nothing` in [`PEPSOptimize`](@ref) to differentiate through all sweeps and recycled bases.

## Fields

$(TYPEDFIELDS)

## Constructors

    SubspaceIterationProjector(; subspace, kwargs...)

Defaults are `n_subspace_iter=1`, `rotate_basis=true`, `rng=Random.default_rng()`, `decomposition_alg=SVDAdjoint()`, `orth_alg=OrthAdjoint()`, and `trunc=FixedSpaceTruncation()`.
"""
struct SubspaceIterationProjector{S <: SVDAdjoint, T, Q <: OrthAdjoint, E <: Union{ElementarySpace, Integer}, R} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    orth_alg::Q
    subspace::E
    charge_refresh::Int
    charge_probe::Union{Nothing, Int}
    verbosity::Int
    n_subspace_iter::Int
    rotate_basis::Bool
    rng::R
end

function SubspaceIterationProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :SubspaceIterationProjector, trunc = FixedSpaceTruncation(), kwargs...)
end

PROJECTOR_SYMBOLS[:SubspaceIterationProjector] = SubspaceIterationProjector

"""Apply a half environment through its enlarged corners without forming their product."""
struct SIHalfEnvironment{L, R}
    left::L
    right::R
end

TensorKit.domain(h::SIHalfEnvironment) = domain(h.right)
TensorKit.codomain(h::SIHalfEnvironment) = codomain(h.left)
TensorKit.scalartype(h::SIHalfEnvironment) = scalartype(h.left)
Base.adjoint(h::SIHalfEnvironment) = SIHalfEnvironment(h.right', h.left')
Base.:*(h::SIHalfEnvironment, x::AbstractTensorMap) = h.left * (h.right * x)
Base.:*(x::AbstractTensorMap, h::SIHalfEnvironment) = (x * h.left) * h.right

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

"""Initialize a right isometry from the trial space into the full space."""
function rangefinder_initialize(::Type{T}, right::ProductSpace, K::ElementarySpace, alg::SubspaceIterationProjector) where {T}
    return first(left_orth!(randn(alg.rng, T, right ← K), alg.orth_alg))
end

"""Initialize a trial basis from factorized half-environment spaces."""
function rangefinder_initialize(L, R, K::ElementarySpace, alg::SubspaceIterationProjector)
    return rangefinder_initialize(scalartype(L), domain(R), K, alg)
end

"""Initialize one directional cache entry without forming enlarged corners."""
function _initialize_subspace_iteration_entry(network::InfiniteSquareNetwork, env::CTMRGEnv, coordinate::NTuple{3, Int}, alg::SubspaceIterationProjector)
    coordinates, trunc = _enlarged_corner_layout(coordinate, env, alg, Val(4))
    left = codomain(EnlargedCorner(network, env, first(coordinates)))
    right = domain(EnlargedCorner(network, env, last(coordinates)))
    K = _subspace_iteration_space(left, right, alg, trunc)
    T = scalartype(network)
    return rangefinder_initialize(T, right, K, alg)
end

"""Initialize the right trial grid."""
function initialize_subspace_iteration_cache(network::InfiniteSquareNetwork, env::CTMRGEnv, alg::SubspaceIterationProjector)
    X = map(c -> _initialize_subspace_iteration_entry(network, env, c, alg), eachcoordinate(env, 1:4))
    return SubspaceIterationCache(0, X)
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
Alternate stabilized QR steps and reuse the final LQ factor.
"""
function rangefinder_update(X::AbstractTensorMap, L, R, K, alg::SubspaceIterationProjector)
    Q, sampled = _subspace_alternating_range(L, R, X, K, alg)
    for _ in 1:alg.n_subspace_iter
        sampled || break
        X, _ = left_orth!(R' * (L' * Q), alg.orth_alg)
        Q, _ = _subspace_alternating_range(L, R, X, K, alg)
    end
    # The LQ of Q'M is the adjoint of the final QR; its triangular factor is ρ.
    left = Q' * L
    B = left * R
    core, Z = sampled ? right_orth!(B, alg.orth_alg) : (B, nothing)
    X = isnothing(Z) ? nothing : Z'
    return (; X, Y = Q, left, core)
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
Normalize this matrix before its SVD.
"""
function _subspace_reduced_svd(core, trunc, alg, complete::Bool)
    svd_alg = decomposition_algorithm(alg)
    fwd = svd_alg.fwd_alg
    fwd = fwd isa TruncatedAlgorithm ? fwd.alg : fwd
    rho = core / norm(core)
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
function _subspace_iteration_decomposition(L, R, alg::SubspaceIterationProjector, ranges)
    trunc = ranges.trunc
    K, refresh = ignore_derivatives() do
        return _adaptive_subspace(alg) ? _adaptive_subspace_space(L, R, alg, ranges) :
            (_subspace_iteration_space(codomain(L), domain(R), alg, trunc), false)
    end
    X = if refresh
        _resize_subspace_basis(ranges.X, domain(R), K, alg, scalartype(L))
    else
        _compatible_rangefinder(ranges.X, L, R, K) ? ranges.X : ignore_derivatives(() -> rangefinder_initialize(L, R, K, alg))
    end
    state = rangefinder_update(X, L, R, K, alg)
    full = _subspace_reduced_svd(state.core, trunc, alg, refresh || alg.rotate_basis || isnothing(state.X))
    U = state.Y * full.uχ
    V = isnothing(state.X) ? full.vχ : full.vχ * state.X'
    isqS = sdiag_pow(full.sχ, -0.5)
    P_left = R * (V' * isqS)
    P_right = (isqS * full.uχ') * state.left
    rotate = refresh || alg.rotate_basis
    X = isnothing(state.X) ? copy(full.v') : (rotate ? state.X * full.v' : copy(state.X))
    if refresh
        E = ignore_derivatives(() -> _adaptive_subspace_isometry(full.s, alg, trunc, scalartype(state.core)))
        X = X * E
    end
    return (; U, S = full.sχ, V, X, truncation_error = full.error, projectors = (P_left, P_right))
end

"""Build factorized half environments without forming their products."""
function compute_projector(corners, alg::SubspaceIterationProjector, ranges)
    L = SIHalfEnvironment(corners[1], corners[2])
    R = SIHalfEnvironment(corners[3], corners[4])
    info = _subspace_iteration_decomposition(L, R, alg, ranges)
    return info.projectors, info
end

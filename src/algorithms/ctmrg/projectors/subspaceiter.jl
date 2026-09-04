"""
$(TYPEDEF)

Projector algorithm using recycled subspace iteration to approximate the full-infinite CTMRG environment.

This implementation currently supports only tensors with trivial symmetry sectors, [`FixedSpaceTruncation`](@ref), and non-differentiated CTMRG contractions.

## Fields

$(TYPEDFIELDS)

## Constructors

    SubspaceIterationProjector(; kwargs...)

Construct a subspace-iteration projector for the supplied oversampled elementary `subspace`, with `subspace_tol=$(Defaults.subspace_tol)`, `min_subspace_iters=$(Defaults.min_subspace_iters)`, `decomposition_alg=SVDAdjoint()`, `orth_alg=QRAdjoint()`, `trunc=FixedSpaceTruncation()`, and the standard projector verbosity.
"""
struct SubspaceIterationProjector{
        S <: SVDAdjoint, T, Q <: QRAdjoint, E <: ElementarySpace,
    } <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    orth_alg::Q
    subspace::E
    subspace_tol::Float64
    min_subspace_iters::Int
    verbosity::Int
end
function SubspaceIterationProjector(; kwargs...)
    defaults = (; trunc = FixedSpaceTruncation())
    return ProjectorAlgorithm(; alg = :SubspaceIterationProjector, defaults..., kwargs...)
end

PROJECTOR_SYMBOLS[:SubspaceIterationProjector] = SubspaceIterationProjector

"""Check the v1 symmetry, truncation, and differentiation limitations of SI-CTMRG."""
function _check_subspace_iteration_input(network, env, alg::SubspaceIterationProjector)
    alg.trunc isa FixedSpaceTruncation ||
        throw(ArgumentError("SubspaceIterationProjector only supports FixedSpaceTruncation."))
    sectortype(spacetype(network)) === Trivial ||
        throw(ArgumentError("SubspaceIterationProjector currently supports only trivial symmetry sectors."))
    sectortype(spacetype(env)) === Trivial ||
        throw(ArgumentError("SubspaceIterationProjector currently supports only trivial symmetry sectors."))
    Zygote.isderiving() &&
        throw(ArgumentError("SubspaceIterationProjector does not currently support automatic differentiation."))
    return nothing
end

"""Initialize orthonormal TensorMap rangefinders from the two half-environment spaces."""
function rangefinder_initialize(
        ::Type{T}, leftspace::ProductSpace, rightspace::ProductSpace,
        subspace::ElementarySpace, alg::SubspaceIterationProjector,
    ) where {T}
    X = randn(T, rightspace ← subspace)
    Y = randn(T, subspace ← leftspace)
    X, _ = left_orth!(X, alg.orth_alg)
    _, Y = right_orth!(Y, alg.orth_alg)
    return X, Y
end

"""Initialize orthonormal TensorMap rangefinders for an SI projector update."""
function rangefinder_initialize(
        halfinf_left::AbstractTensorMap, halfinf_right::AbstractTensorMap,
        subspace::ElementarySpace, alg::SubspaceIterationProjector,
    )
    return rangefinder_initialize(
        scalartype(halfinf_left), codomain(halfinf_left), domain(halfinf_right),
        subspace, alg,
    )
end

"""Initialize one SI cache entry from enlarged-corner boundary spaces."""
function _initialize_subspace_iteration_entry(
        network::InfiniteSquareNetwork, env::CTMRGEnv, coordinate::NTuple{3, Int},
        alg::SubspaceIterationProjector,
    )
    coordinates, trunc = _enlarged_corner_layout(coordinate, env, alg, Val(4))
    left_corner = EnlargedCorner(network, env, first(coordinates))
    right_corner = EnlargedCorner(network, env, last(coordinates))
    leftspace = codomain(left_corner)
    rightspace = domain(right_corner)
    subspace = infimum(alg.subspace, fuse(rightspace), fuse(leftspace))
    infimum(trunc.space, subspace) == trunc.space ||
        throw(ArgumentError("SI subspace must contain the target truncation space."))

    T = scalartype(network)
    X, Y = rangefinder_initialize(T, leftspace, rightspace, subspace, alg)
    U = randn(T, leftspace ← trunc.space)
    V = randn(T, trunc.space ← rightspace)
    U, _ = left_orth!(U, alg.orth_alg)
    _, V = right_orth!(V, alg.orth_alg)
    return (; X, Y, U, V)
end

"""Initialize a complete random SI cache for all directions and unit-cell coordinates."""
function initialize_subspace_iteration_cache(
        network::InfiniteSquareNetwork, env::CTMRGEnv, alg::SubspaceIterationProjector,
    )
    entries = map(eachcoordinate(env, 1:4)) do coordinate
        return _initialize_subspace_iteration_entry(network, env, coordinate, alg)
    end
    return SubspaceIterationCache(
        0, false, map(x -> x.X, entries), map(x -> x.Y, entries),
        map(x -> x.U, entries), map(x -> x.V, entries),
    )
end

"""Update the SI rangefinders `X`, `Y`."""
function rangefinder_update(
        X::AbstractTensorMap, Y::AbstractTensorMap,
        halfinf_left::AbstractTensorMap, halfinf_right::AbstractTensorMap,
        alg::SubspaceIterationProjector,
    )
    MX = halfinf_left * (halfinf_right * X)
    X′ = halfinf_right' * (halfinf_left' * MX)
    YM = (Y * halfinf_left) * halfinf_right
    Y′ = (YM * halfinf_right') * halfinf_left'
    X′, _ = left_orth!(X′, alg.orth_alg)
    _, Y′ = right_orth!(Y′, alg.orth_alg)
    return X′, Y′
end

"""Compute lifted SI singular vectors and recycled TensorMap rangefinders from two half environments."""
function _subspace_iteration_decomposition(
        halfinf_left::AbstractTensorMap, halfinf_right::AbstractTensorMap,
        alg::SubspaceIterationProjector, rangefinders,
    )
    trunc = rangefinders.trunc
    subspace = infimum(
        alg.subspace,
        fuse(domain(halfinf_right)),
        fuse(codomain(halfinf_left)),
    )
    infimum(trunc.space, subspace) == trunc.space ||
        throw(ArgumentError("SI subspace must contain the target truncation space."))
    compatible = _compatible_rangefinders(
        rangefinders.X, rangefinders.Y,
        halfinf_left, halfinf_right, subspace,
    )
    X, Y = if compatible
        rangefinders.X, rangefinders.Y
    else
        rangefinder_initialize(halfinf_left, halfinf_right, subspace, alg)
    end

    recycled = compatible && rangefinders.recycle
    if !recycled
        X, Y = rangefinder_update(X, Y, halfinf_left, halfinf_right, alg)
    end
    rho = (Y * halfinf_left) * (halfinf_right * X)
    svd_alg = decomposition_algorithm(alg)
    U, S, V = svd_compact!(rho, svd_alg.fwd_alg)
    (Uχ, Sχ, Vχ), retained = truncate(svd_trunc!, (U, S, V), trunc)

    Uχ = Y' * Uχ
    Vχ = Vχ * X'
    X = X * V'
    Y = U' * Y
    ϵ = truncation_error(diagview(S), retained) / norm(Sχ)

    return (; U = Uχ, S = Sχ, V = Vχ, truncation_error = ϵ, recycled, X, Y)
end

"""Construct SI-CTMRG projectors and return their decomposition and recycled bases."""
function compute_projector(enlarged_corners, alg::SubspaceIterationProjector, rangefinders)
    Zygote.isderiving() &&
        throw(ArgumentError("SubspaceIterationProjector does not currently support automatic differentiation."))
    halfinf_left = half_infinite_environment(enlarged_corners[1], enlarged_corners[2])
    halfinf_right = half_infinite_environment(enlarged_corners[3], enlarged_corners[4])
    halfinf_left = halfinf_left / norm(halfinf_left)
    halfinf_right = halfinf_right / norm(halfinf_right)
    info = _subspace_iteration_decomposition(
        halfinf_left, halfinf_right, alg, rangefinders
    )

    P_left, P_right = contract_projectors(
        info.U, info.S, info.V, halfinf_left, halfinf_right
    )
    return (P_left, P_right), info
end

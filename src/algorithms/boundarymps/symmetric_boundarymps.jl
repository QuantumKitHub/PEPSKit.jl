#
# Boundary MPS contraction of networks with a single-site unit cell and full spatial symmetry
#

# MPS optimization algorithms which can drive a boundary MPS contraction; these are the
# workhorse of every boundary MPS family, and are selected separately from the family itself
#
# NOTE: `MPSKit.GradientGrassmann` is deliberately absent. Its statmech objective function is
# `-log(real(⟨ψ|O|ψ⟩))`, which is only meaningful for a Hermitian transfer matrix; for a
# generic network the expectation value is complex along the line search and its real part
# can turn negative, throwing a `DomainError` out of `log`. It can still be used by passing
# an instance as `mps_alg` for networks where the transfer matrix is known to be Hermitian.
const MPS_ALGORITHM_SYMBOLS = IdDict{Symbol, Type{<:MPSKit.Algorithm}}(
    :VUMPS => VUMPS, :VOMPS => VOMPS,
)

# add algorithm-specific keyword arguments to the MPS algorithm kwargs if needed
_pad_mps_kwargs(::Type, mps_kwargs) = mps_kwargs
function _pad_mps_kwargs(::Type{<:VUMPS}, mps_kwargs)
    # the network transfer matrix is not Hermitian, so neither is the effective eigenvalue
    # problem solved at every VUMPS iteration
    return (; alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian = false), mps_kwargs...)
end

"""
$(TYPEDEF)

Algorithm for contracting an infinite square network with a single-site unit cell which is
invariant under rotations and Hermitian reflections, using a uniform boundary MPS.

The actual contraction is carried out by an MPSKit MPS optimization algorithm which is
wrapped by this struct, and which acts on the row-to-row transfer matrix of the network.

## Fields

$(TYPEDFIELDS)

## Constructors

    SymmetricBoundaryMPS(; kwargs...)
    SymmetricBoundaryMPS(mps_alg::MPSKit.Algorithm)

Construct a symmetric boundary MPS algorithm either from an MPSKit MPS optimization
algorithm directly, or based on the following keyword arguments:

* `tol::Real=$(Defaults.ctmrg_tol)` : Convergence tolerance of the boundary MPS contraction.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)` : Maximal number of boundary MPS iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)` : Output information verbosity.
* `mps_alg::Union{Symbol,NamedTuple,MPSKit.Algorithm}=(; alg::Symbol=:$(Defaults.boundarymps_mps_alg))` : MPS optimization algorithm driving the contraction, where `alg` can be one of the following:
    - `:VUMPS` : Variational uniform MPS, see [`MPSKit.VUMPS`](@extref) for details.
    - `:VOMPS` : Variational optimization of the MPS through MPO-MPS overlap maximization, see [`MPSKit.VOMPS`](@extref) for details.

  A bare `Symbol` is shorthand for `(; alg = symbol)`; any further entries of the `NamedTuple`
  are passed on to the MPS algorithm constructor and override `tol`, `maxiter` and
  `verbosity`. Supplying an `MPSKit.Algorithm` instance uses it as is, in which case
  `tol`, `maxiter` and `verbosity` are ignored.

!!! note
    The row-to-row transfer matrix of a network is generally not Hermitian, which restricts
    which MPS optimization algorithms are applicable. The eigensolver used by
    [`MPSKit.VUMPS`](@extref) must be able to handle non-Hermitian effective operators, so
    the default constructed here sets `ishermitian = false` accordingly. For the same reason
    [`MPSKit.GradientGrassmann`](@extref) is not offered as a `mps_alg` symbol: its
    objective function assumes a Hermitian transfer matrix and errors otherwise.
"""
struct SymmetricBoundaryMPS{A} <: BoundaryAlgorithm
    "wrapped MPSKit MPS optimization algorithm"
    alg::A
end
BOUNDARY_ALGORITHM_SYMBOLS[:SymmetricBoundaryMPS] = SymmetricBoundaryMPS

function SymmetricBoundaryMPS(;
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        verbosity = Defaults.ctmrg_verbosity,
        mps_alg = (;),
    )
    return SymmetricBoundaryMPS(_mps_algorithm(mps_alg; tol, maxiter, verbosity))
end

"""
    _mps_algorithm(mps_alg; tol, maxiter, verbosity)

Parse the `mps_alg` keyword argument of a boundary MPS algorithm into an MPSKit MPS
optimization algorithm. Accepts a `Symbol`, a `NamedTuple` carrying an `alg` symbol along
with further algorithm keyword arguments, or an `MPSKit.Algorithm` instance which is
returned unchanged.
"""
_mps_algorithm(mps_alg::MPSKit.Algorithm; kwargs...) = mps_alg
_mps_algorithm(mps_alg::Symbol; kwargs...) = _mps_algorithm((; alg = mps_alg); kwargs...)
function _mps_algorithm(mps_alg::NamedTuple; tol, maxiter, verbosity)
    mps_kwargs = (; alg = Defaults.boundarymps_mps_alg, tol, maxiter, verbosity, mps_alg...)

    haskey(MPS_ALGORITHM_SYMBOLS, mps_kwargs.alg) ||
        throw(ArgumentError("unknown MPS optimization algorithm: $(mps_kwargs.alg)"))
    alg_type = MPS_ALGORITHM_SYMBOLS[mps_kwargs.alg]

    # pad kwargs based on algorithm requirements and remove the `alg` keyword argument
    mps_kwargs = _pad_mps_kwargs(alg_type, Base.structdiff(mps_kwargs, (; alg = nothing)))

    return alg_type(; mps_kwargs...)
end

_tol(alg::SymmetricBoundaryMPS) = alg.alg.tol
_tol(alg::SymmetricBoundaryMPS{<:MPSKit.GradientGrassmann}) = alg.alg.method.gradtol

# `SymmetricBoundaryMPS` has no top-level `tol` field (it lives on the wrapped algorithm), so
# the default `MPSKit.DynamicTols._updatetol` (which sets `alg.tol`) doesn't apply
_updatetol(alg::SymmetricBoundaryMPS, tol::Real) = @set alg.alg.tol = tol
function _updatetol(alg::SymmetricBoundaryMPS{<:MPSKit.GradientGrassmann}, tol::Real)
    return @set alg.alg.method.gradtol = tol
end

#
## contraction
#

"""
    diagonalize_center(AL, AR, C, GL, GR)

Gauge a symmetric boundary MPS environment such that its bond tensor is diagonal, by
absorbing the singular vectors of `C` into the neighboring tensors.
"""
function diagonalize_center(
        AL::TE, AR::TE, C::TC, GL::TE, GR::TE
    ) where {TE <: EdgeTensor, TC <: CornerTensor}
    U, C´, V = svd_compact(C)
    AL´ = absorb_left_right(AL, U', U)
    AR´ = absorb_left_right(AR, V, V')
    GL´ = absorb_left_right(GL, U', U)
    GR´ = absorb_left_right(GR, V, V')
    return AL´, AR´, C´, GL´, GR´
end

"""
    leading_boundary(env₀::SymmetricBoundaryMPSEnv, network; kwargs...) -> env, info
    # expert version:
    leading_boundary(env₀::SymmetricBoundaryMPSEnv, network, alg::SymmetricBoundaryMPS)

Contract a single-site `network` which is invariant under rotations and Hermitian
reflections using a uniform boundary MPS, and return the resulting environment.

## Return values

* `env` : The final environment.
* `info` : A `NamedTuple` containing information about the contraction, with fields
  `converged`, `convergence_error`, `contraction_metrics` and the network value `N`.
"""
function MPSKit.leading_boundary(
        env₀::SymmetricBoundaryMPSEnv, network::InfiniteSquareNetwork,
        alg::SymmetricBoundaryMPS,
    )
    size(network) == (1, 1) ||
        throw(ArgumentError("symmetric boundary MPS contraction requires a single-site unit cell"))

    # convert to MPSKit language
    mps₀ = InfiniteMPS([env₀.AL], env₀.C)
    envs₀ = MPSKit.InfiniteEnvironments(
        PeriodicVector([env₀.GL]), PeriodicVector([env₀.GR])
    )
    O = InfiniteMPO(unitcell(network)[1, :])

    # run the actual boundary MPS algorithm
    mps, envs, ϵ = MPSKit.leading_boundary(mps₀, O, alg.alg, envs₀)

    # unpack
    AL, AR, C = only(mps.AL), only(mps.AR), only(mps.C)
    GL, GR = only(envs.GLs), only(envs.GRs)

    # diagonalize the bond tensor (optional, but assumed in the current implementation of the characteristic equations)
    AL, AR, C, GL, GR = diagonalize_center(AL, AR, C, GL, GR)

    # HACK: keep pretending the bond tensor is a dense and (possibly) complex tensor, such
    # that backpropagation through subsequent observable evaluations yields dense and
    # complex bond tensor cotangents, which implicit differentiation requires here
    C = TensorMap(C)
    if real(scalartype(network)) != scalartype(network)
        C = complex(C)
    end

    env = SymmetricBoundaryMPSEnv(AL, AR, C, GL, GR)
    N = network_value(network, env)

    info = (;
        converged = ϵ < _tol(alg),
        convergence_error = ϵ,
        contraction_metrics = (;),
        N,
    )

    return env, info
end
function MPSKit.leading_boundary(
        env₀::SymmetricBoundaryMPSEnv, network::InfiniteSquareNetwork; kwargs...
    )
    return MPSKit.leading_boundary(
        env₀, network, select_algorithm(leading_boundary, env₀; kwargs...)
    )
end
function MPSKit.leading_boundary(env₀::SymmetricBoundaryMPSEnv, state, args...; kwargs...)
    return MPSKit.leading_boundary(
        env₀, InfiniteSquareNetwork(state), args...; kwargs...
    )
end

#
# Boundary MPS contraction of infinite square networks
#

"""
$(TYPEDEF)

Abstract super type for boundary MPS algorithms for contracting infinite square networks.
"""
abstract type BoundaryMPSAlgorithm <: BoundaryAlgorithm end

const BOUNDARY_MPS_SYMBOLS = IdDict{Symbol, Type{<:BoundaryMPSAlgorithm}}()

# MPS optimization algorithms which can drive a boundary MPS contraction
const MPS_ALGORITHM_SYMBOLS = IdDict{Symbol, Type{<:MPSKit.Algorithm}}(
    :VUMPS => VUMPS, :VOMPS => VOMPS, :GradientGrassmann => GradientGrassmann,
)

# add algorithm-specific keyword arguments to the MPS algorithm kwargs if needed
_pad_mps_kwargs(::Type, mps_kwargs) = mps_kwargs
function _pad_mps_kwargs(::Type{<:VUMPS}, mps_kwargs)
    # the network transfer matrix is not Hermitian, so neither is the effective eigenvalue
    # problem solved at every VUMPS iteration
    return (; alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian = false), mps_kwargs...)
end

"""
    BoundaryMPSAlgorithm(; kwargs...)

Keyword argument parser returning the appropriate `BoundaryMPSAlgorithm` algorithm struct.
"""
function BoundaryMPSAlgorithm(;
        alg = Defaults.boundarymps_alg,
        tol = Defaults.ctmrg_tol,
        maxiter = Defaults.ctmrg_maxiter,
        verbosity = Defaults.ctmrg_verbosity,
        mps_alg = Defaults.boundarymps_mps_alg,
    )
    # replace symbols with algorithm types
    haskey(BOUNDARY_MPS_SYMBOLS, alg) ||
        throw(ArgumentError("unknown boundary MPS algorithm: $alg"))
    alg_type = BOUNDARY_MPS_SYMBOLS[alg]
    haskey(MPS_ALGORITHM_SYMBOLS, mps_alg) ||
        throw(ArgumentError("unknown MPS optimization algorithm: $mps_alg"))
    mps_alg_type = MPS_ALGORITHM_SYMBOLS[mps_alg]

    # parse MPS optimization algorithm
    mps_kwargs = _pad_mps_kwargs(mps_alg_type, (; tol, maxiter, verbosity))
    mps_algorithm = mps_alg_type(; mps_kwargs...)

    return _boundary_mps_algorithm(alg_type, mps_algorithm, tol)
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

    C4vBoundaryMPS(; kwargs...)
    C4vBoundaryMPS(mps_alg::MPSKit.Algorithm)

Construct a C4v boundary MPS algorithm either from an MPSKit MPS optimization
algorithm directly, or based on the following keyword arguments:

* `tol::Real=$(Defaults.ctmrg_tol)` : Convergence tolerance of the boundary MPS contraction.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)` : Maximal number of boundary MPS iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)` : Output information verbosity.
* `mps_alg::Symbol=:$(Defaults.boundarymps_mps_alg)` : MPS optimization algorithm driving the contraction, which can be one of the following:
    - `:VUMPS` : Variational uniform MPS, see [`MPSKit.VUMPS`](@extref) for details.
    - `:VOMPS` : Variational optimization of the MPS through MPO-MPS overlap maximization, see [`MPSKit.VOMPS`](@extref) for details.
    - `:GradientGrassmann` : Gradient optimization of the MPS on the Grassmann manifold, see [`MPSKit.GradientGrassmann`](@extref) for details.

!!! note
    The row-to-row transfer matrix of a network is generally not Hermitian, which restricts
    which MPS optimization algorithms are applicable. The eigensolver used by
    [`MPSKit.VUMPS`](@extref) must be able to handle non-Hermitian effective operators, so
    the default constructed here sets `ishermitian = false` accordingly.
"""
struct C4vBoundaryMPS{A} <: BoundaryMPSAlgorithm
    "wrapped MPSKit MPS optimization algorithm"
    alg::A
end
function C4vBoundaryMPS(; kwargs...)
    return BoundaryMPSAlgorithm(; alg = :C4vBoundaryMPS, kwargs...)
end
BOUNDARY_MPS_SYMBOLS[:C4vBoundaryMPS] = C4vBoundaryMPS

_boundary_mps_algorithm(::Type{<:C4vBoundaryMPS}, mps_alg, tol) = C4vBoundaryMPS(mps_alg)

"""
$(TYPEDEF)

Algorithm for contracting an infinite square network with an arbitrary unit cell using a
top and a bottom boundary MPS.

The top boundary MPS is the leading right eigenvector of the row-to-row transfer matrix of
the network, and the bottom boundary MPS is computed as the top boundary MPS of the network
rotated by 180 degrees. Both are found by an MPSKit MPS optimization algorithm acting on the
multiline transfer matrix of the unit cell. The mixed environments between the two are then
obtained per row as the leading eigenvectors of the mixed transfer matrix through the whole
row.

## Fields

$(TYPEDFIELDS)

## Constructors

    BoundaryMPS(; kwargs...)
    BoundaryMPS(mps_alg::MPSKit.Algorithm, eigsolve_alg)

Construct a boundary MPS algorithm either from an MPSKit MPS optimization algorithm and an
eigensolver directly, or based on the following keyword arguments:

* `tol::Real=$(Defaults.ctmrg_tol)` : Convergence tolerance of the boundary MPS contraction. The mixed environments are computed to a tolerance of `1e-2 * tol`.
* `maxiter::Int=$(Defaults.ctmrg_maxiter)` : Maximal number of boundary MPS iterations.
* `verbosity::Int=$(Defaults.ctmrg_verbosity)` : Output information verbosity.
* `mps_alg::Symbol=:$(Defaults.boundarymps_mps_alg)` : MPS optimization algorithm driving the contraction, `:VUMPS` or `:VOMPS` as for [`C4vBoundaryMPS`](@ref).
"""
struct BoundaryMPS{A, E} <: BoundaryMPSAlgorithm
    "wrapped MPSKit MPS optimization algorithm"
    alg::A
    "eigensolver for the mixed environments"
    eigsolve_alg::E
end
function BoundaryMPS(; kwargs...)
    return BoundaryMPSAlgorithm(; alg = :BoundaryMPS, kwargs...)
end
BOUNDARY_MPS_SYMBOLS[:BoundaryMPS] = BoundaryMPS

function _boundary_mps_algorithm(::Type{<:BoundaryMPS}, mps_alg, tol)
    mps_alg isa GradientGrassmann && throw(
        ArgumentError("`GradientGrassmann` requires a Hermitian transfer matrix, use `C4vBoundaryMPS`")
    )
    return BoundaryMPS(mps_alg, _mixed_eigsolve_alg(tol))
end

_mixed_eigsolve_alg(tol) = Arnoldi(; tol = 1.0e-2 * tol, eager = true, verbosity = 0)

_tol(alg::BoundaryMPSAlgorithm) = alg.alg.tol

# boundary MPS algorithms have no top-level `tol` field (it lives on the wrapped algorithm),
# so the default `MPSKit.DynamicTols._updatetol` (which sets `alg.tol`) doesn't apply
_updatetol(alg::C4vBoundaryMPS, tol::Real) = @set alg.alg.tol = tol
function _updatetol(alg::BoundaryMPS, tol::Real)
    alg = @set alg.alg.tol = tol
    return @set alg.eigsolve_alg.tol = 1.0e-2 * tol
end

# the transfer matrix of a C4v-symmetric network is Hermitian, so it can also be contracted with
# `GradientGrassmann`, which keeps its tolerance on its optimizer and only has a keyword
# constructor
_tol(alg::C4vBoundaryMPS{<:GradientGrassmann}) = alg.alg.method.gradtol
function _updatetol(alg::C4vBoundaryMPS{<:GradientGrassmann}, tol::Real)
    gg = alg.alg
    method = gg.method
    method = @set method.gradtol = tol
    # a default convergence criterion is set up from the tolerance, so rebuild it
    hasconverged = gg.hasconverged isa OptimKit.DefaultHasConverged ? nothing : gg.hasconverged
    return C4vBoundaryMPS(
        GradientGrassmann(;
            method, (finalize!) = gg.finalize!,
            backend = gg.backend, hasconverged, shouldstop = gg.shouldstop,
        )
    )
end

#
## contraction
#

# The network rows as the rows of an MPSKit multiline transfer matrix
function _multiline_mpo(network::InfiniteSquareNetwork)
    return MultilineMPO(map(r -> InfiniteMPO(unitcell(network)[r, :]), axes(network, 1)))
end

"""
    diagonalize_center(AL, AR, C, GL, GR)

Gauge a boundary MPS, given as matrices of tensors in the positions of a
[`BoundaryMPSEnv`](@ref), together with its environments such that every bond tensor is
diagonal, by absorbing the singular vectors of the bond tensors into the neighboring tensors.
"""
function diagonalize_center(
        AL::Matrix{TE}, AR::Matrix{TE}, C::Matrix{TC}, GL::Matrix{TG}, GR::Matrix{TG}
    ) where {TE <: EdgeTensor, TC <: CornerTensor, TG <: EdgeTensor}
    USV = map(svd_compact, C)
    U, C´, V = map(first, USV), map(x -> x[2], USV), map(last, USV)
    AL´ = _cellmap((r, c) -> absorb_left_right(AL[r, c], _at(U, r, c - 1)', U[r, c]), size(AL))
    AR´ = _cellmap((r, c) -> absorb_left_right(AR[r, c], _at(V, r, c - 1), V[r, c]'), size(AR))
    # GL[r, c] lies on the bond to the right of column c and GR[r, c] on the bond to its left,
    # both between the MPS rows r - 1 (above, last leg) and r (below, first leg)
    GL´ = _cellmap((r, c) -> absorb_left_right(GL[r, c], U[r, c]', _at(U, r - 1, c)), size(GL))
    GR´ = _cellmap(size(GR)) do r, c
        return absorb_left_right(GR[r, c], _at(V, r - 1, c - 1), _at(V, r, c - 1)')
    end
    return AL´, AR´, C´, GL´, GR´
end

# One boundary MPS on top of `network` and its environments, from the initial guess
# `(AL, AR, C, GL, GR)`, with every bond tensor diagonalized. Returns the boundary MPS and
# environments in the same form, and the convergence error.
function _boundary_mps_fixedpoint((AL₀, _, C₀, GL₀, GR₀), network, mps_alg)
    mps, envs, info = leading_boundary(
        _multiline_mps(AL₀, C₀), _multiline_mpo(network), mps_alg,
        _multiline_environments(GL₀, GR₀),
    )
    AL, AR, C = _mps_tensors(mps)
    GL, GR = _mps_environments(envs)

    # diagonalize the bond tensors, which the characteristic equations assume
    AL, AR, C, GL, GR = diagonalize_center(AL, AR, C, GL, GR)

    # HACK: keep pretending the bond tensors are dense and (possibly) complex tensors, such
    # that backpropagation through subsequent observable evaluations yields dense and
    # complex bond tensor cotangents, which implicit differentiation requires here
    C = map(C) do c
        c = TensorMap(c)
        return scalartype(network) <: Complex ? complex(c) : c
    end

    return (AL, AR, C, GL, GR), MPSKit.convergence_measure(info)
end

# Fixed points along a row of `n` sites. `t(x, i)` maps the vector to the left of site `i`
# to the one to its right (`_left_fixedpoints`) or the one to its right to the one to its
# left (`_right_fixedpoints`), and the leading eigenvector of the product of all maps around
# the row determines all others. Products and orbits are built by recursion on a tuple, since
# Zygote's adjoint of `foldl` silently drops the gradient with respect to the captures of the
# folded function.
_through(t, x, ::Tuple{}) = x
_through(t, x, is::Tuple) = _through(t, t(x, first(is)), Base.tail(is))
_orbit(t, x, ::Tuple{}) = (x,)
_orbit(t, x, is::Tuple) = (x, _orbit(t, t(x, first(is)), Base.tail(is))...)

function _left_fixedpoints(t, x₀, n, alg)
    _, xs, = eigsolve(x -> _through(t, x, Tuple(1:n)), x₀, 1, :LM, alg)
    orbit = _orbit(t, first(xs), Tuple(1:(n - 1))) # to the right of sites n, 1, …, n - 1
    return [orbit[mod1(i + 1, n)] for i in 1:n]
end
function _right_fixedpoints(t, x₀, n, alg)
    _, xs, = eigsolve(x -> _through(t, x, Tuple(n:-1:1)), x₀, 1, :LM, alg)
    orbit = _orbit(t, first(xs), Tuple(n:-1:2)) # to the left of sites 1, n, n - 1, …, 2
    return [orbit[i == 1 ? 1 : n - i + 2] for i in 1:n]
end

"""
    mixed_environments(ALt, ARt, Ct, ALb, ARb, Cb, network, alg, GL₀, GR₀)

Compute the mixed environments between a top boundary MPS `(ALt, ARt, Ct)` and a bottom
boundary MPS `(ALb, ARb, Cb)` of `network`, in the positions of a [`BoundaryMPSEnv`](@ref),
using the eigensolver `alg` and the initial guesses `GL₀` and `GR₀`. The right environments
are normalized per site, and the left ones such that the contraction of every bond with the
bond tensors of both boundary MPS and the environments on either side equals one. With this
normalization, the eigenvalues of the left and right mixed transfer matrices through every
site coincide.
"""
function mixed_environments(ALt, ARt, Ct, ALb, ARb, Cb, network, alg, GL₀, GR₀)
    rows, cols = size(network)
    envs = map(1:rows) do r
        left_map(x, c) = edge_transfer_left(x, network[r, c], _at(ALt, r - 1, c), _at(ARb, r + 1, c))
        right_map(x, c) = edge_transfer_right(x, network[r, c], _at(ARt, r - 1, c), _at(ALb, r + 1, c))
        GLr = _left_fixedpoints(left_map, GL₀[r, cols], cols, alg)
        GRr = normalize.(_right_fixedpoints(right_map, GR₀[r, 1], cols, alg))
        GLr = map(1:cols) do c
            return GLr[c] / _mixed_bond_value(
                _at(Cb, r + 1, c), _at(Ct, r - 1, c), GLr[c], GRr[mod1(c + 1, cols)]
            )
        end
        return GLr, GRr
    end
    GL = _cellmap((r, c) -> envs[r][1][c], (rows, cols))
    GR = _cellmap((r, c) -> envs[r][2][c], (rows, cols))
    return GL, GR
end

# the closed contraction of a bond with the bond tensors of the top and bottom MPS and the
# mixed environments on either side of it
_mixed_bond_value(Cb, Ct, GL, GR) = tr(Cb * ∂C(Ct, GL, GR))

"""
    leading_boundary(env₀::C4vBoundaryMPSEnv, network; kwargs...) -> env, info
    leading_boundary(env₀::BoundaryMPSEnv, network; kwargs...) -> env, info
    # expert version:
    leading_boundary(env₀::C4vBoundaryMPSEnv, network, alg::C4vBoundaryMPS)
    leading_boundary(env₀::BoundaryMPSEnv, network, alg::BoundaryMPS)

Contract `network` using boundary MPS, and return the resulting environment. A
[`C4vBoundaryMPSEnv`](@ref) contracts a single-site network which is invariant under
rotations and Hermitian reflections with a single boundary MPS, while a
[`BoundaryMPSEnv`](@ref) contracts a network with any unit cell with a top and a bottom
boundary MPS.

## Return values

* `env` : The final environment.
* `info` : A `NamedTuple` containing information about the contraction, with fields
  `converged`, `convergence_error`, `contraction_metrics` and the network value `N`.
"""
function MPSKit.leading_boundary(
        env₀::C4vBoundaryMPSEnv, network::InfiniteSquareNetwork,
        alg::C4vBoundaryMPS,
    )
    size(network) == (1, 1) ||
        throw(ArgumentError("C4v boundary MPS contraction requires a single-site unit cell"))

    # a single-site boundary MPS is the 1×1 case of the multi-site one
    (AL, AR, C, GL, GR), ϵ = _boundary_mps_fixedpoint(
        map(x -> fill(x, 1, 1), _unpack(env₀)), network, alg.alg
    )
    env = C4vBoundaryMPSEnv(only(AL), only(AR), only(C), only(GL), only(GR))

    info = (;
        converged = ϵ < _tol(alg),
        convergence_error = ϵ,
        contraction_metrics = (;),
        N = network_value(network, env),
    )
    return env, info
end
function MPSKit.leading_boundary(
        env₀::BoundaryMPSEnv, network::InfiniteSquareNetwork, alg::BoundaryMPS
    )
    size(env₀) == size(network) ||
        throw(ArgumentError("the environment and the network have different unit cells"))

    # the top boundary MPS, and the bottom one as the top boundary MPS of the rotated network
    (ALt, ARt, Ct, GLt, GRt), ϵt = _boundary_mps_fixedpoint(_top(env₀), network, alg.alg)
    bottom, ϵb = _boundary_mps_fixedpoint(_bottom(env₀), rot180(network), alg.alg)
    ALb, ARb, Cb, GLb, GRb = rotate_bottom(bottom)

    GLm, GRm = mixed_environments(
        ALt, ARt, Ct, ALb, ARb, Cb, network, alg.eigsolve_alg, env₀.GLm, env₀.GRm
    )
    env = BoundaryMPSEnv(ALt, ARt, Ct, GLt, GRt, ALb, ARb, Cb, GLb, GRb, GLm, GRm)

    ϵ = max(ϵt, ϵb)
    info = (;
        converged = ϵ < _tol(alg),
        convergence_error = ϵ,
        contraction_metrics = (;),
        N = network_value(network, env),
    )
    return env, info
end
function MPSKit.leading_boundary(env₀::BoundaryMPSEnvs, network::InfiniteSquareNetwork; kwargs...)
    return MPSKit.leading_boundary(
        env₀, network, select_algorithm(leading_boundary, env₀; kwargs...)
    )
end
function MPSKit.leading_boundary(env₀::BoundaryMPSEnvs, state, args...; kwargs...)
    return MPSKit.leading_boundary(
        env₀, InfiniteSquareNetwork(state), args...; kwargs...
    )
end

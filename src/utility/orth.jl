"""
$(TYPEDEF)

Wrapper for an orthogonalization algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
Use the same instance with `left_orth(!)` for QR decomposition and `right_orth(!)` for LQ decomposition.

## Fields

$(TYPEDFIELDS)

## Constructors

    OrthAdjoint(; kwargs...)

Construct an `OrthAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.orth_fwd_alg))`: Orthogonalization algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where the algorithm is specified by the `alg` keyword.
  The available algorithms are provided through MatrixAlgebraKit and include:
    - `:DefaultAlgorithm` : MatrixAlgebraKit's [default orthogonalization algorithm](@extref MatrixAlgebraKit.DefaultAlgorithm) for a given matrix type.
    - `:Householder` : MatrixAlgebraKit's [`Householder`](@extref MatrixAlgebraKit.Householder)
* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.orth_rrule_alg))`: Reverse-rule algorithm for differentiating the QR or LQ decomposition.
  Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:FullPullback` : MatrixAlgebraKit's [`qr_pullback!`](@extref MatrixAlgebraKit.qr_pullback!) or [`lq_pullback!`](@extref MatrixAlgebraKit.lq_pullback!), selected by the orthogonalization direction.

!!! note
    Manually specifying a `rrule_alg` is considered expert-mode usage, and should only be done when full control over the implementation is desired.
    For all regular use cases, the default reverse rule algorithms, automatically chosen based on the forward algorithm, should be sufficient.
"""
struct OrthAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end

"""Compatibility alias for [`OrthAdjoint`](@ref), supporting both QR and LQ decompositions."""
const QRAdjoint = OrthAdjoint

const ORTH_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :DefaultAlgorithm => DefaultAlgorithm,
    :Householder => Householder,
)
const ORTH_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :FullPullback => FullPullback
)

function OrthAdjoint(; fwd_alg = (;), rrule_alg = (;))
    # parse forward algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (;
            alg = Defaults.orth_fwd_alg,
            positive = Defaults.orth_fwd_positive,
            fwd_alg...,
        ) # overwrite with specified kwargs
        haskey(ORTH_FWD_SYMBOLS, fwd_kwargs.alg) ||
            throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
        fwd_type = ORTH_FWD_SYMBOLS[fwd_kwargs.alg]
        fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg = nothing)) # remove `alg` keyword argument
        fwd_type(; fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg = Defaults.orth_rrule_alg,
            verbosity = Defaults.orth_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(ORTH_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = ORTH_RRULE_SYMBOLS[rrule_kwargs.alg]
        rrule_type()
    else
        rrule_alg
    end

    return OrthAdjoint(fwd_algorithm, rrule_algorithm)
end

"""
    left_orth(t, alg::OrthAdjoint)
    left_orth!(t, alg::OrthAdjoint)

Wrapper around `left_orth(!)` which dispatches on the `OrthAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
"""
MatrixAlgebraKit.left_orth(t, alg::OrthAdjoint) = left_orth!(copy(t), alg)
MatrixAlgebraKit.left_orth!(t, alg::OrthAdjoint) = left_orth!(t, alg.fwd_alg)

# left_orth! rrule wrapping MatrixAlgebraKit's qr_pullback!
# https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/blob/b76c7bb60014ecfead6925d0df6cb4b8d7c2668a/src/pullbacks/qr.jl#L49
function ChainRulesCore.rrule(
        ::typeof(left_orth!),
        t::AbstractTensorMap,
        alg::OrthAdjoint{F, R},
    ) where {F <: MatrixAlgebraKit.Algorithm, R <: FullPullback}
    QR = left_orth(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    """Propagate QR cotangents with the configured gauge-sensitivity tolerance."""
    function left_orth!_pullback(ΔQR)
        Δt = zeros(storagetype(t), space(t))
        MatrixAlgebraKit.qr_pullback!(Δt, t, QR, unthunk.(ΔQR); gauge_atol = gtol(ΔQR))
        return NoTangent(), Δt, NoTangent()
    end
    function left_orth!_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return QR, left_orth!_pullback
end

"""Compute an LQ decomposition using the forward algorithm and reverse rule configured by `OrthAdjoint`."""
MatrixAlgebraKit.right_orth(t, alg::OrthAdjoint) = right_orth!(copy(t), alg)
"""Factorize the input in place using the LQ forward algorithm configured by `OrthAdjoint`."""
MatrixAlgebraKit.right_orth!(t, alg::OrthAdjoint) = right_orth!(t, alg.fwd_alg)

# right_orth! rrule wrapping MatrixAlgebraKit's lq_pullback!
function ChainRulesCore.rrule(
        ::typeof(right_orth!),
        t::AbstractTensorMap,
        alg::OrthAdjoint{F, R},
    ) where {F <: MatrixAlgebraKit.Algorithm, R <: FullPullback}
    LQ = right_orth(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    """Propagate LQ cotangents with the configured gauge-sensitivity tolerance."""
    function right_orth!_pullback(ΔLQ)
        Δt = zeros(storagetype(t), space(t))
        MatrixAlgebraKit.lq_pullback!(Δt, t, LQ, unthunk.(ΔLQ); gauge_atol = gtol(ΔLQ))
        return NoTangent(), Δt, NoTangent()
    end
    function right_orth!_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return LQ, right_orth!_pullback
end

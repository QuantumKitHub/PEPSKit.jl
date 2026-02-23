"""
$(TYPEDEF)

QR reverse-rule algorithm which wraps MatrixAlgebraKit's `qr_pullback!`.
"""
@kwdef struct QRPullback
    verbosity::Int = 0
end

"""
$(TYPEDEF)

Wrapper for a QR decomposition algorithm `fwd_alg` with a defined reverse rule `rrule_alg`.
If `isnothing(rrule_alg)`, Zygote differentiates the forward call automatically.

## Fields

$(TYPEDFIELDS)

## Constructors

    QRAdjoint(; kwargs...)

Construct a `QRAdjoint` algorithm struct based on the following keyword arguments:

* `fwd_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.qr_fwd_alg))`: Eig algorithm of the forward pass which can either be passed as an `Algorithm` instance or a `NamedTuple` where `alg` is one of the following:
    - `:qr` : MatrixAlgebraKit's `LAPACK_HouseholderQR`

* `rrule_alg::Union{Algorithm,NamedTuple}=(; alg::Symbol=$(Defaults.qr_rrule_alg))`: Reverse-rule algorithm for differentiating the eigenvalue decomposition. Can be supplied by an `Algorithm` instance directly or as a `NamedTuple` where `alg` is one of the following:
    - `:qr` : MatrixAlgebraKit's `qr_pullback`
"""
struct QRAdjoint{F, R}
    fwd_alg::F
    rrule_alg::R
end  # Keep truncation algorithm separate to be able to specify CTMRG dependent information

const QR_FWD_SYMBOLS = IdDict{Symbol, Any}(
    :qr => LAPACK_HouseholderQR
)
const QR_RRULE_SYMBOLS = IdDict{Symbol, Type{<:Any}}(
    :qr => QRPullback
)

function QRAdjoint(; fwd_alg = (;), rrule_alg = (;))
    # parse forward algorithm
    fwd_algorithm = if fwd_alg isa NamedTuple
        fwd_kwargs = (;
            alg = Defaults.qr_fwd_alg,
            positive = Defaults.qr_fwd_positive,
            fwd_alg...,
        ) # overwrite with specified kwargs
        haskey(QR_FWD_SYMBOLS, fwd_kwargs.alg) ||
            throw(ArgumentError("unknown forward algorithm: $(fwd_kwargs.alg)"))
        fwd_type = QR_FWD_SYMBOLS[fwd_kwargs.alg]
        fwd_kwargs = Base.structdiff(fwd_kwargs, (; alg = nothing)) # remove `alg` keyword argument
        fwd_type(; fwd_kwargs...)
    else
        fwd_alg
    end

    # parse reverse-rule algorithm
    rrule_algorithm = if rrule_alg isa NamedTuple
        rrule_kwargs = (;
            alg = Defaults.qr_rrule_alg,
            verbosity = Defaults.qr_rrule_verbosity,
            rrule_alg...,
        ) # overwrite with specified kwargs

        haskey(QR_RRULE_SYMBOLS, rrule_kwargs.alg) ||
            throw(ArgumentError("unknown rrule algorithm: $(rrule_kwargs.alg)"))
        rrule_type = QR_RRULE_SYMBOLS[rrule_kwargs.alg]
        rrule_type()
    else
        rrule_alg
    end

    return QRAdjoint(fwd_algorithm, rrule_algorithm)
end

"""
    left_orth(t, alg::QRAdjoint)
    left_orth!(t, alg::QRAdjoint)

Wrapper around `left_orth(!)` which dispatches on the `QRAdjoint` algorithm.
This is needed since a custom adjoint may be defined, depending on the `alg`.
"""
MatrixAlgebraKit.left_orth(t, alg::QRAdjoint) = left_orth!(copy(t), alg)
MatrixAlgebraKit.left_orth!(t, alg::QRAdjoint) = _left_orth!(t, alg.fwd_alg)
_left_orth!(t, alg::LAPACK_HouseholderQR) = left_orth!(t; alg)

"""
$(TYPEDEF)

QR decomposition struct containing a pre-computed decomposition. Th call to `left_orth(!)`
just returns the precomputed `Q` and `R`. In the reverse pass, the adjoint is computed with
these exact `D` and `R`.

## Fields

$(TYPEDFIELDS)
"""
struct FixedQR{Qt, Rt}
    Q::Qt
    R::Rt
end

_left_orth!(_, alg::FixedQR) = alg.Q, alg.R

# left_orth! rrule wrapping MatrixAlgebraKit's qr_pullback!
function ChainRulesCore.rrule(
        ::typeof(left_orth!),
        t::AbstractTensorMap,
        alg::QRAdjoint{F, R},
    ) where {F <: Union{LAPACK_HouseholderQR, FixedQR}, R <: QRPullback}
    QR = left_orth(t, alg)
    gtol = _get_pullback_gauge_tol(alg.rrule_alg.verbosity)

    function left_orth!_pullback(ΔQR)
        Δt = zeros(scalartype(t), space(t))
        MatrixAlgebraKit.qr_pullback!(Δt, t, QR, unthunk.(ΔQR); gauge_atol = gtol(ΔQR))
        return NoTangent(), Δt, NoTangent()
    end
    function left_orth!_pullback(::Tuple{ZeroTangent, ZeroTangent})
        return NoTangent(), ZeroTangent(), NoTangent()
    end

    return QR, left_orth!_pullback
end

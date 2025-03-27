"""
$(TYPEDEF)

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

const TRUNCATION_SCHEME_SYMBOLS = IdDict{Symbol,Type{<:TruncationScheme}}(
    :fixedspace => FixedSpaceTruncation,
    :notrunc => TensorKit.NoTruncation,
    :truncerr => TensorKit.TruncationError,
    :truncdim => TensorKit.TruncationDimension,
    :truncspace => TensorKit.TruncationSpace,
    :truncbelow => TensorKit.TruncationCutoff,
)

"""
    _TruncationScheme(; kwargs...)

Keyword argument parser returning the appropriate `TruncationScheme` algorithm struct.
"""
function _TruncationScheme(; alg=Defaults.trscheme, η=nothing)
    # replace Symbol with TruncationScheme type
    haskey(TRUNCATION_SCHEME_SYMBOLS, alg) ||
        throw(ArgumentError("unknown truncation scheme: $alg"))
    alg_type = TRUNCATION_SCHEME_SYMBOLS[alg]

    return isnothing(η) ? alg_type() : alg_type(η)
end

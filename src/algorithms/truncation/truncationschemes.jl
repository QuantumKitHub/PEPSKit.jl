"""
$(TYPEDEF)

SVD truncation strategy which preserves the `CTMRGEnv` environment virtual spaces,
or `InfinitePEPS`, `InfinitePEPO` virtual spaces.
"""
struct FixedSpaceTruncation <: TruncationStrategy end

"""
$(TYPEDEF)

SVD truncation strategy specified for each nearest neighbor bond 
in `InfinitePEPS`, `InfinitePEPO`:
- `trunc[1, r, c]` applies to the x-bond between `[r, c]` and `[r, c+1]`.
    If it is a `TruncationSpace`, the space refers to the east domain of
    `[r, c]` or its `flip`, whichever is non-dual.
- `trunc[2, r, c]` applies to the y-bond between `[r, c]` and `[r-1, c]`.
    If it is a `TruncationSpace`, the space refers to the north domain of
    `[r, c]` or its `flip`, whichever is non-dual.
"""
struct SiteDependentTruncation{T <: TruncationStrategy} <: TruncationStrategy
    truncs::Array{T, 3}

    function SiteDependentTruncation(truncs::Array{T, 3}) where {T}
        # TODO: generalize it to CTMRGEnv
        size(truncs, 1) != 2 && throw(
            DimensionMismatch(
                "The first dimension of `truncs` must have a size of 2. Got $(size(truncs, 1))."
            )
        )
        return new{T}(truncs)
    end
end

const TRUNCATION_STRATEGY_SYMBOLS = IdDict{Symbol, Type{<:TruncationStrategy}}(
    :notrunc => MatrixAlgebraKit.NoTruncation,
    :truncerror => MatrixAlgebraKit.TruncationByError,
    :truncrank => MatrixAlgebraKit.TruncationByOrder,
    :trunctol => MatrixAlgebraKit.TruncationByValue,
    :truncspace => TruncationSpace,
    :fixedspace => FixedSpaceTruncation,
    :sitedependent => SiteDependentTruncation,
)

# Should be TruncationStrategy but rename to avoid type piracy
function _TruncationStrategy(; alg = Defaults.trunc, η = nothing)
    # replace Symbol with TruncationStrategy type
    haskey(TRUNCATION_STRATEGY_SYMBOLS, alg) ||
        throw(ArgumentError("unknown truncation strategy: $alg"))
    alg_type = TRUNCATION_STRATEGY_SYMBOLS[alg]

    return isnothing(η) ? alg_type() : alg_type(η)
end

function truncation_strategy(
        trunc::TruncationStrategy, direction::Int, row::Int, col::Int; kwargs...
    )
    return trunc
end

function truncation_strategy(
        trunc::SiteDependentTruncation, direction::Int, row::Int, col::Int
    )
    return trunc.truncs[direction, row, col]
end

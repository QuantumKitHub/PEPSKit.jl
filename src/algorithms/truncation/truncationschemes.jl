"""
$(TYPEDEF)

CTMRG specific truncation strategy for `svd_trunc` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TruncationStrategy end

struct SiteDependentTruncation{T <: TruncationStrategy} <: TruncationStrategy
    truncs::Array{T, 3}
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

# TODO: type piracy
Base.rotl90(trunc::TruncationStrategy) = trunc

function Base.rotl90(trunc::SiteDependentTruncation)
    directions, rows, cols = size(trunc.truncs)
    truncs_rotated = similar(trunc.truncs, directions, cols, rows)

    if directions == 2
        truncs_rotated[NORTH, :, :] = circshift(
            rotl90(trunc.truncs[EAST, :, :]), (0, -1)
        )
        truncs_rotated[EAST, :, :] = rotl90(trunc.truncs[NORTH, :, :])
    elseif directions == 4
        for dir in 1:4
            dir′ = _prev(dir, 4)
            truncs_rotated[dir′, :, :] = rotl90(trunc.truncs[dir, :, :])
        end
    else
        throw(ArgumentError("Unsupported number of directions for rotl90: $directions"))
    end
    return SiteDependentTruncation(truncs_rotated)
end

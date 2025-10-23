"""
$(TYPEDEF)

CTMRG specific truncation strategy for `svd_trunc` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TruncationStrategy end

struct SiteDependentTruncation{T <: TruncationStrategy} <: TruncationStrategy
    trschemes::Array{T, 3}
end

const TRUNCATION_STRATEGY_SYMBOLS = IdDict{Symbol, Type{<:TruncationStrategy}}(
    :notrunc => MatrixAlgebraKit.NoTruncation,
    :truncerror => MatrixAlgebraKit.TruncationByError,
    :truncrank => MatrixAlgebraKit.TruncationByOrder,
    :trunctol => MatrixAlgebraKit.TruncationByValue,
    :truncspace => TensorKit.TruncationSpace,
    :fixedspace => FixedSpaceTruncation,
    :sitedependent => SiteDependentTruncation,
)

# Should be TruncationStrategy but rename to avoid type piracy
function _TruncationStrategy(; alg = Defaults.trscheme, η = nothing)
    # replace Symbol with TruncationStrategy type
    haskey(TRUNCATION_STRATEGY_SYMBOLS, alg) ||
        throw(ArgumentError("unknown truncation strategy: $alg"))
    alg_type = TRUNCATION_STRATEGY_SYMBOLS[alg]

    return isnothing(η) ? alg_type() : alg_type(η)
end

function truncation_strategy(
        trscheme::TruncationStrategy, direction::Int, row::Int, col::Int; kwargs...
    )
    return trscheme
end

function truncation_strategy(
        trscheme::SiteDependentTruncation, direction::Int, row::Int, col::Int
    )
    return trscheme.trschemes[direction, row, col]
end

# TODO: type piracy
Base.rotl90(trscheme::TruncationStrategy) = trscheme

function Base.rotl90(trscheme::SiteDependentTruncation)
    directions, rows, cols = size(trscheme.trschemes)
    trschemes_rotated = similar(trscheme.trschemes, directions, cols, rows)

    if directions == 2
        trschemes_rotated[NORTH, :, :] = circshift(
            rotl90(trscheme.trschemes[EAST, :, :]), (0, -1)
        )
        trschemes_rotated[EAST, :, :] = rotl90(trscheme.trschemes[NORTH, :, :])
    elseif directions == 4
        for dir in 1:4
            dir′ = _prev(dir, 4)
            trschemes_rotated[dir′, :, :] = rotl90(trscheme.trschemes[dir, :, :])
        end
    else
        throw(ArgumentError("Unsupported number of directions for rotl90: $directions"))
    end
    return SiteDependentTruncation(trschemes_rotated)
end

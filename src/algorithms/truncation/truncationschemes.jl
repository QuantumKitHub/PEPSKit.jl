"""
$(TYPEDEF)

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TruncationScheme end

struct SiteDependentTruncation{T<:TruncationScheme} <: TruncationScheme
    trschemes::Array{T,3}
end

const TRUNCATION_SCHEME_SYMBOLS = IdDict{Symbol,Type{<:TruncationScheme}}(
    :fixedspace => FixedSpaceTruncation,
    :notrunc => TensorKit.NoTruncation,
    :truncerr => TensorKit.TruncationError,
    :truncdim => TensorKit.TruncationDimension,
    :truncspace => TensorKit.TruncationSpace,
    :truncbelow => TensorKit.TruncationCutoff,
    :sitedependent => SiteDependentTruncation,
)

# Should be TruncationScheme but rename to avoid type piracy
function _TruncationScheme(; alg=Defaults.trscheme, η=nothing)
    # replace Symbol with TruncationScheme type
    haskey(TRUNCATION_SCHEME_SYMBOLS, alg) ||
        throw(ArgumentError("unknown truncation scheme: $alg"))
    alg_type = TRUNCATION_SCHEME_SYMBOLS[alg]

    return isnothing(η) ? alg_type() : alg_type(η)
end

function truncation_scheme(
    trscheme::TruncationScheme, direction::Int, row::Int, col::Int; kwargs...
)
    return trscheme
end

function truncation_scheme(
    trscheme::SiteDependentTruncation, direction::Int, row::Int, col::Int;
)
    return trscheme.trschemes[direction, row, col]
end

# Mirror a TruncationScheme by its anti-diagonal line.
# When the number of directions is 2, it swaps the first and second direction, consistent with xbonds and ybonds, respectively.
# When the number of directions is 4, it swaps the first and second, and third and fourth directions, consistent with the order NORTH, EAST, SOUTH, WEST.
mirror_antidiag(trscheme::TruncationScheme) = trscheme
function mirror_antidiag(trscheme::SiteDependentTruncation)
    directions = size(trscheme.trschemes)[1]
    if directions == 2
        trschemes_mirrored = stack(
            (
                mirror_antidiag(trscheme.trschemes[EAST, :, :]),
                mirror_antidiag(trscheme.trschemes[NORTH, :, :]),
            );
            dims=1,
        )
    elseif directions == 4
        trschemes_mirrored = stack((
            mirror_antidiag(trscheme.trschemes[EAST, :, :]),
            mirror_antidiag(trscheme.trschemes[NORTH, :, :]),
            mirror_antidiag(trscheme.trschemes[WEST, :, :]),
            mirror_antidiag(trscheme.trschemes[SOUTH, :, :]),
        ))
    else
        error("Unsupported number of directions for mirror_antidiag: $directions")
    end
    return SiteDependentTruncation(trschemes_mirrored)
end

# TODO: type piracy
Base.rotl90(trscheme::TruncationScheme) = trscheme

function Base.rotl90(trscheme::SiteDependentTruncation)
    directions = size(trscheme.trschemes)[1]
    trschemes_rotated = permutedims(trscheme.trschemes, (1, 3, 2))
    if directions == 2
        trschemes_rotated[NORTH, :, :] = circshift(
            rotl90(trscheme.trschemes[EAST, :, :]), (0, -1)
        )
        trschemes_rotated[EAST, :, :] = rotl90(trscheme.trschemes[NORTH, :, :])
    elseif directions == 4
        trschemes_rotated[NORTH, :, :] = rotl90(trscheme.trschemes[EAST, :, :])
        trschemes_rotated[EAST, :, :] = rotl90(trscheme.trschemes[SOUTH, :, :])
        trschemes_rotated[SOUTH, :, :] = rotl90(trscheme.trschemes[WEST, :, :])
        trschemes_rotated[WEST, :, :] = rotl90(trscheme.trschemes[NORTH, :, :])
    else
        error("Unsupported number of directions for rotl90: $directions")
    end
    return SiteDependentTruncation(trschemes_rotated)
end

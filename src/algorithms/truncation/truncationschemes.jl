"""
$(TYPEDEF)

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TruncationScheme end

struct SiteDependentTruncation <: TruncationScheme
    trschemes::Array{T,3} where {T<:TruncationScheme}
end

const TRUNCATION_SCHEME_SYMBOLS = IdDict{Symbol,Type{<:TruncationScheme}}(
    :fixedspace => FixedSpaceTruncation,
    :notrunc => TensorKit.NoTruncation,
    :truncerr => TensorKit.TruncationError,
    :truncdim => TensorKit.TruncationDimension,
    :truncspace => TensorKit.TruncationSpace,
    :truncbelow => TensorKit.TruncationCutoff,
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
    trscheme::T, direction::Int, row::Int, col::Int; kwargs...
) where {T<:TruncationScheme}
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
function mirror_antidiag(trscheme::T) where {T<:TruncationScheme}
    return trscheme
end
function mirror_antidiag(trscheme::T) where {T<:SiteDependentTruncation}
    directions = size(trscheme.trschemes)[1]
    trschemes_mirrored = permutedims(trscheme.trschemes, (1, 3, 2))
    if directions == 2
        trschemes_mirrored[1, :, :] = mirror_antidiag(trscheme.trschemes[2, :, :])
        trschemes_mirrored[2, :, :] = mirror_antidiag(trscheme.trschemes[1, :, :])
    elseif directions == 4
        trschemes_mirrored[1, :, :] = mirror_antidiag(trscheme.trschemes[2, :, :])
        trschemes_mirrored[2, :, :] = mirror_antidiag(trscheme.trschemes[1, :, :])
        trschemes_mirrored[3, :, :] = mirror_antidiag(trscheme.trschemes[4, :, :])
        trschemes_mirrored[4, :, :] = mirror_antidiag(trscheme.trschemes[3, :, :])
    else
        error("Unsupported number of directions for mirror_antidiag: $directions")
    end
    return SiteDependentTruncation(trschemes_mirrored)
end

# TODO: type piracy
function Base.rotl90(trscheme::T) where {T<:TruncationScheme}
    return trscheme
end

function Base.rotl90(trscheme::T) where {T<:SiteDependentTruncation}
    directions = size(trscheme.trschemes)[1]
    trschemes_rotated = permutedims(trscheme.trschemes, (1, 3, 2))
    if directions == EAST
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

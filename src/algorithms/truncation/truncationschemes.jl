"""
$(TYPEDEF)

CTMRG specific truncation scheme for `tsvd` which keeps the bond space on which the SVD
is performed fixed. Since different environment directions and unit cell entries might
have different spaces, this truncation style is different from `TruncationSpace`.
"""
struct FixedSpaceTruncation <: TensorKit.TruncationScheme end

struct SiteDependentTruncation <: TensorKit.TruncationScheme
    trschemes::Array{T,3} where {T<:TensorKit.TruncationScheme}
end

function SiteDependentTruncation(trscheme::TensorKit.TruncationScheme, Nr::Int, Nc::Int)
    return SiteDependentTruncation(reshape(fill(trscheme, Nr, Nc), 2, Nr, Nc))
end

function SiteDependentTruncation(
    trschemes::Tuple{T,S}, Nr::Int, Nc::Int
) where {T<:TensorKit.TruncationScheme,S<:TensorKit.TruncationScheme}
    return SiteDependentTruncation(
        reshape([trschemes[mod1(i, 2)] for i in 1:(2 * Nr * Nc)], 2, Nr, Nc)
    )
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

function truncation_scheme(trscheme::T; kwargs...) where {T<:TensorKit.TruncationScheme}
    return trscheme
end

function truncation_scheme(
    trscheme::SiteDependentTruncation;
    direction::Int,
    r::Int,
    c::Int,
    mirror_antidiag::Bool=false,
)
    if mirror_antidiag && direction == 2
        depth, Nr, Nc = size(trscheme.trschemes)
        @assert depth == 2
        return trscheme.trschemes[direction, Nc - c + 1, Nr - r + 1]
    end
    return trscheme.trschemes[direction, r, c]
end

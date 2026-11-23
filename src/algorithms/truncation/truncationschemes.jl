"""
$(TYPEDEF)

SVD truncation strategy which preserves the `CTMRGEnv` environment virtual spaces,
or `InfinitePEPS`, `InfinitePEPO` virtual spaces.
"""
struct FixedSpaceTruncation <: TruncationStrategy end

"""
$(TYPEDEF)

SVD truncation strategy specified for each nearest neighbor bond
in `InfinitePEPS`, `InfinitePEPO` (`2 × rows × cols`):
- `trunc[1, r, c]` applies to the x-bond between `[r, c]` and `[r, c+1]`.
    If it is a `TruncationSpace`, the space refers to the east domain of
    `[r, c]` or its `flip`, whichever is non-dual.
- `trunc[2, r, c]` applies to the y-bond between `[r, c]` and `[r-1, c]`.
    If it is a `TruncationSpace`, the space refers to the north domain of
    `[r, c]` or its `flip`, whichever is non-dual.

or for each environment bond of a `CTMRGEnv` (`4 × rows × cols`, `SimultaneousCTMRG` and `SequentialCTMRG` only):
- `trunc[dir, r, c]` applies to the bond of the first virtual space of `env.edges[dir, r, c]`.
    If it is a `TruncationSpace`, the space refers to that space or its `flip`,
    whichever is non-dual.
"""
struct SiteDependentTruncation{T <: TruncationStrategy} <: TruncationStrategy
    truncs::Array{T, 3}

    function SiteDependentTruncation(truncs::Array{T, 3}) where {T}
        size(truncs, 1) in (2, 4) || throw(
            DimensionMismatch(
                "The first dimension of `truncs` must have a size of 2 (PEPS bonds) or 4 (environment bonds). Got $(size(truncs, 1))."
            )
        )
        return new{T}(truncs)
    end
end

Base.@propagate_inbounds Base.getindex(trunc::SiteDependentTruncation, I...) =
    periodic_getindex(trunc, trunc.truncs, I)

# TODO: _is_bipartite(trunc::SiteDependentTruncation)

const TRUNCATION_STRATEGY_SYMBOLS = IdDict{Symbol, Base.Callable}(
    :notrunc => notrunc,
    :truncerror => (η::Real) -> truncerror(; atol = η),
    :truncrank => truncrank,
    :trunctol => (η::Real) -> trunctol(; atol = η),
    :truncspace => truncspace,
    :FixedSpaceTruncation => FixedSpaceTruncation,
    :SiteDependentTruncation => SiteDependentTruncation,
)

# Should be TruncationStrategy but rename to avoid type piracy
function _TruncationStrategy(; alg = Defaults.trunc, η = nothing)
    # replace Symbol with TruncationStrategy constructor
    haskey(TRUNCATION_STRATEGY_SYMBOLS, alg) ||
        throw(ArgumentError("unknown truncation strategy: $alg"))
    f = TRUNCATION_STRATEGY_SYMBOLS[alg]

    args = isnothing(η) ? () : (η,)
    applicable(f, args...) ||
        throw(ArgumentError("invalid truncation parameter for $alg: η = $η"))
    return f(args...)
end

function truncation_strategy(
        trunc::TruncationStrategy, direction::Int, row::Int, col::Int; kwargs...
    )
    return trunc
end

function truncation_strategy(
        trunc::SiteDependentTruncation, direction::Int, row::Int, col::Int
    )
    return trunc[direction, row, col]
end

# rotation of SiteDependentTruncation
# (similar to rotation of SUWeight)
function _rotl90_trunc_x(trunc_x::AbstractMatrix{<:TruncationStrategy})
    trunc_y = rotl90(trunc_x)
    return trunc_y
end
function _rotr90_trunc_x(trunc_x::AbstractMatrix{<:TruncationStrategy})
    trunc_y = circshift(rotr90(trunc_x), (1, 0))
    for (i, t) in enumerate(trunc_y)
        if t isa TruncationSpace
            trunc_y[i] = truncspace(flip(t.space)')
        end
    end
    return trunc_y
end
function _rot180_trunc_x(trunc_x::AbstractMatrix{<:TruncationStrategy})
    trunc_x_ = circshift(rot180(trunc_x), (0, -1))
    for (i, t) in enumerate(trunc_x_)
        trunc_x_[i] = truncspace(flip(t.space)')
    end
    return trunc_x_
end

function _rotl90_trunc_y(trunc_y::AbstractMatrix{<:TruncationStrategy})
    trunc_x = circshift(rotl90(trunc_y), (0, -1))
    for (i, t) in enumerate(trunc_x)
        if t isa TruncationSpace
            trunc_x[i] = truncspace(flip(t.space)')
        end
    end
    return trunc_x
end
function _rotr90_trunc_y(trunc_y::AbstractMatrix{<:TruncationStrategy})
    trunc_x = rotr90(trunc_y)
    return trunc_x
end
function _rot180_trunc_y(trunc_y::AbstractMatrix{<:TruncationStrategy})
    trunc_y_ = circshift(rot180(trunc_y), (1, 0))
    for (i, t) in enumerate(trunc_y_)
        trunc_y_[i] = truncspace(flip(t.space)')
    end
    return trunc_y_
end

# environment bonds rotate like the edges of a `CTMRGEnv`
function _rotate_env_truncs(rot, shift::Int, truncs)
    return SiteDependentTruncation(stack(rot(truncs[mod1(dir + shift, 4), :, :]) for dir in 1:4; dims = 1))
end

function Base.rotl90(trunc::SiteDependentTruncation)
    size(trunc.truncs, 1) == 4 && return _rotate_env_truncs(rotl90, 1, trunc.truncs)
    trunc_y = _rotl90_trunc_x(trunc.truncs[1, :, :])
    trunc_x = _rotl90_trunc_y(trunc.truncs[2, :, :])
    trunc = stack((trunc_x, trunc_y); dims = 1)
    return SiteDependentTruncation(trunc)
end
function Base.rotr90(trunc::SiteDependentTruncation)
    size(trunc.truncs, 1) == 4 && return _rotate_env_truncs(rotr90, -1, trunc.truncs)
    trunc_y = _rotr90_trunc_x(trunc.truncs[1, :, :])
    trunc_x = _rotr90_trunc_y(trunc.truncs[2, :, :])
    trunc = stack((trunc_x, trunc_y); dims = 1)
    return SiteDependentTruncation(trunc)
end
function Base.rot180(trunc::SiteDependentTruncation)
    size(trunc.truncs, 1) == 4 && return _rotate_env_truncs(rot180, 2, trunc.truncs)
    trunc_x = _rot180_trunc_x(trunc.truncs[1, :, :])
    trunc_y = _rot180_trunc_y(trunc.truncs[2, :, :])
    trunc = stack((trunc_x, trunc_y); dims = 1)
    return SiteDependentTruncation(trunc)
end

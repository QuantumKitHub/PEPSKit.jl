# PEPS
# ----

"""
    struct PEPS_∂∂C{T<:GenericMPSTensor{S,N}}

Represents the effective Hamiltonian for the zero-site derivative of an MPS.
"""
struct PEPS_∂∂C{T}
    GL::T
    GR::T
end

"""
    struct PEPS_∂∂AC{T,O1,O2}

Represents the effective Hamiltonian for the one-site derivative of an MPS.
"""
struct PEPS_∂∂AC{T,O}
    top::O
    bot::O
    GL::T
    GR::T
end

(H::PEPS_∂∂C)(x) = MPSKit.∂C(x, H.GL, H.GR)
(H::PEPS_∂∂AC)(x) = MPSKit.∂AC(x, (H.top, H.bot), H.GL, H.GR)

function MPSKit.∂AC(x::Vector, O::Tuple, GL, GR)
    return circshift(
        map((v, O1, O2, l, r) -> MPSKit.∂AC(v, (O1, O2), l, r), x, O[1], O[2], GL, GR), 1
    )
end

Base.:*(H::Union{<:PEPS_∂∂AC,<:PEPS_∂∂C}, v) = H(v)

# operator constructors
function MPSKit.∂∂C(pos::Int, mps, mpo::InfiniteTransferPEPS, cache)
    return PEPS_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂C(row::Int, col::Int, mps, mpo::MultilineTransferPEPS, cache)
    return PEPS_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))
end
function MPSKit.∂∂C(col::Int, mps, mpo::MultilineTransferPEPS, cache)
    return PEPS_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
end

function MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferPEPS, cache)
    return PEPS_∂∂AC(mpo[pos]..., leftenv(cache, pos, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::MultilineTransferPEPS, cache)
    return PEPS_∂∂AC(
        mpo[row, col]..., leftenv(cache, row, col, mps), rightenv(cache, row, col, mps)
    )
end
function MPSKit.∂∂AC(col::Int, mps, mpo::MultilineTransferPEPS, cache)
    return PEPS_∂∂AC(
        first.(mpo[:, col]),
        last.(mpo[:, col]),
        leftenv(cache, col, mps),
        rightenv(cache, col, mps),
    )
end

# PEPO
# ----

"""
    struct PEPO_∂∂C{T<:GenericMPSTensor{S,N}}

Represents the effective Hamiltonian for the zero-site derivative of an MPS.
"""
struct PEPO_∂∂C{T}
    GL::T
    GR::T
end

"""
    struct PEPO_∂∂AC{T,O,P}

Represents the effective Hamiltonian for the one-site derivative of an MPS.
"""
struct PEPO_∂∂AC{T,O,P}
    top::O
    bot::O
    mid::P
    GL::T
    GR::T
end

(H::PEPO_∂∂C)(x) = MPSKit.∂C(x, H.GL, H.GR)
(H::PEPO_∂∂AC)(x) = MPSKit.∂AC(x, (H.top, H.bot, H.mid), H.GL, H.GR)

function MPSKit.∂AC(x::Vector, O::Tuple{T,T,P}, GL, GR) where {T,P}
    return circshift(
        map(
            (v, O1, O2, O3, l, r) -> MPSKit.∂AC(v, (O1, O2, O3), l, r),
            x,
            O[1],
            O[2],
            O[3],
            GL,
            GR,
        ),
        1,
    )
end

Base.:*(H::Union{<:PEPO_∂∂AC,<:PEPO_∂∂C}, v) = H(v)

# Operator constructors
# ---------------------

function MPSKit.∂∂C(pos::Int, mps, ::InfiniteTransferPEPO, cache)
    return PEPO_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂C(col::Int, mps, ::MultilineTransferPEPO, cache)
    return PEPO_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
end
function MPSKit.∂∂C(row::Int, col::Int, mps, ::MultilineTransferPEPO, cache)
    return PEPO_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))
end

function MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferPEPO, cache)
    return PEPO_∂∂AC(mpo[pos]..., leftenv(cache, pos, mps), rightenv(cache, pos, mps))
end
function MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::MultilineTransferPEPO, cache)
    return PEPO_∂∂AC(
        mpo[row, col]..., leftenv(cache, row, col, mps), rightenv(cache, row, col, mps)
    )
end
function MPSKit.∂∂AC(col::Int, mps, mpo::MultilineTransferPEPO, cache)
    return PEPO_∂∂AC(
        map(x -> x[1], mpo[:, col]),
        map(x -> x[2], mpo[:, col]),
        map(x -> x[3], mpo[:, col]),
        leftenv(cache, col, mps),
        rightenv(cache, col, mps),
    )
end

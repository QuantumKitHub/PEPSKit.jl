# only need derivative operator constructions, the rest is handled by the contractions

# TODO: remove all of this once the type restrictions in MPSKit are removed

using MPSKit: MPO_∂∂C, MPO_∂∂AC

function MPSKit.∂∂C(pos::Int, mps, mpo::InfiniteTransferMatrix, cache)
    return MPO_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function MPSKit.MPSKit.∂∂C(row::Int, col::Int, mps, mpo::MultilineTransferMatrix, cache)
    return MPO_∂∂C(leftenv(cache, row, col + 1, mps), rightenv(cache, row, col, mps))
end
function MPSKit.MPSKit.∂∂C(col::Int, mps, mpo::MultilineTransferMatrix, cache)
    return MPO_∂∂C(leftenv(cache, col + 1, mps), rightenv(cache, col, mps))
end

function MPSKit.MPSKit.∂∂AC(pos::Int, mps, mpo::InfiniteTransferMatrix, cache)
    return MPO_∂∂AC(mpo[pos], leftenv(cache, pos, mps), rightenv(cache, pos, mps))
end
function MPSKit.MPSKit.∂∂AC(row::Int, col::Int, mps, mpo::MultilineTransferPEPS, cache)
    return MPO_∂∂AC(
        mpo[row, col], leftenv(cache, row, col, mps), rightenv(cache, row, col, mps)
    )
end
function MPSKit.MPSKit.∂∂AC(col::Int, mps, mpo::MultilineTransferMatrix, cache)
    return MPO_∂∂AC(mpo[:, col], leftenv(cache, col, mps), rightenv(cache, col, mps))
end

# TODO: AC2 operators and contractions

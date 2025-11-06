#
# Edge transfer matrices
#

# single site transfer
struct EdgeTransferMatrix{A <: CTMRGEdgeTensor, B, C <: CTMRGEdgeTensor} <: MPSKit.AbstractTransferMatrix
    top::A
    mid::B
    bot::C
    isflipped::Bool
end

Base.:*(tm1::T, tm2::T) where {T <: EdgeTransferMatrix} = ProductTransferMatrix([tm1, tm2])

# TODO: really not sure it TensorKit.flip is the suitable method for this...
function TensorKit.flip(tm::EdgeTransferMatrix)
    return EdgeTransferMatrix(tm.top, tm.mid, tm.bot, !tm.isflipped)
end

# action on a vector using * is dispatched in MPSKit to regular function application
function (d::EdgeTransferMatrix)(vec)
    return if d.isflipped
        edge_transfer_left(vec, d.mid, d.top, d.bot)
    else
        edge_transfer_right(vec, d.mid, d.top, d.bot)
    end
end

# constructors
edge_transfermatrix(a) = edge_transfermatrix(a, nothing, a)
edge_transfermatrix(a, b) = edge_transfermatrix(a, nothing, b)
function edge_transfermatrix(a::CTMRGEdgeTensor, b, c::CTMRGEdgeTensor, isflipped = false)
    return EdgeTransferMatrix(a, b, c, isflipped)
end
function edge_transfermatrix(a::AbstractVector, b, c::AbstractVector, isflipped = false)
    tot = ProductTransferMatrix(convert(Vector, edge_transfermatrix.(a, b, c)))
    return isflipped ? flip(tot) : tot
end

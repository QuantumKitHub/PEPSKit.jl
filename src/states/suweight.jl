"""
Schmidt bond weight used in simple/cluster update
"""
struct SUWeight{T<:AbstractTensorMap}
    x::Matrix{T}
    y::Matrix{T}

    function SUWeight(wxs::Matrix{T}, wys::Matrix{T}) where {T}
        return new{T}(wxs, wys)
    end
end

function Base.size(wts::SUWeight)
    @assert size(wts.x) == size(wts.y)
    return size(wts.x)
end

function Base.:(==)(wts1::SUWeight, wts2::SUWeight)
    return wts1.x == wts2.x && wts1.y == wts2.y
end

function Base.:(+)(wts1::SUWeight, wts2::SUWeight)
    return SUWeight(wts1.x + wts2.x, wts1.y + wts2.y)
end

function Base.:(-)(wts1::SUWeight, wts2::SUWeight)
    return SUWeight(wts1.x - wts2.x, wts1.y - wts2.y)
end

function Base.iterate(wts::SUWeight, state=1)
    nx = prod(size(wts.x))
    if 1 <= state <= nx
        return wts.x[state], state + 1
    elseif nx + 1 <= state <= 2 * nx
        return wts.y[state - nx], state + 1
    else
        return nothing
    end
end

function Base.length(wts::SUWeight)
    @assert size(wts.x) == size(wts.y)
    return 2 * prod(size(wts.x))
end

function Base.isapprox(wts1::SUWeight, wts2::SUWeight; atol=0.0, rtol=1e-5)
    return (
        isapprox(wts1.x, wts2.x; atol=atol, rtol=rtol) &&
        isapprox(wts1.y, wts2.y; atol=atol, rtol=rtol)
    )
end

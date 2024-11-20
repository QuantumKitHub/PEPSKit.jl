
"""
    const PEPSWeight{S}

Default type for PEPS bond weights with 2 virtual indices,
conventionally ordered as: ``wt : ES ← WN``. 
Here, `ES`, `WN` denote the east/south, west/north spaces, respectively.
"""
const PEPSWeight{S} = AbstractTensorMap{S,1,1} where {S<:ElementarySpace}

"""
Schmidt bond weight used in simple/cluster update
"""
struct SUWeight{E<:PEPSWeight}
    x::Matrix{E}
    y::Matrix{E}
end

function Base.size(wts::SUWeight)
    @assert size(wts.x) == size(wts.y)
    return size(wts.x)
end

function Base.eltype(wts::SUWeight)
    @assert eltype(wts.x) == eltype(wts.y)
    return eltype(wts.x)
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

function Base.show(io::IO, wts::SUWeight)
    N1, N2 = size(wts)
    for (direction, r, c) in Iterators.product("xy", 1:N1, 1:N2)
        println(io, "$direction[$r,$c]: ")
        wt = (direction == 'x' ? wts.x[r, c] : wts.y[r, c])
        for (k, b) in blocks(wt)
            println(io, k, " = ", diag(b))
        end
    end
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

function compare_weights(wts1::SUWeight, wts2::SUWeight)
    wtdiff = sum(_singular_value_distance((wt1, wt2)) for (wt1, wt2) in zip(wts1, wts2))
    return wtdiff / (2 * prod(size(wts1)))
end

"""
Represents an infinite projected entangled-pair state on a 2D square lattice
consisting of vertex tensors and bond weights
"""
struct InfiniteWeightPEPS{T<:PEPSTensor,E<:PEPSWeight} <: AbstractPEPS
    vertices::Matrix{T}
    weights::SUWeight{E}

    function InfiniteWeightPEPS(
        vertices::Matrix{T}, weights::SUWeight{E}
    ) where {T<:PEPSTensor,E<:PEPSWeight}
        @assert size(vertices) == size(weights)
        Nr, Nc = size(vertices)
        for (r, c) in Iterators.product(1:Nr, 1:Nc)
            space(weights.y[r, c], 1)' == space(vertices[r, c], 2) || throw(
                SpaceMismatch("South space of bond weight y$((r, c)) does not match.")
            )
            space(weights.y[r, c], 2)' == space(vertices[_prev(r, Nr), c], 4) || throw(
                SpaceMismatch("North space of bond weight y$((r, c)) does not match.")
            )
            space(weights.x[r, c], 1)' == space(vertices[r, c], 3) ||
                throw(SpaceMismatch("West space of bond weight x$((r, c)) does not match."))
            space(weights.x[r, c], 2)' == space(vertices[r, _next(c, Nc)], 5) ||
                throw(SpaceMismatch("West space of bond weight x$((r, c)) does not match."))
        end
        return new{T,E}(vertices, weights)
    end
end

"""
Create an InfiniteWeightPEPS from matrices of vertex tensors,
x-weights and y-weights
"""
function InfiniteWeightPEPS(
    vertices::Matrix{T}, wts_x::Matrix{E}, wts_y::Matrix{E}
) where {T<:PEPSTensor,E<:PEPSWeight}
    weights = SUWeight(wts_x, wts_y)
    return InfiniteWeightPEPS(vertices, weights)
end

"""
Create an InfiniteWeightPEPS by specifying its physical, north and east spaces and unit cell.
Spaces can be specified either via `Int` or via `ElementarySpace`.
Bond weights are initialized as identity matrices. 
"""
function InfiniteWeightPEPS(
    f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpace}
    vertices = InfinitePEPS(f, T, Pspace, Nspace, Espace; unitcell=unitcell).A
    weights = SUWeight(fill(id(Espace), unitcell), fill(id(Nspace), unitcell))
    return InfiniteWeightPEPS(vertices, weights)
end

"""
Absorb bond weights into vertex tensors
"""
function InfinitePEPS(peps::InfiniteWeightPEPS)
    vertices = deepcopy(peps.vertices)
    N1, N2 = size(vertices)
    for (r, c) in Iterators.product(1:N1, 1:N2)
        for ax in 2:5
            vertices[r, c] = absorb_wt(vertices[r, c], r, c, ax, peps.weights; sqrtwt=true)
        end
    end
    return InfinitePEPS(vertices)
end

function Base.size(peps::InfiniteWeightPEPS)
    @assert size(peps.weights.x) == size(peps.weights.y) == size(peps.vertices)
    return size(peps.vertices)
end

function Base.eltype(peps::InfiniteWeightPEPS)
    @assert eltype(peps.weights) == eltype(peps.vertices)
    return eltype(peps.vertices)
end

"""
Mirror the unit cell of an iPEPS with weights by its anti-diagonal line
"""
function mirror_antidiag!(peps::InfiniteWeightPEPS)
    peps.vertices[:] = mirror_antidiag(peps.vertices)
    for (i, t) in enumerate(peps.vertices)
        peps.vertices[i] = permute(t, (1,), (3, 2, 5, 4))
    end
    peps.weights.x[:], peps.weights.y[:] = mirror_antidiag(peps.weights.y),
    mirror_antidiag(peps.weights.x)
    return nothing
end

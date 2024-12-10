
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
const SUWeight{E} = Array{E,3} where {E<:PEPSWeight}

function compare_weights(wts1::SUWeight, wts2::SUWeight)
    @assert size(wts1) == size(wts2)
    wtdiff = sum(_singular_value_distance((wt1, wt2)) for (wt1, wt2) in zip(wts1, wts2))
    return wtdiff / length(wts1)
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
        @assert size(vertices) == size(weights)[2:end]
        Nr, Nc = size(vertices)
        for (r, c) in Iterators.product(1:Nr, 1:Nc)
            space(weights[2, r, c], 1)' == space(vertices[r, c], 2) || throw(
                SpaceMismatch("South space of bond weight y$((r, c)) does not match.")
            )
            space(weights[2, r, c], 2)' == space(vertices[_prev(r, Nr), c], 4) || throw(
                SpaceMismatch("North space of bond weight y$((r, c)) does not match.")
            )
            space(weights[1, r, c], 1)' == space(vertices[r, c], 3) ||
                throw(SpaceMismatch("West space of bond weight x$((r, c)) does not match."))
            space(weights[1, r, c], 2)' == space(vertices[r, _next(c, Nc)], 5) ||
                throw(SpaceMismatch("West space of bond weight x$((r, c)) does not match."))
        end
        return new{T,E}(vertices, weights)
    end
end

"""
Create an InfiniteWeightPEPS from matrices of vertex tensors,
and separate matrices of weights on each type of bond.
"""
function InfiniteWeightPEPS(
    vertices::Matrix{T}, weight_mats::Matrix{E}...
) where {T<:PEPSTensor,E<:PEPSWeight}
    n_mat = length(weight_mats)
    Nr, Nc = size(weight_mats[1])
    @assert all((Nr, Nc) == size(weight_mat) for weight_mat in weight_mats)
    weights = collect(
        weight_mats[d][r, c] for (d, r, c) in Iterators.product(1:n_mat, 1:Nr, 1:Nc)
    )
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
    Nr, Nc = unitcell
    weights = collect(
        id(d == 1 ? Espace : Nspace) for
        (d, r, c) in Iterators.product(1:2, 1:Nr, 1:Nc)
    )
    return InfiniteWeightPEPS(vertices, weights)
end

"""
    absorb_weight(t::T, row::Int, col::Int, ax::Int, weights::SUWeight; sqrtwt::Bool=false, invwt::Bool=false) where {T<:PEPSTensor}

Absorb or remove environment weight on axis `ax` of PEPS tensor `t` 
known to be located at position (`row`, `col`) in the unit cell. 
Weights around the tensor at `(row, col)` are
```
                ↓
                y[r,c]
                ↓
    ←x[r,c-1] ← T[r,c] ← x[r,c] ←
                ↓
                y[r+1,c]
                ↓
```

# Arguments
- `t::T`: The tensor of type `T` (a subtype of `PEPSTensor`) to which the weight will be absorbed.
- `row::Int`: The row index specifying the position in the tensor network.
- `col::Int`: The column index specifying the position in the tensor network.
- `ax::Int`: The axis along which the weight is absorbed.
- `weights::SUWeight`: The weight object to absorb into the tensor.
- `sqrtwt::Bool=false` (optional): If `true`, the square root of the weight is used during absorption.
- `invwt::Bool=false` (optional): If `true`, the inverse of the weight is used during absorption.

# Details
The optional keywords `sqrtwt` and `invwt` allow for additional transformations on the weight before absorption. 
If both `sqrtwt` and `invwt` are `true`, the square root of the inverse weight will be used.
The first axis of `t` should be the physical axis. 

# Examples
```julia
# Absorb the weight into the 2nd axis of tensor at position (2, 3)
absorb_weight(t, 2, 3, 2, weights)

# Absorb the square root of the weight into the tensor
absorb_weight(t, 2, 3, 2, weights; sqrtwt=true)

# Absorb the inverse of (i.e. remove) the weight into the tensor
absorb_weight(t, 2, 3, 2, weights; invwt=true)
```
"""
function absorb_weight(
    t::T,
    row::Int,
    col::Int,
    ax::Int,
    weights::SUWeight;
    sqrtwt::Bool=false,
    invwt::Bool=false,
) where {T<:PEPSTensor}
    Nr, Nc = size(weights)[2:end]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 2 <= ax <= 5
    pow = (sqrtwt ? 1 / 2 : 1) * (invwt ? -1 : 1)
    if ax == 2 # north
        wt = weights[2, row, col]
    elseif ax == 3 # east
        wt = weights[1, row, col]
    elseif ax == 4 # south
        wt = weights[2, _next(row, Nr), col]
    else # west
        wt = weights[1, row, _prev(col, Nc)]
    end
    wt2 = sdiag_pow(wt, pow)
    indices_t = collect(-1:-1:-5)
    indices_t[ax] = 1
    indices_wt = (ax in (2, 3) ? [1, -ax] : [-ax, 1])
    t2 = permute(ncon((t, wt2), (indices_t, indices_wt)), ((1,), Tuple(2:5)))
    return t2
end

"""
Create `InfinitePEPS` from `InfiniteWeightPEPS` by absorbing bond weights into vertex tensors
"""
function InfinitePEPS(peps::InfiniteWeightPEPS)
    vertices = deepcopy(peps.vertices)
    N1, N2 = size(vertices)
    for (r, c) in Iterators.product(1:N1, 1:N2)
        for ax in 2:5
            vertices[r, c] = absorb_weight(
                vertices[r, c], r, c, ax, peps.weights; sqrtwt=true
            )
        end
    end
    return InfinitePEPS(vertices)
end

function Base.size(peps::InfiniteWeightPEPS)
    @assert size(peps.weights)[2:end] == size(peps.vertices)
    return size(peps.vertices)
end

function Base.eltype(peps::InfiniteWeightPEPS)
    @assert eltype(peps.weights) == eltype(peps.vertices)
    return eltype(peps.vertices)
end

"""
Mirror the unit cell of an iPEPS with weights by its anti-diagonal line
"""
function mirror_antidiag(peps::InfiniteWeightPEPS)
    Nr, Nc = size(peps)
    vertices2 = mirror_antidiag(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = permute(t, ((1,), (3, 2, 5, 4)))
    end
    weights2 = similar(peps.weights, (2, Nc, Nr))
    weights2[1, :, :] = mirror_antidiag(peps.weights[2, :, :])
    weights2[2, :, :] = mirror_antidiag(peps.weights[1, :, :])
    return InfiniteWeightPEPS(vertices2, weights2)
end


"""
    const PEPSWeight

Default type for PEPS bond weights with 2 virtual indices, conventionally ordered as: ``wt : WS ← EN``. 
`WS`, `EN` denote the west/south, east/north spaces for x/y-weights on the square lattice, respectively.
"""
const PEPSWeight{T,S} = AbstractTensorMap{T,S,1,1}

"""
    struct SUWeight{E<:PEPSWeight}

Schmidt bond weights used in simple/cluster update.
Weight elements are always real.
"""
struct SUWeight{E<:PEPSWeight}
    data::Array{E,3}

    function SUWeight(data::Array{E,3}) where {E<:PEPSWeight}
        @assert eltype(data[1]) <: Real
        return new{E}(data)
    end
end

function SUWeight(wts_mats::AbstractMatrix{E}...) where {E<:PEPSWeight}
    n_mat = length(wts_mats)
    Nr, Nc = size(wts_mats[1])
    @assert all((Nr, Nc) == size(wts_mat) for wts_mat in wts_mats)
    weights = collect(
        wts_mats[d][r, c] for (d, r, c) in Iterators.product(1:n_mat, 1:Nr, 1:Nc)
    )
    return SUWeight(weights)
end

## Shape and size
Base.size(W::SUWeight) = size(W.data)
Base.size(W::SUWeight, i) = size(W.data, i)
Base.length(W::SUWeight) = length(W.data)
Base.eltype(W::SUWeight) = eltype(typeof(W))
Base.eltype(::Type{SUWeight{E}}) where {E} = E
VectorInterface.scalartype(::Type{T}) where {T<:SUWeight} = scalartype(eltype(T))

Base.getindex(W::SUWeight, args...) = Base.getindex(W.data, args...)
Base.setindex!(W::SUWeight, args...) = (Base.setindex!(W.data, args...); W)
Base.axes(W::SUWeight, args...) = axes(W.data, args...)

function compare_weights(wts1::SUWeight, wts2::SUWeight)
    @assert size(wts1) == size(wts2)
    return sum(_singular_value_distance, zip(wts1.data, wts2.data)) / length(wts1)
end

function Base.show(io::IO, wts::SUWeight)
    println(io, typeof(wts))
    for idx in CartesianIndices(wts.data)
        println(io, Tuple(idx), ":")
        for (k, b) in blocks(wts.data[idx])
            println(io, k, " = ", diag(b))
        end
    end
    return nothing
end

"""
    struct InfiniteWeightPEPS{T<:PEPSTensor,E<:PEPSWeight}

Represents an infinite projected entangled-pair state on a 2D square lattice
consisting of vertex tensors and bond weights.
"""
struct InfiniteWeightPEPS{T<:PEPSTensor,E<:PEPSWeight}
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
                throw(SpaceMismatch("East space of bond weight x$((r, c)) does not match."))
        end
        return new{T,E}(vertices, weights)
    end
end

"""
    InfiniteWeightPEPS(
        vertices::Matrix{T}, weight_mats::Matrix{E}...
    ) where {T<:PEPSTensor,E<:PEPSWeight}

Create an InfiniteWeightPEPS from matrices of vertex tensors,
and separate matrices of weights on each type of bond at all locations in the unit cell.
"""
function InfiniteWeightPEPS(
    vertices::Matrix{T}, weight_mats::Matrix{E}...
) where {T<:PEPSTensor,E<:PEPSWeight}
    return InfiniteWeightPEPS(vertices, SUWeight(weight_mats...))
end

"""
    InfiniteWeightPEPS(
        f=randn, T=ComplexF64, Pspaces::M, Nspaces::M, [Espaces::M]
    ) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Create an InfiniteWeightPEPS by specifying the physical, north virtual and east virtual spaces
of the PEPS vertex tensor at each site in the unit cell as a matrix.
Each individual space can be specified as either an `Int` or an `ElementarySpace`.
Bond weights are initialized as identity matrices of element type `Float64`. 
"""
function InfiniteWeightPEPS(
    Pspaces::M, Nspaces::M, Espaces::M
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    return InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
function InfiniteWeightPEPS(
    f, T, Pspaces::M, Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    vertices = InfinitePEPS(f, T, Pspaces, Nspaces, Espaces).A
    Nr, Nc = size(vertices)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? Espaces[r, c] : Nspaces[r, c])
        DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS(vertices, SUWeight(weights))
end

"""
    InfiniteWeightPEPS(
        f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
    ) where {S<:ElementarySpace}

Create an InfiniteWeightPEPS by specifying its physical, north and east spaces (as `ElementarySpace`s) and unit cell size.
Use `T` to specify the element type of the vertex tensors. 
Bond weights are initialized as identity matrices of element type `Float64`. 
"""
function InfiniteWeightPEPS(
    f, T, Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpace}
    return InfiniteWeightPEPS(
        f, T, fill(Pspace, unitcell), fill(Nspace, unitcell), fill(Espace, unitcell)
    )
end

function Base.size(peps::InfiniteWeightPEPS)
    return size(peps.vertices)
end

function _absorb_weights(
    t::PEPSTensor,
    weights::SUWeight,
    row::Int,
    col::Int,
    axs::NTuple{N,Int},
    sqrtwts::NTuple{N,Bool},
    invwt::Bool,
) where {N}
    Nr, Nc = size(weights)[2:end]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 1 <= N <= 4
    tensors = Vector{AbstractTensorMap}()
    indices = Vector{Vector{Int}}()
    indices_t = collect(-1:-1:-5)
    for (ax, sqrtwt) in zip(axs, sqrtwts)
        @assert 1 <= ax <= 4
        axp1 = ax + 1
        indices_t[axp1] *= -1
        wt = if ax == NORTH
            weights[2, row, col]
        elseif ax == EAST
            weights[1, row, col]
        elseif ax == SOUTH
            weights[2, _next(row, Nr), col]
        else # WEST
            weights[1, row, _prev(col, Nc)]
        end
        # TODO: remove the dual constraint
        @assert !isdual(space(wt, 1)) && isdual(space(wt, 2))
        if (!sqrtwt && !invwt)
            push!(tensors, wt)
        else
            pow = (sqrtwt ? 1 / 2 : 1) * (invwt ? -1 : 1)
            push!(tensors, sdiag_pow(wt, pow))
        end
        push!(indices, (ax in (NORTH, EAST) ? [axp1, -axp1] : [-axp1, axp1]))
    end
    push!(tensors, t)
    push!(indices, indices_t)
    t2 = permute(ncon(tensors, indices), ((1,), Tuple(2:5)))
    return t2
end

"""
    absorb_weight(t::PEPSTensor, row::Int, col::Int, ax::Int, weights::SUWeight;
                  sqrtwt::Bool=false, invwt::Bool=false)

Absorb or remove environment weight on an axis of vertex tensor `t` 
known to be located at position (`row`, `col`) in the unit cell. 
Weights around the tensor at `(row, col)` are
```
                    ↓
                [2,r,c]
                    ↓
    ← [1,r,c-1] ← T[r,c] ← [1,r,c] ←
                    ↓
                [1,r+1,c]
                    ↓
```

# Arguments
- `t::T`: The vertex tensor to which the weight will be absorbed. The first axis of `t` should be the physical axis. 
- `row::Int`: The row index specifying the position in the tensor network.
- `col::Int`: The column index specifying the position in the tensor network.
- `ax::Int`: The axis into which the weight is absorbed, taking values from 1 to 4, standing for north, east, south, west respectively.
- `weights::SUWeight`: The weight object to absorb into the tensor.
- `sqrtwt::Bool=false` (optional): If `true`, the square root of the weight is absorbed.
- `invwt::Bool=false` (optional): If `true`, the inverse of the weight is absorbed.

# Details
The optional kwargs `sqrtwt` and `invwt` allow taking the square root or the inverse of the weight before absorption. 

# Examples
```julia
# Absorb the weight into the north axis of tensor at position (2, 3)
absorb_weight(t, 2, 3, 1, weights)

# Absorb the square root of the weight into the south axis
absorb_weight(t, 2, 3, 3, weights; sqrtwt=true)

# Absorb the inverse of (i.e. remove) the weight into the east axis
absorb_weight(t, 2, 3, 2, weights; invwt=true)
```
"""
function absorb_weight(
    t::PEPSTensor,
    row::Int,
    col::Int,
    ax::Int,
    weights::SUWeight;
    sqrtwt::Bool=false,
    invwt::Bool=false,
)
    return _absorb_weights(t, weights, row, col, (ax,), (sqrtwt,), invwt)
end

"""
    InfinitePEPS(peps::InfiniteWeightPEPS)

Create `InfinitePEPS` from `InfiniteWeightPEPS` by absorbing bond weights into vertex tensors.
"""
function InfinitePEPS(peps::InfiniteWeightPEPS)
    Nr, Nc = size(peps)
    axs = Tuple(1:4)
    _alltrue = ntuple(Returns(true), 4)
    return InfinitePEPS(
        collect(
            _absorb_weights(peps.vertices[r, c], peps.weights, r, c, axs, _alltrue, false)
            for r in 1:Nr, c in 1:Nc
        ),
    )
end

"""
    mirror_antidiag(peps::InfiniteWeightPEPS)

Mirror the unit cell of an iPEPS with weights by its anti-diagonal line.
"""
function mirror_antidiag(peps::InfiniteWeightPEPS)
    vertices2 = mirror_antidiag(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = permute(t, ((1,), (3, 2, 5, 4)))
    end
    weights2_x = mirror_antidiag(peps.weights[2, :, :])
    weights2_y = mirror_antidiag(peps.weights[1, :, :])
    return InfiniteWeightPEPS(vertices2, weights2_x, weights2_y)
end


"""
    const PEPSWeight

Default type for PEPS bond weights with 2 virtual indices.
"""
const PEPSWeight{T,S} = AbstractTensorMap{T,S,1,1}

"""
    struct SUWeight{E<:PEPSWeight}

Schmidt bond weights used in simple/cluster update. 
Weight elements are always real and non-negative.
The domain and codomain of each weight matrix must be an un-dualed `ElementarySpace`.

For a square lattice InfinitePEPS, the weights are placed as
```
        |
    -T[r-1,c]-
        |
    wt[2,r,c]
        |                   |
    --T[r,c]--wt[1,r,c]--T[r,c+1]--
        |                   |
```

## Fields

$(TYPEDFIELDS)

## Constructors

    SUWeight(wts_mats::AbstractMatrix{E}...) where {E<:PEPSWeight}
"""
struct SUWeight{E<:PEPSWeight}
    data::Array{E,3}
    SUWeight{E}(data::Array{E,3}) where {E} = new{E}(data)
end

function SUWeight(data::Array{E,3}) where {E<:PEPSWeight}
    scalartype(data) <: Real || error("Weight elements must be real numbers.")
    for wt in data
        isa(wt, DiagonalTensorMap) ||
            error("Each weight matrix should be a DiagonalTensorMap")
        domain(wt, 1) == codomain(wt, 1) ||
            error("Domain and codomain of each weight matrix must be the same.")
        !isdual(codomain(wt, 1)) ||
            error("Domain and codomain of each weight matrix cannot be a dual space.")
        all(wt.data .>= 0) || error("Weight elements must be non-negative.")
    end
    return SUWeight{E}(data)
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

"""
    SUWeight(Nspaces::M, [Espaces::M]) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Create an SUWeight by specifying the vertical (north) or horizontal (east) virtual bond spaces.
Each individual space can be specified as either an `Int` or an `ElementarySpace`.
The weights are initialized as identity matrices of element type `Float64`.
"""
function SUWeight(
    Nspaces::M, Espaces::M=Nspaces
) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    @assert all(!isdual, Nspaces)
    @assert all(!isdual, Espaces)
    @assert size(Nspaces) == size(Espaces)
    Nr, Nc = size(Nspaces)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? Espaces[r, c] : Nspaces[r, c])
        DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return SUWeight(weights)
end

"""
    SUWeight(Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)) where {S<:ElementarySpace}

Create an SUWeight by specifying its vertical (north) and horizontal (east) 
as `ElementarySpace`s) and unit cell size.
The weights are initialized as identity matrices of element type `Float64`.
"""
function SUWeight(
    Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)
) where {S<:ElementarySpace}
    return SUWeight(fill(Nspace, unitcell), fill(Espace, unitcell))
end

"""
    SUWeight(peps::InfinitePEPS)

Create an SUWeight for a given InfinitePEPS. 
The weights are initialized as identity matrices of element type `Float64`.
"""
function SUWeight(peps::InfinitePEPS)
    Nspaces = map(peps.A) do t
        V = domain(t, NORTH)
        isdual(V) ? V' : V
    end
    Espaces = map(peps.A) do t
        V = domain(t, EAST)
        isdual(V) ? V' : V
    end
    return SUWeight(Nspaces, Espaces)
end

## Shape and size
Base.size(W::SUWeight) = size(W.data)
Base.size(W::SUWeight, i) = size(W.data, i)
Base.length(W::SUWeight) = length(W.data)
Base.eltype(W::SUWeight) = eltype(typeof(W))
Base.eltype(::Type{SUWeight{E}}) where {E} = E
VI.scalartype(::Type{T}) where {T<:SUWeight} = scalartype(eltype(T))

Base.getindex(W::SUWeight, args...) = Base.getindex(W.data, args...)
Base.setindex!(W::SUWeight, args...) = (Base.setindex!(W.data, args...); W)
Base.axes(W::SUWeight, args...) = axes(W.data, args...)
Base.iterate(W::SUWeight, args...) = iterate(W.data, args...)

## spaces
TensorKit.spacetype(w::SUWeight) = spacetype(typeof(w))
TensorKit.spacetype(::Type{T}) where {E,T<:SUWeight{E}} = spacetype(E)
TensorKit.sectortype(w::SUWeight) = sectortype(typeof(w))
TensorKit.sectortype(::Type{<:SUWeight{T}}) where {T} = sectortype(spacetype(T))

## (Approximate) equality
function Base.:(==)(wts1::SUWeight, wts2::SUWeight)
    return wts1.data == wts2.data
end
function Base.isapprox(wts1::SUWeight, wts2::SUWeight; kwargs...)
    for (wt1, wt2) in zip(wts1, wts2)
        !isapprox(wt1, wt2; kwargs...) && return false
    end
    return true
end

function compare_weights(wts1::SUWeight, wts2::SUWeight)
    @assert size(wts1) == size(wts2)
    return sum(_singular_value_distance, zip(wts1.data, wts2.data)) / length(wts1)
end

function Base.show(io::IO, ::MIME"text/plain", wts::SUWeight)
    println(io, typeof(wts))
    for idx in CartesianIndices(wts.data)
        println(io, Tuple(idx), ":")
        for (k, b) in blocks(wts.data[idx])
            println(io, k, " = ", diag(b))
        end
    end
    return nothing
end

function _absorb_weights(
    t::PEPSTensor,
    row::Int,
    col::Int,
    axs::NTuple{N,Int},
    weights::SUWeight;
    invwt::Bool=false,
) where {N}
    Nr, Nc = size(weights)[2:end]
    @assert 1 <= row <= Nr && 1 <= col <= Nc && 1 <= N <= 4
    t2 = copy(t)
    for ax in axs
        t2 = absorb_weight(t2, row, col, ax, weights; invwt)
    end
    return t2
end

"""
    absorb_weight(t::PEPSTensor, row::Int, col::Int, ax::Int, weights::SUWeight; invwt::Bool=false)

Absorb or remove environment weight on an axis of vertex tensor `t`  known to be located at
position (`row`, `col`) in the unit cell. Weights around the tensor at `(row, col)` are
```
                    |
                [2,r,c]
                    |
    - [1,r,c-1] - T[r,c] - [1,r,c] -
                    |
                [1,r+1,c]
                    |
```

## Arguments

- `t::T` : The vertex tensor to which the weight will be absorbed. The first axis of `t` should be the physical axis.
- `row::Int` : The row index specifying the position in the tensor network.
- `col::Int` : The column index specifying the position in the tensor network.
- `ax::Int` : The axis into which the weight is absorbed, taking values from 1 to 4, standing for north, east, south, west respectively.
- `weights::SUWeight` : All simple update weights.

## Keyword arguments

- `invwt::Bool=false` : If `true`, the inverse square root of the weight is absorbed.

## Examples

```julia
# Absorb the weight into the north axis of tensor at position (2, 3)
absorb_weight(t, 2, 3, 1, weights)

# Absorb the inverse of (i.e. remove) the weight into the east axis
absorb_weight(t, 2, 3, 2, weights; invwt=true)
```
"""
function absorb_weight(
    t::PEPSTensor, row::Int, col::Int, ax::Int, weights::SUWeight; invwt::Bool=false
)
    Nr, Nc = size(weights)[2:end]
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 1 <= ax <= 4
    pow = invwt ? -1 / 2 : 1 / 2
    wt = sdiag_pow(
        if ax == NORTH
            weights[2, row, col]
        elseif ax == EAST
            weights[1, row, col]
        elseif ax == SOUTH
            weights[2, _next(row, Nr), col]
        else # WEST
            weights[1, row, _prev(col, Nc)]
        end,
        pow,
    )
    axp1 = ax + 1
    t_idx = collect((n - 1 == ax) ? 1 : -n for n in 1:5)
    wt_idx = if isdual(space(t, ax + 1))
        [1, -axp1] # t ← wt
    else
        [-axp1, 1] # t → wt
    end
    return permute(ncon((t, wt), (t_idx, wt_idx)), ((1,), Tuple(2:5)))
end

#= Rotation of SUWeight. Example: 3 x 3 network

- Original
    ```
            |         |         |
            y₁₁       y₁₂       y₁₃
            |         |         |
    ..x₁₃...┼---x₁₁---┼---x₁₂---┼---x₁₃---
            |         |         |
            y₂₁       y₂₂       y₂₃         |
            |         |         |           y
    ..x₂₃...┼---x₂₁---┼---x₂₂---┼---x₂₃---  |
            |         |         |
            y₃₁       y₃₂       y₃₃         -- x --
            |         |         |
    ..x₃₃...┼---x₃₁---┼---x₃₂---┼---x₃₃---
            :         :         :
            y₁₁       y₁₂       y₁₃
            :         :         :
    ```

- `rotl90`:
    ```
            |         |         |
            x₁₃       x₂₃       x₃₃
            |         |         |
    --y₁₃---┼---y₂₃---┼---y₃₃---┼...y₁₃...
            |         |         |
            x₁₂       x₂₂       x₃₂                 |
            |         |         |                   x
    --y₁₂---┼---y₂₂---┼---y₃₂---┼...y₁₂...          |
            |         |         |
            x₁₁       x₂₁       x₃₁         -- y --
            |         |         |
    --y₁₁---┼---y₂₁---┼---y₃₁---┼...y₁₁...
            :         :         :
            x₁₃       x₂₃       x₃₃
            :         :         :
    ```
    - x/y-weights are exchanged.
    - need to further move 1st column of x-weights to the last column.

- `rotr90`:
    ```
            :         :         :
            x₃₃       x₂₃       x₁₃
            :         :         :
    ..y₁₁...┼---y₃₁---┼---y₂₁---┼---y₁₁---
            |         |         |
            x₃₁       x₂₁       x₁₁         -- y --
            |         |         |
    ..y₁₂...┼---y₃₂---┼---y₂₂---┼---y₁₂---  |
            |         |         |           x
            x₃₂       x₂₂       x₁₂         |
            |         |         |
    ..y₁₃...┼---y₃₃---┼---y₂₃---┼---y₁₃---
            |         |         |
            x₃₃       x₂₃       x₁₃
            |         |         |
    ```
    - x/y-weights are exchanged.
    - need to further move last row of y-weights to the 1st row.

- `rot180`:
    ```
            :         :         :
            y₁₃       y₁₂       y₁₁
            :         :         :
    --x₃₃---┼---x₃₂---┼---x₃₁---┼...x₃₃...
            |         |         |
            y₃₃       y₃₂       y₃₁         -- x --
            |         |         |
    --x₂₃---┼---x₂₂---┼---x₂₁---┼...x₂₃...          |
            |         |         |                   y
            y₂₃       y₂₂       y₂₁                 |
            |         |         |
    --x₁₃---┼---x₁₂---┼---x₁₁---┼...x₁₃...
            |         |         |
            y₁₃       y₁₂       y₁₁
            |         |         |
    ```
    - need to move 1st column of x-weights to the last column.
    - need to move last row of y-weights to the 1st row.
=#

function Base.rotl90(wts::SUWeight)
    wts_x = circshift(rotl90(wts[2, :, :]), (0, -1))
    wts_y = rotl90(wts[1, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rotr90(wts::SUWeight)
    wts_x = rotr90(wts[2, :, :])
    wts_y = circshift(rotr90(wts[1, :, :]), (1, 0))
    return SUWeight(wts_x, wts_y)
end
function Base.rot180(wts::SUWeight)
    wts_x = circshift(rot180(wts[1, :, :]), (0, -1))
    wts_y = circshift(rot180(wts[2, :, :]), (1, 0))
    return SUWeight(wts_x, wts_y)
end

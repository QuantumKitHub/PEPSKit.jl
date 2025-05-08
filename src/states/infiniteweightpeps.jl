
"""
    const PEPSWeight

Default type for PEPS bond weights with 2 virtual indices, conventionally ordered as: ``wt : WS ← EN``. 
`WS`, `EN` denote the west/south, east/north spaces for x/y-weights on the square lattice, respectively.
"""
const PEPSWeight{T,S} = AbstractTensorMap{T,S,1,1}

"""
    struct SUWeight{E<:PEPSWeight}

Schmidt bond weights used in simple/cluster update. Weight elements are always real.

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
Base.iterate(W::SUWeight, args...) = iterate(W.data, args...)

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

"""
    struct InfiniteWeightPEPS{T<:PEPSTensor,E<:PEPSWeight}

Represents an infinite projected entangled-pair state on a 2D square lattice
consisting of vertex tensors and bond weights.

## Fields

$(TYPEDFIELDS)

## Constructors

    InfiniteWeightPEPS(vertices::Matrix{T}, weight_mats::Matrix{E}...) where {T<:PEPSTensor,E<:PEPSWeight}
    InfiniteWeightPEPS([f=randn, T=ComplexF64,] Pspaces::M, Nspaces::M, [Espaces::M]) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}
    InfiniteWeightPEPS([f=randn, T=ComplexF64,] Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)) where {S<:ElementarySpace}
"""
struct InfiniteWeightPEPS{T<:PEPSTensor,E<:PEPSWeight}
    vertices::Matrix{T}
    weights::SUWeight{E}

    function InfiniteWeightPEPS(
        vertices::Matrix{T}, weights::SUWeight{E}
    ) where {T<:PEPSTensor,E<:PEPSWeight}
        @assert size(vertices) == size(weights)[2:end]
        Nr, Nc = size(vertices)
        # check space matching between vertex tensors and weight matrices
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
    InfiniteWeightPEPS(vertices::Matrix{T}, weight_mats::Matrix{E}...) where {T<:PEPSTensor,E<:PEPSWeight}

Create an InfiniteWeightPEPS from matrices of vertex tensors,
and separate matrices of weights on each type of bond at all locations in the unit cell.
"""
function InfiniteWeightPEPS(
    vertices::Matrix{T}, weight_mats::Matrix{E}...
) where {T<:PEPSTensor,E<:PEPSWeight}
    return InfiniteWeightPEPS(vertices, SUWeight(weight_mats...))
end

"""
    InfiniteWeightPEPS([f=randn, T=ComplexF64,] Pspaces::M, Nspaces::M, [Espaces::M]) where {M<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

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
    @assert all(!isdual(Pspace) for Pspace in Pspaces)
    @assert all(!isdual(Nspace) for Nspace in Nspaces)
    @assert all(!isdual(Espace) for Espace in Espaces)
    vertices = InfinitePEPS(f, T, Pspaces, Nspaces, Espaces).A
    Nr, Nc = size(vertices)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? Espaces[r, c] : Nspaces[r, c])
        DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS(vertices, SUWeight(weights))
end

"""
    InfiniteWeightPEPS([f=randn, T=ComplexF64,] Pspace::S, Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1)) where {S<:ElementarySpace}

Create an InfiniteWeightPEPS by specifying its physical, north and east spaces (as `ElementarySpace`s) and unit cell size.
Use `T` to specify the element type of the vertex tensors. 
Bond weights are initialized as identity matrices of element type `Float64`. 
"""
function InfiniteWeightPEPS(Pspaces::S, Nspaces::S, Espaces::S) where {S<:ElementarySpace}
    return InfiniteWeightPEPS(randn, ComplexF64, Pspaces, Nspaces, Espaces)
end
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
    absorb_weight(t::PEPSTensor, row::Int, col::Int, ax::Int, weights::SUWeight; sqrtwt::Bool=false, invwt::Bool=false)

Absorb or remove environment weight on an axis of vertex tensor `t`  known to be located at
position (`row`, `col`) in the unit cell. Weights around the tensor at `(row, col)` are
```
                    ↓
                [2,r,c]
                    ↓
    ← [1,r,c-1] ← T[r,c] ← [1,r,c] ←
                    ↓
                [1,r+1,c]
                    ↓
```

## Arguments

- `t::T` : The vertex tensor to which the weight will be absorbed. The first axis of `t` should be the physical axis. 
- `row::Int` : The row index specifying the position in the tensor network.
- `col::Int` : The column index specifying the position in the tensor network.
- `ax::Int` : The axis into which the weight is absorbed, taking values from 1 to 4, standing for north, east, south, west respectively.
- `weights::SUWeight` : The weight object to absorb into the tensor.

## Keyword arguments 

- `sqrtwt::Bool=false` : If `true`, the square root of the weight is absorbed.
- `invwt::Bool=false` : If `true`, the inverse of the weight is absorbed.

## Examples

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
    InfiniteWeightPEPS(peps::InfinitePEPS)

Create `InfiniteWeightPEPS` from `InfinitePEPS` by initializing the bond weights as identity matrices of element type `Float64`.
"""
function InfiniteWeightPEPS(peps::InfinitePEPS)
    Nr, Nc = size(peps)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? domain(peps[r, c])[2] : domain(peps[r, c])[1])
        @assert !isdual(V)
        DiagonalTensorMap(ones(reduceddim(V)), V)
    end
    return InfiniteWeightPEPS(peps.A, SUWeight(weights))
end

## (Approximate) equality (gauge freedom is not allowed)
function Base.:(==)(peps1::InfiniteWeightPEPS, peps2::InfiniteWeightPEPS)
    return peps1.vertices == peps2.vertices && peps1.weights == peps2.weights
end
function Base.isapprox(peps1::InfiniteWeightPEPS, peps2::InfiniteWeightPEPS; kwargs...)
    for (v1, v2) in zip(peps1.vertices, peps2.vertices)
        !isapprox(v1, v2; kwargs...) && return false
    end
    !isapprox(peps1.weights, peps2.weights; kwargs...) && return false
    return true
end

# Mirroring and rotation
#= Example: 3 x 3 network

- Original
    ```
            |         |         |
            y₁₁       y₁₂       y₁₃
            |         |         |
    ..x₁₃...┼---x₁₁---┼---x₁₂---┼---x₁₃---
            |         |         |           2
            y₂₁       y₂₂       y₂₃         ↓
            |         |         |           y
    ..x₂₃...┼---x₂₁---┼---x₂₂---┼---x₂₃---  ↓
            |         |         |           1
            y₃₁       y₃₂       y₃₃
            |         |         |           1 ←- x ←- 2
    ..x₃₃...┼---x₃₁---┼---x₃₂---┼---x₃₃---
            :         :         :
            y₁₁       y₁₂       y₁₃
            :         :         :
    ```

- After `mirror_antidiag`, x/y-weights are exchanged. 
    ```
            |         |         |
            x₃₃       x₂₃       x₁₃
            |         |         |
    ..y₁₃...┼---y₃₃---┼---y₂₃---┼---y₁₃---
            |         |         |           2
            x₃₂       x₂₂       x₁₂         ↓
            |         |         |           x
    ..y₁₂...┼---y₃₂---┼---y₂₂---┼---y₁₂---  ↓
            |         |         |           1
            x₃₁       x₂₁       x₁₁
            |         |         |           1 ←- y ←- 2
    ..y₁₁...┼---y₃₁---┼---y₂₁---┼---y₁₁---
            :         :         :
            x₃₃       x₂₃       x₁₃
            :         :         :
    ```
    No further operations are needed. 

- After `rotl90`, x/y-weights are exchanged. 
    ```
            |         |         |
            x₁₃       x₂₃       x₃₃
            |         |         |
    --y₁₃---┼---y₂₃---┼---y₃₃---┼...y₁₃...
            |         |         |                     2
            x₁₂       x₂₂       x₃₂                   ↓
            |         |         |                     x
    --y₁₂---┼---y₂₂---┼---y₃₂---┼...y₁₂...            ↓
            |         |         |                     1
            x₁₁       x₂₁       x₃₁
            |         |         |           2 -→ y -→ 1
    --y₁₁---┼---y₂₁---┼---y₃₁---┼...y₁₁...
            :         :         :
            x₁₃       x₂₃       x₃₃
            :         :         :
    ```
    We need to further:
    - Move 1st column of x-weights to the last column.
    - Permute axes of x-weights.
    - Flip x-arrows from → to ←. 

- After `rotr90`, x/y-weights are exchanged. 
    ```
            :         :         :
            x₃₃       x₂₃       x₁₃
            :         :         :
    ..y₁₁...┼---y₃₁---┼---y₂₁---┼---y₁₁---
            |         |         |           1 ←- y ←- 2
            x₃₁       x₂₁       x₁₁
            |         |         |           1
    ..y₁₂...┼---y₃₂---┼---y₂₂---┼---y₁₂---  ↑
            |         |         |           x
            x₃₂       x₂₂       x₁₂         ↑
            |         |         |           2
    ..y₁₃...┼---y₃₃---┼---y₂₃---┼---y₁₃---
            |         |         |
            x₃₃       x₂₃       x₁₃
            |         |         |
    ```
    We need to further:
    - Move last row of y-weights to the 1st row. 
    - Permute axes of y-weights. 
    - Flip y-arrows from ↑ to ↓. 

After `rot180`, x/y-weights are not exchanged. 
    ```
            :         :         :
            y₁₃       y₁₂       y₁₁
            :         :         :
    --x₃₃---┼---x₃₂---┼---x₃₁---┼...x₃₃...
            |         |         |           2 -→ x -→ 1
            y₃₃       y₃₂       y₃₁
            |         |         |                     1
    --x₂₃---┼---x₂₂---┼---x₂₁---┼...x₂₃...            ↑
            |         |         |                     y
            y₂₃       y₂₂       y₂₁                   ↑
            |         |         |                     2
    --x₁₃---┼---x₁₂---┼---x₁₁---┼...x₁₃...
            |         |         |
            y₁₃       y₁₂       y₁₁
            |         |         |
    ```
    We need to further:
    - Move 1st column of x-weights to the last column.
    - Move last row of y-weights to the 1st row.
    - Permute axes of all weights and twist their axis 1. 
    - Flip x-arrows from → to ←, and y-arrows from ↑ to ↓. 
=#

"""
Mirror an `SUWeight` by its anti-diagonal line.
"""
function mirror_antidiag(wts::SUWeight)
    weights2_x = mirror_antidiag(wts[2, :, :])
    weights2_y = mirror_antidiag(wts[1, :, :])
    return SUWeight(weights2_x, weights2_y)
end
function Base.rotl90(wts::SUWeight)
    wts_x = circshift(rotl90(wts[2, :, :]), (0, -1))
    for (i, wt) in enumerate(wts_x)
        wts_x[i] = DiagonalTensorMap(flip(permute(wt, ((2,), (1,))), (1, 2)))
    end
    wts_y = rotl90(wts[1, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rotr90(wts::SUWeight)
    wts_x = rotr90(wts[2, :, :])
    wts_y = circshift(rotr90(wts[1, :, :]), (1, 0))
    for (i, wt) in enumerate(wts_y)
        wts_y[i] = DiagonalTensorMap(flip(permute(wt, ((2,), (1,))), (1, 2)))
    end
    return SUWeight(wts_x, wts_y)
end
function Base.rot180(wts::SUWeight)
    wts_x = circshift(rot180(wts[1, :, :]), (0, -1))
    wts_y = circshift(rot180(wts[2, :, :]), (1, 0))
    for (i, wt) in enumerate(wts_x)
        wts_x[i] = DiagonalTensorMap(flip(permute(wt, ((2,), (1,))), (1, 2)))
    end
    for (i, wt) in enumerate(wts_y)
        wts_y[i] = DiagonalTensorMap(flip(permute(wt, ((2,), (1,))), (1, 2)))
    end
    return SUWeight(wts_x, wts_y)
end

"""
    mirror_antidiag(peps::InfiniteWeightPEPS)

Mirror an `InfiniteWeightPEPS` by its anti-diagonal line.
"""
function mirror_antidiag(peps::InfiniteWeightPEPS)
    vertices2 = mirror_antidiag(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = mirror_antidiag(t)
    end
    weights2 = mirror_antidiag(peps.weights)
    return InfiniteWeightPEPS(vertices2, weights2)
end
function Base.rotl90(peps::InfiniteWeightPEPS)
    vertices2 = rotl90(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = flip(rotl90(t), (3, 5))
    end
    weights2 = rotl90(peps.weights)
    return InfiniteWeightPEPS(vertices2, weights2)
end
function Base.rotr90(peps::InfiniteWeightPEPS)
    vertices2 = rotr90(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = flip(rotr90(t), (2, 4))
    end
    weights2 = rotr90(peps.weights)
    return InfiniteWeightPEPS(vertices2, weights2)
end
function Base.rot180(peps::InfiniteWeightPEPS)
    vertices2 = rot180(peps.vertices)
    for (i, t) in enumerate(vertices2)
        vertices2[i] = flip(rot180(t), Tuple(2:5))
    end
    weights2 = rot180(peps.weights)
    return InfiniteWeightPEPS(vertices2, weights2)
end

"""
    const PEPSWeight

Default type for PEPS bond weights with 2 virtual indices.
"""
const PEPSWeight{T, S} = AbstractTensorMap{T, S, 1, 1}

"""
    struct SUWeight{E<:PEPSWeight}

Schmidt bond weights used in simple/cluster update.
Each weight is a real and semi-positive definite
`DiagonalTensorMap`, with the same codomain and domain.

On the square lattice,
- `wt[1,r,c]` is on the x-bond between `[r,c]` and `[r,c+1]`;
- `wt[2,r,c]` is on the y-bond between `[r,c]` and `[r-1,c]`.

Axis order of each weight matrix is
```
    x-weights:      y-weights:

    1 - x - 2           2
                        |
                        y
                        |
                        1
```

## Fields

$(TYPEDFIELDS)

## Constructors

    SUWeight(wts_mats::AbstractMatrix{E}...) where {E<:PEPSWeight}
"""
struct SUWeight{E <: PEPSWeight}
    data::Array{E, 3}
    SUWeight{E}(data::Array{E, 3}) where {E} = new{E}(data)
end

function SUWeight(data::Array{E, 3}) where {E <: PEPSWeight}
    scalartype(data) <: Real || error("Weight elements must be real numbers.")
    for wt in data
        isa(wt, DiagonalTensorMap) ||
            error("Each weight matrix should be a DiagonalTensorMap")
        # in case TensorKit drops this requirement
        domain(wt, 1) == codomain(wt, 1) ||
            error("Domain and codomain of each weight matrix must be the same.")
        all(wt.data .>= 0) || error("Weight elements must be non-negative.")
    end
    return SUWeight{E}(data)
end

function SUWeight(wts_mats::AbstractMatrix{E}...) where {E <: PEPSWeight}
    return SUWeight(stack(wts_mats; dims = 1))
end

"""
    SUWeight(Nspaces::M, [Espaces::M]) where {M<:AbstractMatrix{<:ElementarySpace}}

Create a trivial `SUWeight` by specifying the vertical (north) or horizontal (east) virtual bond spaces.
"""
function SUWeight(
        Nspaces::M, Espaces::M = Nspaces
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
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

Create a trivial `SUWeight` by specifying its vertical (north) and horizontal (east)
as `ElementarySpace`s) and unit cell size.
"""
function SUWeight(
        Nspace::S, Espace::S = Nspace; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return SUWeight(fill(Nspace, unitcell), fill(Espace, unitcell))
end

"""
    SUWeight(peps::InfinitePEPS)

Create a trivial `SUWeight` for a given InfinitePEPS.
"""
function SUWeight(peps::InfinitePEPS)
    Nspaces = map(Base.Fix2(domain, NORTH), unitcell(peps))
    Espaces = map(Base.Fix2(domain, EAST), unitcell(peps))
    return SUWeight(Nspaces, Espaces)
end

"""
    SUWeight(pepo::InfinitePEPO)

Create a trivial `SUWeight` for a given one-layer InfinitePEPO.
"""
function SUWeight(pepo::InfinitePEPO)
    @assert size(pepo, 3) == 1
    Nspaces = map(Base.Fix2(domain, NORTH), @view(unitcell(pepo)[:, :, 1]))
    Espaces = map(Base.Fix2(domain, EAST), @view(unitcell(pepo)[:, :, 1]))
    return SUWeight(Nspaces, Espaces)
end

Random.rand!(wts::SUWeight) = rand!(Random.default_rng(), wts)
function Random.rand!(rng::Random.AbstractRNG, wts::SUWeight)
    foreach(wts.data) do wt
        for (_, b) in blocks(wt)
            sort!(rand!(rng, b.diag); rev = true)
        end
    end
    return wts
end

## Shape and size
Base.size(W::SUWeight) = size(W.data)
Base.size(W::SUWeight, i) = size(W.data, i)
Base.length(W::SUWeight) = length(W.data)
Base.eltype(W::SUWeight) = eltype(typeof(W))
Base.eltype(::Type{SUWeight{E}}) where {E} = E
VI.scalartype(::Type{T}) where {T <: SUWeight} = scalartype(eltype(T))

Base.@propagate_inbounds Base.getindex(W::SUWeight, I...) =
    periodic_getindex(W, W.data, I)
Base.@propagate_inbounds Base.setindex!(W::SUWeight, v, I...) =
    periodic_setindex!(W, W.data, v, I)
Base.axes(W::SUWeight, args...) = axes(W.data, args...)
Base.iterate(W::SUWeight, args...) = iterate(W.data, args...)

## spaces
TensorKit.spacetype(w::SUWeight) = spacetype(typeof(w))
TensorKit.spacetype(::Type{T}) where {E, T <: SUWeight{E}} = spacetype(E)
TensorKit.sectortype(w::SUWeight) = sectortype(typeof(w))
TensorKit.sectortype(::Type{<:SUWeight{T}}) where {T} = sectortype(spacetype(T))

## Bipartite check
function _is_bipartite(wts::SUWeight)
    (size(wts, 2) == size(wts, 3) == 2) || (return false)
    for (d, c) in Iterators.product(1:2, 1:2)
        (wts[d, 1, c] == wts[d, 2, _next(c, 2)]) || (return false)
    end
    return true
end

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
    return sum(splat(_singular_value_distance), zip(wts1.data, wts2.data)) / length(wts1)
end

function Base.show(io::IO, ::MIME"text/plain", wts::SUWeight)
    println(io, typeof(wts))
    for idx in CartesianIndices(wts.data)
        print(io, Tuple(idx), ": ")
        println(space(wts.data[idx]))
        for (k, b) in blocks(wts.data[idx])
            println(io, k, " = ", diag(b))
        end
    end
    return nothing
end

#= Rotation of SUWeight. Example: 3 x 3 network

- Original
    ```
            |         |         |
            y₁₁       y₁₂       y₁₃
            |         |         |
    ..x₁₃...┼---x₁₁---┼---x₁₂---┼---x₁₃---
            |         |         |           2
            y₂₁       y₂₂       y₂₃         |
            |         |         |           y
    ..x₂₃...┼---x₂₁---┼---x₂₂---┼---x₂₃---  |
            |         |         |           1
            y₃₁       y₃₂       y₃₃
            |         |         |           1 -- x -- 2
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
            |         |         |                   2
            x₁₂       x₂₂       x₃₂                 |
            |         |         |                   x
    --y₁₂---┼---y₂₂---┼---y₃₂---┼...y₁₂...          |
            |         |         |                   1
            x₁₁       x₂₁       x₃₁
            |         |         |           2 -- y -- 1
    --y₁₁---┼---y₂₁---┼---y₃₁---┼...y₁₁...
            :         :         :
            x₁₃       x₂₃       x₃₃
            :         :         :
    ```
    - x/y-weights are exchanged.
    - need to further transpose x-weights.
    - need to further move 1st column of x-weights to the last column.

- `rotr90`:
    ```
            :         :         :
            x₃₃       x₂₃       x₁₃
            :         :         :
    ..y₁₁...┼---y₃₁---┼---y₂₁---┼---y₁₁---
            |         |         |
            x₃₁       x₂₁       x₁₁         1 -- y -- 2
            |         |         |
    ..y₁₂...┼---y₃₂---┼---y₂₂---┼---y₁₂---  1
            |         |         |           |
            x₃₂       x₂₂       x₁₂         x
            |         |         |           |
    ..y₁₃...┼---y₃₃---┼---y₂₃---┼---y₁₃---  2
            |         |         |
            x₃₃       x₂₃       x₁₃
            |         |         |
    ```
    - x/y-weights are exchanged.
    - need to further transpose y-weights.
    - need to further move last row of y-weights to the 1st row.

- `rot180`:
    ```
            :         :         :
            y₁₃       y₁₂       y₁₁
            :         :         :
    --x₃₃---┼---x₃₂---┼---x₃₁---┼...x₃₃...
            |         |         |
            y₃₃       y₃₂       y₃₁        2 -- x -- 1
            |         |         |
    --x₂₃---┼---x₂₂---┼---x₂₁---┼...x₂₃...          1
            |         |         |                   |
            y₂₃       y₂₂       y₂₁                 y
            |         |         |                   |
    --x₁₃---┼---x₁₂---┼---x₁₁---┼...x₁₃...          2
            |         |         |
            y₁₃       y₁₂       y₁₁
            |         |         |
    ```
    - need to transpose all weights.
    - need to move 1st column of x-weights to the last column.
    - need to move last row of y-weights to the 1st row.
=#

function _rotl90_wts_x(wts_x::AbstractMatrix{<:PEPSWeight})
    wts_y = rotl90(wts_x)
    return wts_y
end
function _rotr90_wts_x(wts_x::AbstractMatrix{<:PEPSWeight})
    wts_y = circshift(rotr90(wts_x), (1, 0))
    for (i, wt) in enumerate(wts_y)
        wts_y[i] = DiagonalTensorMap(transpose(wt; copy = true))
    end
    return wts_y
end
function _rot180_wts_x(wts_x::AbstractMatrix{<:PEPSWeight})
    wts_x_ = circshift(rot180(wts_x), (0, -1))
    for (i, wt) in enumerate(wts_x_)
        wts_x_[i] = DiagonalTensorMap(transpose(wt; copy = true))
    end
    return wts_x_
end

function _rotl90_wts_y(wts_y::AbstractMatrix{<:PEPSWeight})
    wts_x = circshift(rotl90(wts_y), (0, -1))
    for (i, wt) in enumerate(wts_x)
        wts_x[i] = DiagonalTensorMap(transpose(wt; copy = true))
    end
    return wts_x
end
function _rotr90_wts_y(wts_y::AbstractMatrix{<:PEPSWeight})
    wts_x = rotr90(wts_y)
    return wts_x
end
function _rot180_wts_y(wts_y::AbstractMatrix{<:PEPSWeight})
    wts_y_ = circshift(rot180(wts_y), (1, 0))
    for (i, wt) in enumerate(wts_y_)
        wts_y_[i] = DiagonalTensorMap(transpose(wt; copy = true))
    end
    return wts_y_
end

function Base.rotl90(wts::SUWeight)
    wts_y = _rotl90_wts_x(wts.data[1, :, :])
    wts_x = _rotl90_wts_y(wts.data[2, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rotr90(wts::SUWeight)
    wts_y = _rotr90_wts_x(wts.data[1, :, :])
    wts_x = _rotr90_wts_y(wts.data[2, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rot180(wts::SUWeight)
    wts_x = _rot180_wts_x(wts.data[1, :, :])
    wts_y = _rot180_wts_y(wts.data[2, :, :])
    return SUWeight(wts_x, wts_y)
end

"""
    CTMRGEnv(wts::SUWeight)

Construct a CTMRG environment with a trivial environment space
(bond dimension χ = 1) from SUWeight `wts`,
which has the same real scalartype as ``wts`.
"""
function CTMRGEnv(wts::SUWeight)
    _, Nr, Nc = size(wts)
    elt = scalartype(wts)
    V_env = oneunit(spacetype(wts))
    edges = map(Iterators.product(1:4, 1:Nr, 1:Nc)) do (d, r, c)
        wt_idx = if d == NORTH
            CartesianIndex(2, r + 1, c)
        elseif d == EAST
            CartesianIndex(1, r, c - 1)
        elseif d == SOUTH
            CartesianIndex(2, r, c)
        else # WEST
            CartesianIndex(1, r, c)
        end
        wt = if d in (NORTH, EAST)
            twist!(repartition(wts[wt_idx], 2, 0; copy = true), 1)
        else
            permute(wts[wt_idx], ((2, 1), ()); copy = true)
        end
        # attach identity on environment space
        return insertleftunit(insertleftunit(wt), 1)
    end
    corners = map(CartesianIndices(edges)) do idx
        return TensorKit.id(elt, V_env)
    end
    return CTMRGEnv(corners, edges)
end

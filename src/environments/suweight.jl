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
    n_mat = length(wts_mats)
    Nr, Nc = size(wts_mats[1])
    @assert all((Nr, Nc) == size(wts_mat) for wts_mat in wts_mats)
    weights = map(Iterators.product(1:n_mat, 1:Nr, 1:Nc)) do (d, r, c)
        return wts_mats[d][r, c]
    end
    return SUWeight(weights)
end

"""
    SUWeight(Nspaces::M, [Espaces::M]; random::Bool=false) where {M<:AbstractMatrix{<:ElementarySpace}}

Create an `SUWeight` by specifying the vertical (north) or horizontal (east) virtual bond spaces.
When `random = false`, a trivial `SUWeight` is returned.
Otherwise each weight is randomly generated, and normalized with its `Inf`-norm.
"""
function SUWeight(
        Nspaces::M, Espaces::M = Nspaces; random::Bool = false
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
    @assert size(Nspaces) == size(Espaces)
    Nr, Nc = size(Nspaces)
    weights = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        V = (d == 1 ? Espaces[r, c] : Nspaces[r, c])
        data = random ? sort!(normalize!(rand(reduceddim(V)), Inf)) : ones(reduceddim(V))
        DiagonalTensorMap(data, V)
    end
    return SUWeight(weights)
end

"""
    SUWeight(Nspace::S, Espace::S=Nspace; unitcell::Tuple{Int,Int}=(1, 1), random::Bool=false) where {S<:ElementarySpace}

Create an `SUWeight` by specifying its vertical (north) and horizontal (east) 
as `ElementarySpace`s) and unit cell size.
When `random = false`, a trivial `SUWeight` is returned.
Otherwise each weight is randomly generated, and normalized with its `Inf`-norm.
"""
function SUWeight(
        Nspace::S, Espace::S = Nspace; unitcell::Tuple{Int, Int} = (1, 1), random::Bool = false
    ) where {S <: ElementarySpace}
    return SUWeight(fill(Nspace, unitcell), fill(Espace, unitcell); random)
end

"""
    SUWeight(peps::InfinitePEPS; random::Bool = false)

Create an `SUWeight` for a given InfinitePEPS.
When `random = false`, a trivial `SUWeight` is returned.
Otherwise each weight is randomly generated, and normalized with its `Inf`-norm.
"""
function SUWeight(peps::InfinitePEPS; random::Bool = false)
    Nspaces = map(Base.Fix2(domain, NORTH), unitcell(peps))
    Espaces = map(Base.Fix2(domain, EAST), unitcell(peps))
    return SUWeight(Nspaces, Espaces; random)
end

"""
    SUWeight(pepo::InfinitePEPO; random::Bool = false)

Create an `SUWeight` for a given one-layer InfinitePEPO.
When `random = false`, a trivial `SUWeight` is returned.
Otherwise each weight is randomly generated, and normalized with its `Inf`-norm.
"""
function SUWeight(pepo::InfinitePEPO; random::Bool = false)
    @assert size(pepo, 3) == 1
    Nspaces = map(Base.Fix2(domain, NORTH), @view(unitcell(pepo)[:, :, 1]))
    Espaces = map(Base.Fix2(domain, EAST), @view(unitcell(pepo)[:, :, 1]))
    return SUWeight(Nspaces, Espaces; random)
end

## Shape and size
Base.size(W::SUWeight) = size(W.data)
Base.size(W::SUWeight, i) = size(W.data, i)
Base.length(W::SUWeight) = length(W.data)
Base.eltype(W::SUWeight) = eltype(typeof(W))
Base.eltype(::Type{SUWeight{E}}) where {E} = E
VI.scalartype(::Type{T}) where {T <: SUWeight} = scalartype(eltype(T))

Base.getindex(W::SUWeight, args...) = Base.getindex(W.data, args...)
Base.setindex!(W::SUWeight, args...) = (Base.setindex!(W.data, args...); W)
Base.axes(W::SUWeight, args...) = axes(W.data, args...)
Base.iterate(W::SUWeight, args...) = iterate(W.data, args...)

## spaces
TensorKit.spacetype(w::SUWeight) = spacetype(typeof(w))
TensorKit.spacetype(::Type{T}) where {E, T <: SUWeight{E}} = spacetype(E)
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
        print(io, Tuple(idx), ": ")
        println(space(wts.data[idx]))
        for (k, b) in blocks(wts.data[idx])
            println(io, k, " = ", diag(b))
        end
    end
    return nothing
end

"""
    absorb_weight(t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight, row::Int, col::Int, ax::Int; inv::Bool = false)
    absorb_weight(t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight, row::Int, col::Int, ax::NTuple{N, Int}; inv::Bool = false)

Absorb or remove (in a twist-free way) the square root of environment weight 
on an axis of the PEPS/PEPO tensor `t` known to be at position (`row`, `col`)
in the unit cell of an InfinitePEPS/InfinitePEPO. The involved weights are
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

- `t::Union{PEPSTensor, PEPOTensor}` : PEPSTensor or PEPOTensor to which the weight will be absorbed.
- `weights::SUWeight` : All simple update weights.
- `row::Int` : The row index specifying the position in the tensor network.
- `col::Int` : The column index specifying the position in the tensor network.
- `ax::Int` : The axis into which the weight is absorbed, taking values from 1 to 4, standing for north, east, south, west respectively.

## Keyword arguments

- `inv::Bool=false` : If `true`, the inverse square root of the weight is absorbed.

## Examples

```julia
# Absorb the weight into the north axis of tensor at position (2, 3)
absorb_weight(t, weights, 2, 3, 1)

# Absorb the inverse of (i.e. remove) the weight into the east axis
absorb_weight(t, weights, 2, 3, 2; inv=true)
```
"""
function absorb_weight(
        t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight,
        row::Int, col::Int, ax::Int; inv::Bool = false
    )
    Nr, Nc = size(weights)[2:end]
    nin, nout, ntol = numin(t), numout(t), numind(t)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    @assert 1 <= ax <= nin
    pow = inv ? -1 / 2 : 1 / 2
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
    t_idx = [(n - nout == ax) ? 1 : -n for n in 1:ntol]
    ax′ = ax + nout
    wt_idx = (ax == NORTH || ax == EAST) ? [1, -ax′] : [-ax′, 1]
    # make absorption/removal twist-free
    twistdual!(wt, 1)
    return permute(ncon((t, wt), (t_idx, wt_idx)), (Tuple(1:nout), Tuple((nout + 1):ntol)))
end
function absorb_weight(
        t::Union{PEPSTensor, PEPOTensor}, weights::SUWeight,
        row::Int, col::Int, ax::NTuple{N, Int}; inv::Bool = false
    ) where {N}
    t2 = copy(t)
    for a in ax
        t2 = absorb_weight(t2, weights, row, col, a; inv)
    end
    return t2
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
    wts_y = _rotl90_wts_x(wts[1, :, :])
    wts_x = _rotl90_wts_y(wts[2, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rotr90(wts::SUWeight)
    wts_y = _rotr90_wts_x(wts[1, :, :])
    wts_x = _rotr90_wts_y(wts[2, :, :])
    return SUWeight(wts_x, wts_y)
end
function Base.rot180(wts::SUWeight)
    wts_x = _rot180_wts_x(wts[1, :, :])
    wts_y = _rot180_wts_y(wts[2, :, :])
    return SUWeight(wts_x, wts_y)
end

"""
    CTMRGEnv(wts::SUWeight)

Construct a CTMRG environment with bond dimension χ = 1 from SUWeight `wts`,
with a real scalartype the same as `scalartype(wts)`.
"""
function CTMRGEnv(wts::SUWeight)
    _, Nr, Nc = size(wts)
    elt, S = scalartype(wts), sectortype(wts)
    V_env = Vect[S](one(S) => 1)
    edges = map(Iterators.product(1:4, 1:Nr, 1:Nc)) do (d, r, c)
        wt_idx = if d == NORTH
            CartesianIndex(2, _next(r, Nr), c)
        elseif d == EAST
            CartesianIndex(1, r, _prev(c, Nc))
        elseif d == SOUTH
            CartesianIndex(2, r, c)
        else # WEST
            CartesianIndex(1, r, c)
        end
        # temporarily make wt axis order ([bra], [ket])
        wt = deepcopy(wts[wt_idx])
        if d in (NORTH, EAST)
            wt = transpose(wt)
        end
        # attach identity on environment space
        return @tensor edge[l t b; r] := wt[b; t] * TensorKit.id(elt, V_env)[l; r]
    end
    corners = map(CartesianIndices(edges)) do idx
        return TensorKit.id(elt, V_env)
    end
    return CTMRGEnv(corners, edges)
end

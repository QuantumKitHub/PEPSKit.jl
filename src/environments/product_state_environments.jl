"""
$(TYPEDEF)

Tensor product environment for an infinite square network, containing a 4 x rows x cols
array of tensors, defined for each nearest neighbor bond in the network. 

The product state tensors `p` connect to the network tensors 
`P` at site `[r,c]` in the unit cell as:
```
                    p[1,r-1,c]
                    |
    p[4,r,c-1]------P[r,c]------p[2,r,c+1]
                    |
                    p[3,r+1,c]
```
## Fields

$(TYPEDFIELDS)
"""
struct ProductStateEnv{T}
    "4 x rows x cols array of edge tensors making up a product state environment, where the
    first dimension specifies the spatial direction"
    edges::Array{T, 3}
end

"""
    ProductStateEnv(
        [f=randn, T=ComplexF64], Ds_north::A, Ds_east::A
    ) where {A <: AbstractMatrix{<:ProductSpace}}

Construct a product state environment by specifying matrices of north and east virtual spaces of the
corresponding [`InfiniteSquareNetwork`](@ref). Each matrix entry corresponds to a site in the unit cell.

Each entry of the `Ds_north` and `Ds_east` matrices corresponds to an effective local space
of the network, and can be represented as a `ProductSpace` (e.g.
for the case of a network representing overlaps of PEPSs).
"""
function ProductStateEnv(
        Ds_north::A, Ds_east::A
    ) where {A <: AbstractMatrix{<:ProductSpace}}
    return ProductStateEnv(randn, ComplexF64, N, Ds_north, Ds_east)
end
function ProductStateEnv(
        f, T, Ds_north::A, Ds_east::A
    ) where {A <: AbstractMatrix{<:ProductSpace}}
    Ds_south = _elementwise_dual.(circshift(Ds_north, (-1, 0)))
    Ds_west = _elementwise_dual.(circshift(Ds_east, (0, 1)))
    edges = map(Iterators.product(1:4, axes(Ds_north, 1), axes(Ds_north, 2))) do (dir, r, c)
        msg = if dir == NORTH
            f(T, Ds_north[_next(r, end), c])
        elseif dir == EAST
            f(T, Ds_east[r, _prev(c, end)])
        elseif dir == SOUTH
            f(T, Ds_south[_prev(r, end), c])
        else # WEST
            f(T, Ds_west[r, _next(c, end)])
        end
        return msg
    end
    normalize!.(edges)
    return ProductStateEnv(edges)
end

"""
    ProductStateEnv(
        [f=randn, T=ComplexF64], D_north::P, D_east::P;
        unitcell::Tuple{Int, Int} = (1, 1)
    ) where {P <: ProductSpace}

Construct a product state environment by specifying the north and east virtual spaces of the
corresponding [`InfiniteSquareNetwork`](@ref). The network unit cell can be specified
by the `unitcell` keyword argument.
"""
function ProductStateEnv(
        D_north::P, D_east::P;
        unitcell::Tuple{Int, Int} = (1, 1)
    ) where {P <: ProductSpace}
    return ProductStateEnv(randn, ComplexF64, D_north, D_east; unitcell)
end
function ProductStateEnv(
        f, T, D_north::P, D_east::P;
        unitcell::Tuple{Int, Int} = (1, 1)
    ) where {P <: ProductSpace}
    return ProductStateEnv(f, T, N, fill(D_north, unitcell), fill(D_east, unitcell))
end

"""
    ProductStateEnv([f=ones, T=ComplexF64], network::InfiniteSquareNetwork)

Construct a product state environment by specifying a corresponding [`InfiniteSquareNetwork`](@ref).
"""
function ProductStateEnv(f, T, network::InfiniteSquareNetwork)
    Ds_north = _north_edge_physical_spaces(network)
    Ds_east = _east_edge_physical_spaces(network)
    return ProductStateEnv(f, T, Ds_north, Ds_east)
end
function ProductStateEnv(network::InfiniteSquareNetwork)
    return ProductStateEnv(ones, scalartype(network), network) # TODO: do we want to use a different default function?
end

function ProductStateEnv(state::Union{InfinitePartitionFunction, InfinitePEPS, InfinitePEPO}, args...; kwargs...)
    return ProductStateEnv(InfiniteSquareNetwork(state), args...; kwargs...)
end
function ProductStateEnv(state::Union{InfinitePEPS, InfinitePEPO}, args...; kwargs...)
    return ProductStateEnv(InfiniteSquareNetwork(state), args...; kwargs...)
end
function ProductStateEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS, InfinitePEPO}, args...; kwargs...)
    return ProductStateEnv(f, T, InfiniteSquareNetwork(state), args...; kwargs...)
end

Base.eltype(::Type{ProductStateEnv{T}}) where {T} = T
Base.size(env::ProductStateEnv, args...) = size(env.edges, args...)
Base.getindex(env::ProductStateEnv, args...) = Base.getindex(env.edges, args...)
Base.axes(env::ProductStateEnv, args...) = Base.axes(env.edges, args...)
Base.eachindex(env::ProductStateEnv) = eachindex(IndexCartesian(), env.edges)
VectorInterface.scalartype(::Type{ProductStateEnv{T}}) where {T} = scalartype(T)
TensorKit.spacetype(::Type{ProductStateEnv{T}}) where {T} = spacetype(T)

function eachcoordinate(x::ProductStateEnv)
    return collect(Iterators.product(axes(x, 2), axes(x, 3)))
end
function eachcoordinate(x::ProductStateEnv, dirs)
    return collect(Iterators.product(dirs, axes(x, 2), axes(x, 3)))
end

# conversion to CTMRGEnv
"""
    CTMRGEnv(prod_env::ProductStateEnv)

Construct a CTMRG environment with a trivial virtual space of bond dimension χ = 1 
from the product state environment `prod_env`.
"""
function CTMRGEnv(prod_env::ProductStateEnv)
    edges = map(CartesianIndices(prod_env.edges)) do idx
        return insertleftunit(insertleftunit(prod_env.edges[idx]), 1)
    end
    corners = map(CartesianIndices(edges)) do _
        return TensorKit.id(scalartype(prod_env), oneunit(spacetype(prod_env)))
    end
    return CTMRGEnv(corners, edges)
end

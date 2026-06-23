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
    ProductStateEnv{T}(edges::Array{T, 3}) where {T} = new{T}(edges)
    function ProductStateEnv(edges::Array{T, 3}) where {T}
        foreach(Iterators.product(axes(edges)[2:3]...)) do (d, w)
            codomain(edges[NORTH, d, w]) == _elementwise_dual(codomain(edges[SOUTH, _prev(d, end), w])) ||
                throw(
                SpaceMismatch("North virtual space at site $((d, w)) does not match: $(space(edges[NORTH, d, w])) vs $(space(edges[SOUTH, _prev(d, end), w])).")
            )
            codomain(edges[EAST, d, w]) == _elementwise_dual(codomain(edges[WEST, d, _next(w, end)])) ||
                throw(SpaceMismatch("East virtual space at site $((d, w)) does not match: $(space(edges[EAST, d, w])) vs $(space(edges[WEST, d, _next(w, end)]))."))
        end
        foreach(Iterators.product(axes(edges)...)) do (dir, d, w)
            dim(space(edges[dir, d, w])) > 0 || @warn "no fusion channels for edge ($dir, $d, $w)"
        end
        return new{T}(edges)
    end
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
function ProductStateEnv(Ds_north::A, args...; kwargs...) where {A <: AbstractMatrix{<:VectorSpace}}
    return ProductStateEnv(randn, ComplexF64, Ds_north, args...; kwargs...)
end

"""
    ProductStateEnv([f=randn, T=ComplexF64], network::InfiniteSquareNetwork)

Construct a product state environment by specifying a corresponding [`InfiniteSquareNetwork`](@ref).
"""
function ProductStateEnv(f, T, network::InfiniteSquareNetwork)
    Ds_north = _north_edge_physical_spaces(network)
    Ds_east = _east_edge_physical_spaces(network)
    return ProductStateEnv(f, T, Ds_north, Ds_east)
end
function ProductStateEnv(network::Union{InfiniteSquareNetwork, InfinitePartitionFunction, InfinitePEPS})
    return ProductStateEnv(randn, scalartype(network), network)
end
function ProductStateEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return ProductStateEnv(f, T, InfiniteSquareNetwork(state), args...)
end

Base.eltype(::Type{ProductStateEnv{T}}) where {T} = T
Base.size(env::ProductStateEnv, args...) = size(env.edges, args...)
Base.getindex(env::ProductStateEnv, args...) = Base.getindex(env.edges, args...)
Base.eachindex(index_style, env::ProductStateEnv) = eachindex(index_style, env.edges)
VectorInterface.scalartype(::Type{ProductStateEnv{T}}) where {T} = scalartype(T)
TensorKit.spacetype(::Type{ProductStateEnv{T}}) where {T} = spacetype(T)

# conversion to CTMRGEnv
"""
    CTMRGEnv(prod_env::ProductStateEnv)

Construct a CTMRG environment with a trivial virtual space of bond dimension χ = 1 
from the product state environment `prod_env`.
"""
function CTMRGEnv(prod_env::ProductStateEnv)
    edges = map(eachindex(IndexCartesian(), prod_env)) do idx
        return insertleftunit(insertleftunit(prod_env[idx]), 1)
    end
    corners = map(eachindex(IndexCartesian(), prod_env)) do _
        return TensorKit.id(storagetype(prod_env), oneunit(spacetype(prod_env)))
    end
    return CTMRGEnv(corners, edges)
end

"""
$(TYPEDEF)

Belief propagation (BP) environment for a square lattice norm network, 
containing a 4 x rows x cols array of message tensors, defined for 
each *oriented* nearest neighbor bond in the network. 

The message tensors connect to the network tensors 
`P` at site `[r,c]` in the unit cell as:
```
                    m[1,r-1,c]
                    |
    m[4,r,c-1]------P[r,c]------m[2,r,c+1]
                    |
                    m[3,r+1,c]
```
- `[1,r-1,c]`: message from `P[r-1,c]` to `P[r,c]` (axis order: ket ← bra)
- `[2,r,c+1]`: message from `P[r,c+1]` to `P[r,c]` (axis order: ket ← bra)
- `[3,r+1,c]`: message from `P[r+1,c]` to `P[r,c]` (axis order: bra ← ket)
- `[4,r,c-1]`: message from `P[r,c-1]` to `P[r,c]` (axis order: bra ← ket)

## Fields

$(TYPEDFIELDS)
"""
struct BPEnv{T}
    "4 x rows x cols array of message tensors, where the first dimension specifies the spatial direction"
    messages::Array{T, 3}
end

"""
Construct a message tensor on a certain bond of a network,
with bond space specified by `pspaces`. 
In the 2-layer case, the message tensor will be
```
    ┌--- pspaces[1] (ket layer)
    m
    └--- pspaces[2] (bra layer)
```
"""
function _message_tensor(f, ::Type{T}, pspaces::P) where {T, P <: ProductSpace}
    N = length(pspaces)
    @assert N == 2 "BPEnv is currently only defined for 2-layer InfiniteSquareNetwork."
    V = permute(pspaces ← one(pspaces), ((1,), (2,)))
    return f(T, V)
end

# TODO: enforce positive semi-definiteness when initializing message tensors

"""
    BPEnv(
        [f=randn, T=ComplexF64], Ds_north::A, Ds_east::A
    ) where {A <: AbstractMatrix{<:ProductSpace}}

Construct a BP environment by specifying matrices of north and east virtual spaces of the
corresponding `InfiniteSquareNetwork`. Each matrix entry corresponds to a site in the unit cell.

Each entry of the `Ds_north` and `Ds_east` matrices corresponds to an effective local space
of the network, and can be represented as a `ProductSpace` (e.g.
for the case of a network representing overlaps of PEPSs).
"""
function BPEnv(Ds_north::A, Ds_east::A) where {A <: AbstractMatrix{<:ProductSpace}}
    return BPEnv(randn, ComplexF64, N, Ds_north, Ds_east)
end
function BPEnv(f, T, Ds_north::A, Ds_east::A) where {A <: AbstractMatrix{<:ProductSpace}}
    # no recursive broadcasting?
    Ds_south = _elementwise_dual.(circshift(Ds_north, (-1, 0)))
    Ds_west = _elementwise_dual.(circshift(Ds_east, (0, 1)))
    messages = map(Iterators.product(1:4, axes(Ds_north, 1), axes(Ds_north, 2))) do (dir, r, c)
        if dir == NORTH
            _message_tensor(f, T, Ds_north[_next(r, end), c])
        elseif dir == EAST
            _message_tensor(f, T, Ds_east[r, _prev(c, end)])
        elseif dir == SOUTH
            permute(_message_tensor(f, T, Ds_south[_prev(r, end), c]), ((2,), (1,)))
        else # WEST
            permute(_message_tensor(f, T, Ds_west[r, _next(c, end)]), ((2,), (1,)))
        end
    end
    normalize!.(messages)
    return BPEnv(messages)
end

"""
    BPEnv(
        [f=randn, T=ComplexF64], D_north::P, D_east::P;
        unitcell::Tuple{Int, Int} = (1, 1),
    ) where {P <: ProductSpace}

Construct a BP environment by specifying the north and east virtual spaces of the
corresponding [`InfiniteSquareNetwork`](@ref). The network unit cell can be specified
by the `unitcell` keyword argument.
"""
function BPEnv(
        D_north::P, D_east::P; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {P <: ProductSpace}
    return BPEnv(randn, ComplexF64, fill(D_north, unitcell), fill(D_east, unitcell))
end
function BPEnv(
        f, T, D_north::P, D_east::P; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {P <: ProductSpace}
    return BPEnv(f, T, N, fill(D_north, unitcell), fill(D_east, unitcell))
end

"""
    BPEnv([f=randn, T=ComplexF64], network::InfiniteSquareNetwork)

Construct a BP environment by specifying a corresponding [`InfiniteSquareNetwork`](@ref).
"""
function BPEnv(f, T, network::InfiniteSquareNetwork)
    Ds_north = _north_edge_physical_spaces(network)
    Ds_east = _east_edge_physical_spaces(network)
    return BPEnv(f, T, Ds_north, Ds_east)
end
BPEnv(network::InfiniteSquareNetwork) = BPEnv(randn, scalartype(network), network)

function BPEnv(state::InfinitePartitionFunction, args...)
    return BPEnv(InfiniteSquareNetwork(state), args...)
end
function BPEnv(state::InfinitePEPS, args...)
    bp_env = BPEnv(InfiniteSquareNetwork(state), args...)
    TensorKit.id!.(bp_env.messages)
    return bp_env
end
function BPEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return BPEnv(f, T, InfiniteSquareNetwork(state), args...)
end

Base.eltype(::Type{BPEnv{T}}) where {T} = T
Base.size(env::BPEnv, args...) = size(env.messages, args...)
Base.getindex(env::BPEnv, args...) = Base.getindex(env.messages, args...)
Base.axes(env::BPEnv, args...) = Base.axes(env.messages, args...)
Base.eachindex(env::BPEnv) = eachindex(IndexCartesian(), env.messages)
VectorInterface.scalartype(::Type{BPEnv{T}}) where {T} = scalartype(T)
TensorKit.spacetype(::Type{BPEnv{T}}) where {T} = spacetype(T)

function eachcoordinate(x::BPEnv)
    return collect(Iterators.product(axes(x, 2), axes(x, 3)))
end
function eachcoordinate(x::BPEnv, dirs)
    return collect(Iterators.product(dirs, axes(x, 2), axes(x, 3)))
end

# conversion to CTMRGEnv
"""
    CTMRGEnv(bp_env::BPEnv)

Construct a CTMRG environment with bond dimension χ = 1 
from the belief propagation environment `bp_env`.
"""
function CTMRGEnv(bp_env::BPEnv)
    edges = map(CartesianIndices(bp_env.messages)) do idx
        M = bp_env.messages[idx]
        M = if idx[1] == SOUTH || idx[1] == WEST
            permute(M, ((2, 1), ()))
        else
            repartition(M, numind(M), 0; copy = true)
        end
        return insertleftunit(insertleftunit(M), 1)
    end
    corners = map(CartesianIndices(edges)) do _
        return TensorKit.id(scalartype(bp_env), oneunit(spacetype(bp_env)))
    end
    return CTMRGEnv(corners, edges)
end

# rotation (the behavior is the same as CTMRGEnv edges)

function Base.rotl90(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 3), size(env.messages, 2))
    for dir in 1:4
        dir2 = _prev(dir, 4)
        messages′[dir2, :, :] = rotl90(env.messages[dir, :, :])
    end
    env2 = BPEnv(copy(messages′))
    for idx in eachcoordinate(env2, (EAST, WEST))
        cidx = CartesianIndex(idx)
        env2.messages[cidx] = permute(env2.messages[cidx], ((2,), (1,)))
    end
    return env2
end

function Base.rotr90(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 3), size(env.messages, 2))
    for dir in 1:4
        dir2 = _next(dir, 4)
        messages′[dir2, :, :] = rotr90(env.messages[dir, :, :])
    end
    env2 = BPEnv(copy(messages′))
    for idx in eachcoordinate(env2, (NORTH, SOUTH))
        cidx = CartesianIndex(idx)
        env2.messages[cidx] = permute(env2.messages[cidx], ((2,), (1,)))
    end
    return env2
end

function Base.rot180(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 2), size(env.messages, 3))
    for dir in 1:4
        dir2 = _next(_next(dir, 4), 4)
        messages′[dir2, :, :] = rot180(env.messages[dir, :, :])
    end
    env2 = BPEnv(copy(messages′))
    for cidx in CartesianIndices(env2.messages)
        env2.messages[cidx] = permute(env2.messages[cidx], ((2,), (1,)))
    end
    return env2
end

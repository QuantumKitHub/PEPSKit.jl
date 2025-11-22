"""
$(TYPEDEF)

Belief propagation environment for a square lattice norm network, 
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
- `[1,r-1,c]`: message from `P[r-1,c]` to `P[r,c]`
- `[2,r,c+1]`: message from `P[r,c+1]` to `P[r,c]`
- `[3,r+1,c]`: message from `P[r+1,c]` to `P[r,c]`
- `[4,r,c-1]`: message from `P[r,c-1]` to `P[r,c]`

## Fields

$(TYPEDFIELDS)
"""
struct BPEnv{T}
    "4 x rows x cols array of message tensors, where the first dimension specifies the spatial direction"
    messages::Array{T, 3}
end

function _message_tensor(f, ::Type{T}, pspaces::P) where {T, P <: ProductSpace}
    V = permute(
        pspaces ← one(pspaces),
        (ntuple(identity, length(pspaces) - 1), (length(pspaces),))
    )
    return f(T, V)
end

# TODO: enforce positive semi-definiteness when initializing message tensors

function BPEnv(Ds_north::A, Ds_east::A) where {A <: AbstractMatrix{<:ProductSpace}}
    return BPEnv(randn, ComplexF64, N, Ds_north, Ds_east)
end
function BPEnv(f, T, Ds_north::A, Ds_east::A) where {A <: AbstractMatrix{<:ProductSpace}}
    # no recursive broadcasting?
    Ds_south = _elementwise_dual.(circshift(Ds_north, (-1, 0)))
    Ds_west = _elementwise_dual.(circshift(Ds_east, (0, 1)))

    # do the whole thing
    N = length(first(Ds_north))
    @assert N == 2
    st = spacetype(first(Ds_north))

    T_type = tensormaptype(st, N - 1, 1, T)

    # First index is direction
    messages = Array{T_type}(undef, 4, size(Ds_north)...)
    for I in CartesianIndices(Ds_north)
        r, c = I.I
        messages[NORTH, r, c] = _message_tensor(f, T, Ds_north[_next(r, end), c])
        messages[EAST, r, c] = _message_tensor(f, T, Ds_east[r, _prev(c, end)])
        messages[SOUTH, r, c] = _message_tensor(f, T, Ds_south[_prev(r, end), c])
        messages[WEST, r, c] = _message_tensor(f, T, Ds_west[r, _next(c, end)])
    end
    normalize!.(messages)

    return BPEnv(messages)
end

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

function BPEnv(f, T, network::InfiniteSquareNetwork)
    Ds_north = _north_edge_physical_spaces(network)
    Ds_east = _east_edge_physical_spaces(network)
    return BPEnv(f, T, Ds_north, Ds_east)
end
BPEnv(network::InfiniteSquareNetwork) = BPEnv(randn, scalartype(network), network)

function BPEnv(state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return BPEnv(InfiniteSquareNetwork(state), args...)
end
function BPEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return BPEnv(f, T, InfiniteSquareNetwork(state), args...)
end

Base.getindex(A::BPEnv, args...) = Base.getindex(A.messages, args...)
Base.axes(A::BPEnv, args...) = Base.axes(A.messages, args...)

# rotation (the behavior is the same as CTMRGEnv edges)

function Base.rotl90(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 3), size(env.messages, 2))
    for dir in 1:4
        dir2 = _prev(dir, 4)
        messages′[dir2, :, :] = rotl90(env.messages[dir, :, :])
    end
    return BPEnv(copy(messages′))
end

function Base.rotr90(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 3), size(env.messages, 2))
    for dir in 1:4
        dir2 = _next(dir, 4)
        messages′[dir2, :, :] = rotr90(env.messages[dir, :, :])
    end
    return BPEnv(copy(messages′))
end

function Base.rot180(env::BPEnv{T}) where {T}
    messages′ = Array{T, 3}(undef, 4, size(env.messages, 2), size(env.messages, 3))
    for dir in 1:4
        dir2 = _next(_next(dir, 4), 4)
        messages′[dir2, :, :] = rot180(env.messages[dir, :, :])
    end
    return BPEnv(copy(messages′))
end

struct BPEnv{T}
    "4 x rows x cols array of message tensors, where the first dimension specifies the spatial direction"
    messages::Array{T,3}
end

function _message_tensor(f, ::Type{T}, pspaces::P) where {T,P<:ProductSpaceLike}
    Vp = _to_space(pspaces)
    V = permute(Vp ← one(Vp), (ntuple(identity, length(pspaces) - 1), (length(pspaces),)))
    return f(T, V)
end

function BPEnv(Ds_north::A, Ds_east::A) where {A<:AbstractMatrix{<:ProductSpaceLike}}
    return BPEnv(randn, ComplexF64, N, Ds_north, Ds_east)
end
function BPEnv(f, T, Ds_north::A, Ds_east::A) where {A<:AbstractMatrix{<:ProductSpaceLike}}
    # no recursive broadcasting?
    Ds_south = _elementwise_dual.(circshift(Ds_north, (-1, 0)))
    Ds_west = _elementwise_dual.(circshift(Ds_east, (0, 1)))

    # do the whole thing
    N = length(first(Ds_north))
    @assert N == 2
    st = _spacetype(first(Ds_north))

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
    D_north::P, D_east::P; unitcell::Tuple{Int,Int}=(1, 1)
) where {P<:ProductSpaceLike}
    return BPEnv(randn, ComplexF64, fill(D_north, unitcell), fill(D_east, unitcell))
end
function BPEnv(
    f, T, D_north::P, D_east::P; unitcell::Tuple{Int,Int}=(1, 1)
) where {P<:ProductSpaceLike}
    return BPEnv(f, T, N, fill(D_north, unitcell), fill(D_east, unitcell))
end

function BPEnv(network::InfiniteSquareNetwork)
    Ds_north = _north_env_spaces(network)
    Ds_east = _east_env_spaces(network)
    return BPEnv(randn, scalartype(network), Ds_north, Ds_east)
end
function BPEnv(f, T, network::InfiniteSquareNetwork)
    Ds_north = _north_env_spaces(network)
    Ds_east = _east_env_spaces(network)
    return BPEnv(f, T, Ds_north, Ds_east)
end

function BPEnv(state::Union{InfinitePartitionFunction,InfinitePEPS}, args...)
    return BPEnv(InfiniteSquareNetwork(state), args...)
end
function BPEnv(f, T, state::Union{InfinitePartitionFunction,InfinitePEPS}, args...)
    return BPEnv(f, T, InfiniteSquareNetwork(state), args...)
end

# VectorInterface
# ---------------

VI.scalartype(::Type{BPEnv{T}}) where {T} = scalartype(T)

VI.zerovector(env::BPEnv, ::Type{S}) where {S<:Number} = BPEnv(zerovector.(env.messages, S))
VI.zerovector!(env::BPEnv) = (zerovector!.(env.messages); env)
VI.zerovector!!(env::BPEnv) = zerovector!(env)

VI.scale(env::BPEnv, α::Number) = BPEnv(scale.(env.messages, α))
VI.scale!(env::BPEnv, α::Number) = (scale!.(env.messages, α); env)
VI.scale!(dst::BPEnv, src::BPEnv, α::Number) = (scale!.(dst.messages, src.messages, α); dst)
VI.scale!!(env::BPEnv, α::Number) = scale!(env, α)
VI.scale!!(dst::BPEnv, src::BPEnv, α::Number) = scale!(dst, src, α)

function VI.add(dst::BPEnv, src::BPEnv, α::Number, β::Number)
    return BPEnv(add.(dst.messages, src.messages, α, β))
end
function VI.add!(dst::BPEnv, src::BPEnv, α::Number, β::Number)
    (add!.(dst.messages, src.messages, α, β); dst)
end
VI.add!!(dst::BPEnv, src::BPEnv, α::Number, β::Number) = add!(dst, src, α, β)

VI.inner(env1::BPEnv, env2::BPEnv) = inner(env1.messages, env2.messages)
VI.norm(env::BPEnv) = norm(env.messages)

LinearAlgebra.normalize(env::BPEnv) = scale(env, inv(norm(env)))
LinearAlgebra.normalize!(env::BPEnv) = scale!(env, inv(norm(env)))

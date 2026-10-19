"""
    SplitCTMRGEnv(network, virtual_spaces...; interlayer_space=nothing)
    SplitCTMRGEnv(env::CTMRGEnv; trunc=notrunc())

Corner transfer-matrix environment with separate rank-three ket and bra edge tensors.
The corners and unit-cell coordinates follow [`CTMRGEnv`](@ref).
An additional interlayer bond connects the two factors of each edge.
By default, initialization uses the outer environment space for this bond; conversion from an existing `CTMRGEnv` is exact unless `trunc` is specified.

Use `edge(env, :ket, dir, r, c)` and `edge(env, :bra, dir, r, c)` to access the factors, or `CTMRGEnv(env)` to contract them for existing observable routines.
This environment supports [`SplitCTMRG`](@ref) and its fixed-point gradient.
Raw reverse-mode pullbacks may return structural cotangents; `ChainRulesCore.ProjectTo(env)(Δenv)` materializes a concrete environment for vector operations.
"""
struct SplitCTMRGEnv{C, E}
    corners::Array{C, 3}
    edges_ket::Array{E, 3}
    edges_bra::Array{E, 3}
    # Whether the ket factor carries the codomain environment leg of the joined edge.
    ket_first::NTuple{4, Bool}
    function SplitCTMRGEnv(
            corners::Array{C, 3}, edges_ket::Array{E, 3}, edges_bra::Array{E, 3},
            ket_first::NTuple{4, Bool} = (true, true, false, false),
        ) where {C, E}
        size(corners) == size(edges_ket) == size(edges_bra) ||
            throw(DimensionMismatch("Corner and split-edge arrays must have equal sizes."))
        size(corners, 1) == 4 || throw(DimensionMismatch("Expected four environment directions."))
        return new{C, E}(corners, edges_ket, edges_bra, ket_first)
    end
end

function ChainRulesCore.rrule(
        ::Type{SplitCTMRGEnv}, corners::Array{C, 3}, edges_ket::Array{E, 3}, edges_bra::Array{E, 3},
        ket_first::NTuple{4, Bool},
    ) where {C, E}
    env = SplitCTMRGEnv(corners, edges_ket, edges_bra, ket_first)
    function split_env_pullback(Δenv_)
        Δenv = unthunk(Δenv_)
        Δenv isa AbstractZero && return NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
        return NoTangent(), _split_tangent_field(Δenv, :corners),
            _split_tangent_field(Δenv, :edges_ket), _split_tangent_field(Δenv, :edges_bra), NoTangent()
    end
    return env, split_env_pullback
end
function ChainRulesCore.rrule(
        ::Type{SplitCTMRGEnv}, corners::Array{C, 3}, edges_ket::Array{E, 3}, edges_bra::Array{E, 3},
    ) where {C, E}
    env, pullback = rrule(SplitCTMRGEnv, corners, edges_ket, edges_bra, (true, true, false, false))
    return env, Δ -> Base.front(pullback(Δ))
end
function ChainRulesCore.rrule(::typeof(getproperty), env::SplitCTMRGEnv, name::Symbol)
    result = getfield(env, name)
    name === :ket_first && return result, _ -> (NoTangent(), ZeroTangent(), NoTangent())
    function split_property_pullback(Δ_)
        Δ = unthunk(Δ_)
        Δ isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent()
        field = NamedTuple{(name,)}((Δ,))
        return NoTangent(), Tangent{typeof(env)}(; field..., ket_first = NoTangent()), NoTangent()
    end
    return result, split_property_pullback
end

"""Read a cotangent field, treating omitted fields and Zygote's nothing markers as zero."""
function _split_tangent_field(Δ, name::Symbol)
    Δ = unthunk(Δ)
    (Δ === nothing || Δ isa AbstractZero) && return ZeroTangent()
    fields = Δ isa Tangent ? ChainRulesCore.backing(Δ) : Δ
    return fields isa NamedTuple ? get(fields, name, ZeroTangent()) : getfield(fields, name)
end

"""Project accumulated array cotangents into fresh tensor storage, materializing zero entries once."""
function _project_split_array(Δ, primal::Array{<:AbstractTensorMap, 3})
    Δ = unthunk(Δ)
    output = similar(primal)
    if Δ === nothing || Δ isa AbstractZero
        return map!(zerovector, output, primal)
    end
    axes(Δ) == axes(primal) || throw(DimensionMismatch("Split cotangent array axes must match the primal."))
    for I in eachindex(primal)
        delta = unthunk(Δ[I])
        output[I] = delta === nothing || delta isa AbstractZero ?
            zerovector(primal[I]) : copy(ProjectTo(primal[I])(delta))
    end
    return output
end

"""Materialize an owned split cotangent where differentiation or the fixed-point solver requires a concrete environment."""
function _project_split_tangent(Δ, primal::SplitCTMRGEnv)::typeof(primal)
    return SplitCTMRGEnv(
        _project_split_array(_split_tangent_field(Δ, :corners), getfield(primal, :corners)),
        _project_split_array(_split_tangent_field(Δ, :edges_ket), getfield(primal, :edges_ket)),
        _project_split_array(_split_tangent_field(Δ, :edges_bra), getfield(primal, :edges_bra)),
        getfield(primal, :ket_first),
    )
end

ChainRulesCore.ProjectTo(env::SplitCTMRGEnv) = ProjectTo{SplitCTMRGEnv}(; primal = env)
(project::ProjectTo{SplitCTMRGEnv})(Δ::Union{NamedTuple, SplitCTMRGEnv}) = _project_split_tangent(Δ, project.primal)
(project::ProjectTo{SplitCTMRGEnv})(Δ::Tangent) = _project_split_tangent(Δ, project.primal)
# Resolve ChainRulesCore's identity projection for a tangent of the same primal type.
(project::ProjectTo{SplitCTMRGEnv})(Δ::Tangent{<:SplitCTMRGEnv}) = _project_split_tangent(Δ, project.primal)

"""Factor an edge across its interlayer bond, distributing the singular values symmetrically."""
function _split_edge(E::CTMRGEdgeTensor{T, S, 3}, ket_first::Bool, trunc::TruncationStrategy) where {T, S}
    p = ket_first ? ((1, 2), (4, 3)) : ((1, 3), (4, 2))
    U, s, V, = svd_trunc(permute(E, p); trunc)
    sqs = sdiag_pow(s, 0.5)
    left = U * sqs
    right = permute(sqs * V, ((1, 3), (2,)))
    return ket_first ? (left, right) : (right, left)
end

function SplitCTMRGEnv(env::CTMRGEnv; trunc::TruncationStrategy = notrunc())
    ket_first = (true, true, false, false)
    factors = map(CartesianIndices(env.edges)) do I
        return _split_edge(env.edges[I], ket_first[I[1]], trunc)
    end
    corners = map(C -> one(scalartype(env)) * TensorMap(C), env.corners)
    return SplitCTMRGEnv(corners, map(first, factors), map(last, factors), ket_first)
end

function SplitCTMRGEnv(
        f, T, network::InfiniteSquareNetwork{<:PEPSSandwich}, virtual_spaces...;
        interlayer_space = nothing,
    )
    env = CTMRGEnv(f, T, network, virtual_spaces...)
    ket_first = (true, true, false, false)
    factors = map(CartesianIndices(env.edges)) do I
        V = isnothing(interlayer_space) ? space(env.edges[I], 1) : interlayer_space
        return _split_edge(env.edges[I], ket_first[I[1]], truncspace(V))
    end
    return SplitCTMRGEnv(env.corners, map(first, factors), map(last, factors), ket_first)
end
function SplitCTMRGEnv(f, T, state::InfinitePEPS, virtual_spaces...; kwargs...)
    return SplitCTMRGEnv(f, T, InfiniteSquareNetwork(state), virtual_spaces...; kwargs...)
end
function SplitCTMRGEnv(network::Union{InfinitePEPS, InfiniteSquareNetwork{<:PEPSSandwich}}, virtual_spaces...; kwargs...)
    return SplitCTMRGEnv(randn, scalartype(network), network, virtual_spaces...; kwargs...)
end

Base.size(env::SplitCTMRGEnv, args...) = size(env.corners, args...)
Base.axes(env::SplitCTMRGEnv, args...) = axes(env.corners, args...)
Base.@propagate_inbounds corner(env::SplitCTMRGEnv, I...) = periodic_getindex(env, env.corners, I)
Base.@propagate_inbounds setcorner!(env::SplitCTMRGEnv, v, I...) = periodic_setindex!(env, env.corners, v, I)

"""
    edge(env::SplitCTMRGEnv, layer::Symbol, dir, r, c)

Return the `:ket` or `:bra` factor of an edge with periodic unit-cell indexing.
Omitting `layer` contracts the interlayer bond and returns a conventional CTMRG edge.
"""
Base.@propagate_inbounds function edge(env::SplitCTMRGEnv, layer::Symbol, I...)
    data = layer === :ket ? env.edges_ket : layer === :bra ? env.edges_bra :
        throw(ArgumentError("Unknown split edge layer: $layer"))
    return periodic_getindex(env, data, I)
end
Base.@propagate_inbounds function setedge!(env::SplitCTMRGEnv, v, layer::Symbol, I...)
    data = layer === :ket ? env.edges_ket : layer === :bra ? env.edges_bra :
        throw(ArgumentError("Unknown split edge layer: $layer"))
    return periodic_setindex!(env, data, v, I)
end

"""Contract the interlayer bond of a split edge in the conventional index order."""
function _join_split_edge(K, B, ket_first::Bool)
    if ket_first
        return @tensor E[l k b; r] := K[l k; i] * B[i b; r]
    else
        return @tensor E[l k b; r] := B[l b; i] * K[i k; r]
    end
end
function edge(env::SplitCTMRGEnv, dir::Int, I...)
    return _join_split_edge(edge(env, :ket, dir, I...), edge(env, :bra, dir, I...), env.ket_first[dir])
end

"""
    CTMRGEnv(env::SplitCTMRGEnv)

Contract each interlayer bond to obtain an environment for the existing observable routines.
"""
function CTMRGEnv(env::SplitCTMRGEnv)
    edges = map(CartesianIndices(env.corners)) do I
        return edge(env, Tuple(I)...)
    end
    return CTMRGEnv(copy.(env.corners), edges)
end
Base.convert(::Type{CTMRGEnv}, env::SplitCTMRGEnv) = CTMRGEnv(env)

eachcoordinate(env::SplitCTMRGEnv) = collect(Iterators.product(axes(env, 2), axes(env, 3)))
eachcoordinate(env::SplitCTMRGEnv, dirs) = collect(Iterators.product(dirs, axes(env, 2), axes(env, 3)))
cornertype(::Type{SplitCTMRGEnv{C, E}}) where {C, E} = C
edgetype(::Type{SplitCTMRGEnv{C, E}}) where {C, E} = E
cornertype(env::SplitCTMRGEnv) = cornertype(typeof(env))
edgetype(env::SplitCTMRGEnv) = edgetype(typeof(env))
VI.scalartype(::Type{SplitCTMRGEnv{C, E}}) where {C, E} = promote_type(scalartype(C), scalartype(E))
TensorKit.spacetype(::Type{E}) where {E <: SplitCTMRGEnv} = spacetype(cornertype(E))
TensorKit.storagetype(::Type{E}) where {E <: SplitCTMRGEnv} = TensorKit.promote_storagetype(cornertype(E), edgetype(E))

"""Reorder and transform directional tensor arrays without mutating their storage."""
function _transform_split_array(data, arraymap, tensormap, dirs)
    slices = map(1:4) do dir
        source = findfirst(==(dir), dirs)
        slice = tensormap.(arraymap(data[source, :, :]))
        return reshape(slice, 1, size(slice)...)
    end
    return cat(slices...; dims = 1)
end

"""Transform unit-cell coordinates, directions, and tensors together without joining the edges."""
function _transform_split_env(env::SplitCTMRGEnv, arraymap, tensormap, dirs, cornerdirs; reverse_order::Bool = false)
    corners = _transform_split_array(env.corners, arraymap, tensormap, cornerdirs)
    edges_ket = _transform_split_array(env.edges_ket, arraymap, tensormap, dirs)
    edges_bra = _transform_split_array(env.edges_bra, arraymap, tensormap, dirs)
    order = ntuple(i -> xor(env.ket_first[findfirst(==(i), dirs)], reverse_order), 4)
    return SplitCTMRGEnv(corners, edges_ket, edges_bra, order)
end
Base.rotl90(env::SplitCTMRGEnv) = _transform_split_env(env, rotl90, identity, (4, 1, 2, 3), (4, 1, 2, 3))
Base.rotr90(env::SplitCTMRGEnv) = _transform_split_env(env, rotr90, identity, (2, 3, 4, 1), (2, 3, 4, 1))
Base.rot180(env::SplitCTMRGEnv) = rotl90(rotl90(env))

"""Reflect and conjugate an environment tensor, preserving nondual outer environment spaces."""
_reflect_split_tensor(C::CTMRGCornerTensor) = copy(C')
_reflect_split_tensor(E::CTMRGEdgeTensor{<:Any, <:Any, 2}) = copy(permute(E', ((1, 3), (2,))))

"""Reflect and conjugate a split environment across the diagonal or the vertical axis."""
function _reflect_split_env(env::SplitCTMRGEnv, diagonal::Bool)
    arraymap = diagonal ? permutedims : A -> reverse(A; dims = 2)
    dirs = diagonal ? (4, 3, 2, 1) : (1, 4, 3, 2)
    cornerdirs = diagonal ? (1, 4, 3, 2) : (2, 1, 4, 3)
    return _transform_split_env(env, arraymap, _reflect_split_tensor, dirs, cornerdirs; reverse_order = true)
end

"""Map corresponding tensors while preserving the split-edge orientation."""
function _map_split_env(f, env::SplitCTMRGEnv, others::SplitCTMRGEnv...)
    return SplitCTMRGEnv(
        map(f, env.corners, map(e -> e.corners, others)...),
        map(f, env.edges_ket, map(e -> e.edges_ket, others)...),
        map(f, env.edges_bra, map(e -> e.edges_bra, others)...), env.ket_first,
    )
end

"""Apply an in-place operation to corresponding split-environment tensors."""
function _foreach_split_env!(f, env::SplitCTMRGEnv, others::SplitCTMRGEnv...)
    foreach(f, env.corners, map(e -> e.corners, others)...)
    foreach(f, env.edges_ket, map(e -> e.edges_ket, others)...)
    foreach(f, env.edges_bra, map(e -> e.edges_bra, others)...)
    return env
end

function update!(env::SplitCTMRGEnv, other::SplitCTMRGEnv)
    env.ket_first == other.ket_first || throw(ArgumentError("Split-edge orientations must match."))
    env.corners .= other.corners
    env.edges_ket .= other.edges_ket
    env.edges_bra .= other.edges_bra
    return env
end

Base.:+(a::SplitCTMRGEnv, b::SplitCTMRGEnv) = VI.add(a, b, 1, 1)
Base.:-(a::SplitCTMRGEnv, b::SplitCTMRGEnv) = VI.add(a, b, -1, 1)
Base.:*(α::Number, env::SplitCTMRGEnv) = VI.scale(env, α)
Base.similar(env::SplitCTMRGEnv) = _map_split_env(similar, env)
Base.real(env::SplitCTMRGEnv) = _map_split_env(real, env)
Base.complex(env::SplitCTMRGEnv) = _map_split_env(complex, env)
function Base.isapprox(a::SplitCTMRGEnv, b::SplitCTMRGEnv; kwargs...)
    return a.ket_first == b.ket_first &&
        isapprox(a.corners, b.corners; kwargs...) &&
        isapprox(a.edges_ket, b.edges_ket; kwargs...) &&
        isapprox(a.edges_bra, b.edges_bra; kwargs...)
end

VI.zerovector(env::SplitCTMRGEnv, ::Type{T}) where {T <: Number} = _map_split_env(Base.Fix2(zerovector, T), env)
VI.zerovector!(env::SplitCTMRGEnv) = _foreach_split_env!(zerovector!, env)
VI.zerovector!!(env::SplitCTMRGEnv) = zerovector!(env)
VI.scale(env::SplitCTMRGEnv, α::Number) = _map_split_env(Base.Fix2(scale, α), env)
VI.scale!(env::SplitCTMRGEnv, α::Number) = _foreach_split_env!(Base.Fix2(scale!, α), env)
VI.scale!(dst::SplitCTMRGEnv, src::SplitCTMRGEnv, α::Number) = _foreach_split_env!((x, y) -> scale!(x, y, α), dst, src)
VI.scale!!(env::SplitCTMRGEnv, α::Number) = scale!(env, α)
VI.scale!!(dst::SplitCTMRGEnv, src::SplitCTMRGEnv, α::Number) = scale!(dst, src, α)
VI.add(a::SplitCTMRGEnv, b::SplitCTMRGEnv, α::Number, β::Number) = _map_split_env((x, y) -> add(x, y, α, β), a, b)
VI.add!(a::SplitCTMRGEnv, b::SplitCTMRGEnv, α::Number, β::Number) = _foreach_split_env!((x, y) -> add!(x, y, α, β), a, b)
VI.add!!(a::SplitCTMRGEnv, b::SplitCTMRGEnv, α::Number, β::Number) = add!(a, b, α, β)
VI.inner(a::SplitCTMRGEnv, b::SplitCTMRGEnv) = inner((a.corners, a.edges_ket, a.edges_bra), (b.corners, b.edges_ket, b.edges_bra))
VI.norm(env::SplitCTMRGEnv) = norm((env.corners, env.edges_ket, env.edges_bra))

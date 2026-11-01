"""
$(TYPEDEF)

Corner transfer-matrix environment with C4v symmetry containing unit-cell arrays of corner and edge tensors.
The last two indices of the arrays correspond to the row and column indices of the unit
cell, which for C4v CTMRG *must* be 1, whereas the first index corresponds to the direction of the corner or edge tensor.
Since the corners are all identical due to the C4v symmetry, the corners array has a first dimension of size 1.
The directions are labeled in clockwise direction, starting from the north edge.

Given arrays of corners `c` and edges `t`, they connect to the network tensors
`P` at site `(r, c)` in the unit cell as:
```
   c[1,r-1,c-1]---t[1,r-1,c]----c[1,r-1,c+1]
   |              |             |
   t[4,r,c-1]-----P[r,c]--------t[2,r,c+1]
   |              |             |
   c[1,r+1,c-1]---t[3,r+1,c]----c[1,r+1,c+1]
```
Here `P` represents an effective local constituent tensor. This can either be a single
rank-4 tensor, a pair of PEPS tensors, or a stack of PEPS-PEPO-PEPS tensors depending on the
network being contracted.

## Fields

$(TYPEDFIELDS)
"""
struct C4vCTMRGEnv{C, T} <: AbstractCTMRGEnv{C, T}
    "1 x 1 x 1 array of corner tensor, kept 3D for compatibility with generic CTMRG methods"
    corners::Array{C, 3}
    "4 x 1 x 1 array of edge tensors, where the first dimension specifies the spatial direction"
    edges::Array{T, 3}
    function C4vCTMRGEnv{C, T}(corners::Array{C, 3}, edges::Array{T, 3}) where {C, T}
        return new{C, T}(corners, edges)
    end
end
function C4vCTMRGEnv(corners::Array{C, 3}, edges::Array{T, 3}) where {C, T}
    foreach(check_environment_virtualspace, edges)
    return C4vCTMRGEnv{C, T}(corners, edges)
end

"""
    C4vCTMRGEnv(
        [f=randn, T=ComplexF64], Ds::A, chis::B
    ) where {A<:AbstractMatrix{<:ProductSpace}, B<:AbstractMatrix{<:ElementarySpace}}

Construct a C4v symmetric CTMRG environment by specifying 1 x 1 matrices of the north virtual
space of the corresponding partition function and the virtual space of the environment. A
single corner and north edge tensor are generated, from which the remaining edges are
obtained by symmetry. The virtual environment spaces for all directions are the same.

The entry of `Ds` corresponds to an effective local space of the partition function, and
can be represented as an `ElementarySpace` (e.g. for the case of a partition function
defined in terms of local rank-4 tensors) or a `ProductSpace` (e.g. for the case of a
network representing overlaps of PEPSs and PEPOs).
"""
function C4vCTMRGEnv(
        f, T, Ds::A, chis::B,
    ) where {
        A <: AbstractMatrix{<:ProductSpace}, B <: AbstractMatrix{<:ElementarySpace},
    }
    # check all of the sizes
    size(Ds) == size(chis) || throw(ArgumentError("Input spaces should have equal sizes."))
    length(Ds) == 1 || throw(ArgumentError("C4v CTMRG only supports trivial unit cells."))

    # All corners are identical, and the remaining edges follow from the north edge
    D, χ = only(Ds), only(chis)
    corner = _corner_tensor(f, T, χ, χ)
    edge = _edge_tensor(f, T, χ, D, χ)
    return C4vCTMRGEnv(corner / norm(corner), edge / norm(edge))
end
function C4vCTMRGEnv(D::P, args...; kwargs...) where {P <: Union{Matrix{VectorSpace}, VectorSpace}}
    return C4vCTMRGEnv(randn, ComplexF64, D, args...; kwargs...)
end

"""
    C4vCTMRGEnv(
        [f=randn, T=ComplexF64], D::P, chi::S;
        unitcell::Tuple{Int,Int}=(1, 1),
    ) where {P<:VectorSpace,S<:ElementarySpace}

Construct a C4v-symmetric CTMRG environment by specifying the virtual space of the
corresponding [`InfiniteSquareNetwork`](@ref) and the virtual
space of the environment. The `unitcell` keyword argument must be `(1, 1)`. The virtual
environment spaces for all directions are the same.

The environment virtual spaces for each site correspond to virtual space of the
corresponding edge tensor for each direction.
"""
function C4vCTMRGEnv(
        f, T,
        D::S, chi::ElementarySpace; unitcell::Tuple{Int, Int} = (1, 1),
    ) where {S <: VectorSpace}
    return C4vCTMRGEnv(
        f, T,
        first(_fill_edge_physical_spaces(D; unitcell)),
        first(_fill_environment_virtual_spaces(chi; unitcell)),
    )
end

"""
    C4vCTMRGEnv(
        [f=randn, T=ComplexF64], network::InfiniteSquareNetwork, chis::A
    ) where {A<:Union{AbstractMatrix{<:ElementarySpace}, ElementarySpace}}

Construct a C4v-symmetric CTMRG environment by specifying a corresponding
[`InfiniteSquareNetwork`](@ref), and the virtual space of the environment.
The virtual space can either be specified as a 1 x 1 matrix of `ElementarySpace`s,
or as an individual `ElementarySpace`. The virtual spaces for all directions are the same.

The environment virtual spaces for each site correspond to the virtual space
of the corresponding edge tensor for each direction.
"""
function C4vCTMRGEnv(f, T, network::InfiniteSquareNetwork, chis)
    Ds = _north_edge_physical_spaces(network)
    chis = first(_fill_environment_virtual_spaces(chis; unitcell = size(network)))
    return C4vCTMRGEnv(f, similarstoragetype(storagetype(network), eltype(T)), Ds, chis)
end
function C4vCTMRGEnv(network::InfiniteSquareNetwork{O}, chis) where {O}
    return C4vCTMRGEnv(randn, storagetype(O), network, chis)
end
function C4vCTMRGEnv(network::Union{<:InfinitePartitionFunction, <:InfinitePEPS}, chis)
    return C4vCTMRGEnv(randn, storagetype(network), network, chis)
end

# allow constructing environments for implicitly defined contractible networks
function C4vCTMRGEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return C4vCTMRGEnv(f, similarstoragetype(eltype(state), eltype(T)), InfiniteSquareNetwork(state), args...)
end

# copy-like constructor
C4vCTMRGEnv(env::C4vCTMRGEnv) = C4vCTMRGEnv(env.corners, env.edges)

@non_differentiable C4vCTMRGEnv(state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)

# Custom adjoint for C4vCTMRGEnv constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(
        ::Type{C4vCTMRGEnv}, corners::Array{C, 3}, edges::Array{T, 3}
    ) where {C, T}
    c4vctmrgenv_pullback(ē) = NoTangent(), ē.corners, ē.edges
    return C4vCTMRGEnv(corners, edges), c4vctmrgenv_pullback
end

# Custom adjoint for C4vCTMRGEnv getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::C4vCTMRGEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :corners
        function corner_pullback(Δcorners_)
            Δcorners = unthunk(Δcorners_)
            return NoTangent(), C4vCTMRGEnv(Δcorners, zerovector.(e.edges)), NoTangent()
        end
        return result, corner_pullback
    elseif name === :edges
        function edge_pullback(Δedges_)
            Δedges = unthunk(Δedges_)
            return NoTangent(), C4vCTMRGEnv(zerovector.(e.corners), Δedges), NoTangent()
        end
        return result, edge_pullback
    else # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
end

"""
    corner(env::C4vCTMRGEnv, dir, i, j)

Return the corner tensor of `env` at direction `dir` and unit-cell coordinates
`(i, j)`. Due to the C4v symmetry, all the corners are identical.
The coordinates are wrapped periodically (`mod1`); the direction is
passed through unchanged. Mixed `Int`/`CartesianIndex` arguments are supported via
`Base.to_indices`, e.g. `corner(env, NORTHWEST, CartesianIndex(r, c))`.
"""
Base.@propagate_inbounds corner(env::C4vCTMRGEnv, I...) =
    periodic_getindex(env, env.corners, I)

"""
    edge(env::C4vCTMRGEnv, dir, i, j)

Return the edge tensor of `env` at direction `dir` and unit-cell coordinates
`(i, j)`. The coordinates are wrapped periodically (`mod1`); the direction is
passed through unchanged. Mixed `Int`/`CartesianIndex` arguments are supported via
`Base.to_indices`.
"""
Base.@propagate_inbounds edge(env::C4vCTMRGEnv, I...) =
    periodic_getindex(env, env.edges, I)

"""
    setcorner!(env::C4vCTMRGEnv, value, dir, i, j) -> value

Store `value` as the corner tensor of `env` at direction `dir` and unit-cell
coordinates `(i, j)`, with periodic wrapping on `(i, j)`. Returns `value`.
Note that due to the C4v symmetry, all corners are identical, so this modifies
every corner at once!
"""
Base.@propagate_inbounds setcorner!(env::C4vCTMRGEnv, v, I...) =
    periodic_setindex!(env, env.corners, v, I)

"""
    setedge!(env::C4vCTMRGEnv, value, dir, i, j) -> value

Store `value` as the edge tensor of `env` at direction `dir` and unit-cell
coordinates `(i, j)`, with periodic wrapping on `(i, j)`. Returns `value`.
"""
Base.@propagate_inbounds setedge!(env::C4vCTMRGEnv, v, I...) =
    periodic_setindex!(env, env.edges, v, I)
function eachcoordinate(x::C4vCTMRGEnv)
    return collect(Iterators.product(axes(x, 2), axes(x, 3)))
end
function eachcoordinate(x::C4vCTMRGEnv, dirs)
    return collect(Iterators.product(dirs, axes(x, 2), axes(x, 3)))
end
Base.real(env::C4vCTMRGEnv) = C4vCTMRGEnv(real.(env.corners), real.(env.edges))
Base.complex(env::C4vCTMRGEnv) = C4vCTMRGEnv(complex.(env.corners), complex.(env.edges))

# In-place update of environment
function update!(env::C4vCTMRGEnv{C, T}, env´::C4vCTMRGEnv{C, T}) where {C, T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end
# patch for the case where the corners are of a different type, which happens when running
# variational optimization using an implicit gradient
function update!(env::C4vCTMRGEnv{C, T}, env´::C4vCTMRGEnv{C´, T}) where {C, C´, T}
    env.corners .= _convert_cornertype.(C, env´.corners)
    env.edges .= env´.edges
    return env
end

# Rotate corners & edges counter-clockwise
function Base.rotl90(env::C4vCTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 1, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    corners′[1, :, :] = rotl90(env.corners[1, :, :]) # all corners are identical!
    for dir in 1:4
        dir2 = _prev(dir, 4)
        edges′[dir2, :, :] = rotl90(env.edges[dir, :, :])
    end
    return C4vCTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges clockwise
function Base.rotr90(env::C4vCTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 1, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    corners′[1, :, :] = rotr90(env.corners[1, :, :]) # all corners are identical!
    for dir in 1:4
        dir2 = _next(dir, 4)
        edges′[dir2, :, :] = rotr90(env.edges[dir, :, :])
    end
    return C4vCTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges by 180 degrees
function Base.rot180(env::C4vCTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 1, size(env.corners, 2), size(env.corners, 3))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 2), size(env.edges, 3)))
    corners′[1, :, :] = rot180(env.corners[1, :, :]) # all corners are identical!
    for dir in 1:4
        dir2 = _next(_next(dir, 4), 4)
        edges′[dir2, :, :] = rot180(env.edges[dir, :, :])
    end
    return C4vCTMRGEnv(copy(corners′), copy(edges′))
end

# Functions used for FP differentiation and by KrylovKit.linsolve
function Base.:+(e₁::C4vCTMRGEnv, e₂::C4vCTMRGEnv)
    return C4vCTMRGEnv(e₁.corners + e₂.corners, e₁.edges + e₂.edges)
end

function Base.:-(e₁::C4vCTMRGEnv, e₂::C4vCTMRGEnv)
    return C4vCTMRGEnv(e₁.corners - e₂.corners, e₁.edges - e₂.edges)
end

Base.:*(α::Number, e::C4vCTMRGEnv) = C4vCTMRGEnv(α * e.corners, α * e.edges)
Base.similar(e::C4vCTMRGEnv) = C4vCTMRGEnv(similar(e.corners), similar(e.edges))

# (approximate) equality
function Base.isapprox(env1::C4vCTMRGEnv, env2::C4vCTMRGEnv; kwargs...)
    for (c1, c2) in zip(env1.corners, env2.corners)
        !isapprox(c1, c2; kwargs...) && return false
    end
    for (e1, e2) in zip(env1.edges, env2.edges)
        !isapprox(e1, e2; kwargs...) && return false
    end
    return true
end

# VectorInterface
# ---------------

# Note: the following methods consider the environment tensors as separate components of one
# big vector. In other words, the associated vector space is not the natural one associated
# to the original (physical) system, and addition, scaling, etc. are performed element-wise.
function VI.scalartype(::Type{C4vCTMRGEnv{C, T}}) where {C, T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end
function VI.zerovector(env::C4vCTMRGEnv, ::Type{S}) where {S <: Number}
    _zerovector = Base.Fix2(zerovector, S)
    return C4vCTMRGEnv(map(_zerovector, env.corners), map(_zerovector, env.edges))
end

function VI.scale(env::C4vCTMRGEnv, α::Number)
    _scale = Base.Fix2(scale, α)
    return C4vCTMRGEnv(map(_scale, env.corners), map(_scale, env.edges))
end

function VI.add(env₁::C4vCTMRGEnv, env₂::C4vCTMRGEnv, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return C4vCTMRGEnv(
        map(_add, env₁.corners, env₂.corners), map(_add, env₁.edges, env₂.edges)
    )
end
function VI.add!(env₁::C4vCTMRGEnv, env₂::C4vCTMRGEnv, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, env₁.corners, env₂.corners)
    foreach(_add!, env₁.edges, env₂.edges)
    return env₁
end
VI.add!!(env₁::C4vCTMRGEnv, env₂::C4vCTMRGEnv, α::Number, β::Number) = add!(env₁, env₂, α, β)

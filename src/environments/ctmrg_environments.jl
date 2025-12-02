const CTMRGEdgeTensor{T, S, N} = AbstractTensorMap{T, S, N, 1}
const CTMRGCornerTensor{T, S} = AbstractTensorMap{T, S, 1, 1}

"""
$(TYPEDEF)

Corner transfer-matrix environment containing unit-cell arrays of corner and edge tensors.
The last two indices of the arrays correspond to the row and column indices of the unit
cell, whereas the first index corresponds to the direction of the corner or edge tensor. The
directions are labeled in clockwise direction, starting from the north-west corner and north
edge respectively.

Given arrays of corners `c` and edges `t`, they connect to the network tensors
`P` at site `(r, c)` in the unit cell as:
```
   c[1,r-1,c-1]---t[1,r-1,c]----c[2,r-1,c+1]
   |              |             |
   t[4,r,c-1]-----P[r,c]--------t[2,r,c+1]
   |              |             |
   c[4,r+1,c-1]---t[3,r+1,c]----c[3,r+1,c+1]
```
Here `P` represents an effective local constituent tensor. This can either be a single
rank-4 tensor, a pair of PEPS tensors, or a stack of PEPS-PEPO-PEPS tensors depending on the
network being contracted.

## Fields

$(TYPEDFIELDS)
"""
struct CTMRGEnv{C, T}
    "4 x rows x cols array of corner tensors, where the first dimension specifies the spatial direction"
    corners::Array{C, 3}
    "4 x rows x cols array of edge tensors, where the first dimension specifies the spatial direction"
    edges::Array{T, 3}
    function CTMRGEnv{C, T}(corners::Array{C, 3}, edges::Array{T, 3}) where {C, T}
        return new{C, T}(corners, edges)
    end
end
function CTMRGEnv(corners::Array{C, 3}, edges::Array{T, 3}) where {C, T}
    foreach(check_environment_virtualspace, edges)
    return CTMRGEnv{C, T}(corners, edges)
end

check_environment_virtualspace(::AbstractZero) = nothing
function check_environment_virtualspace(E::CTMRGEdgeTensor)
    return isdual(space(E, 1)) &&
        throw(ArgumentError("Dual environment virtual spaces are not allowed (for now)."))
end

function _corner_tensor(
        f, ::Type{T}, left_vspace::S, right_vspace::S = left_vspace
    ) where {T, S <: ElementarySpace}
    return f(T, left_vspace ← right_vspace)
end

function _edge_tensor(
        f, ::Type{T}, left_vspace::S, pspaces::P, right_vspace::S = left_vspace
    ) where {T, S <: ElementarySpace, P <: ProductSpace}
    return f(T, left_vspace ⊗ pspaces, right_vspace)
end

"""
    CTMRGEnv(
        [f=randn, T=ComplexF64], Ds_north::A, Ds_east::A, chis_north::B, [chis_east::B], [chis_south::B], [chis_west::B]
    ) where {A<:AbstractMatrix{<:VectorSpace}, B<:AbstractMatrix{<:ElementarySpace}}

Construct a CTMRG environment by specifying matrices of north and east virtual spaces of the
corresponding partition function and the north, east, south and west virtual spaces of the
environment. Each respective matrix entry corresponds to a site in the unit cell. By
default, the virtual environment spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to the north or east virtual space
of the corresponding edge tensor for each direction. Specifically, for a given site
`(r, c)`, `chis_north[r, c]` corresponds to the east space of the north edge tensor,
`chis_east[r, c]` corresponds to the north space of the east edge tensor,
`chis_south[r, c]` corresponds to the east space of the south edge tensor, and
`chis_west[r, c]` corresponds to the north space of the west edge tensor.

Each entry of the `Ds_north` and `Ds_east` matrices corresponds to an effective local space
of the partition function, and can be represented as an `ElementarySpace` (e.g. for the case
of a partition function defined in terms of local rank-4 tensors) or a `ProductSpace` (e.g.
for the case of a network representing overlaps of PEPSs and PEPOs).
"""
function CTMRGEnv(
        f, T, Ds_north::A, Ds_east::A, chis_north::B, chis_east::B = chis_north,
        chis_south::B = chis_north, chis_west::B = chis_north,
    ) where {
        A <: AbstractMatrix{<:ProductSpace}, B <: AbstractMatrix{<:ElementarySpace},
    }
    # check all of the sizes
    size(Ds_north) == size(Ds_east) == size(chis_north) == size(chis_east) ==
        size(chis_south) == size(chis_west) || throw(ArgumentError("Input spaces should have equal sizes."))

    # no recursive broadcasting?
    Ds_south = _elementwise_dual.(circshift(Ds_north, (-1, 0)))
    Ds_west = _elementwise_dual.(circshift(Ds_east, (0, 1)))

    # do the whole thing
    N = length(first(Ds_north))
    st = spacetype(first(Ds_north))
    C_type = tensormaptype(st, 1, 1, T)
    T_type = tensormaptype(st, N + 1, 1, T)

    # First index is direction
    corners = Array{C_type}(undef, 4, size(Ds_north)...)
    edges = Array{T_type}(undef, 4, size(Ds_north)...)

    for I in CartesianIndices(Ds_north)
        r, c = I.I
        edges[NORTH, r, c] = _edge_tensor(
            f, T, chis_north[r, _prev(c, end)], Ds_north[_next(r, end), c], chis_north[r, c]
        )
        edges[EAST, r, c] = _edge_tensor(
            f, T, chis_east[r, c], Ds_east[r, _prev(c, end)], chis_east[_next(r, end), c]
        )
        edges[SOUTH, r, c] = _edge_tensor(
            f, T, chis_south[r, c], Ds_south[_prev(r, end), c], chis_south[r, _prev(c, end)]
        )
        edges[WEST, r, c] = _edge_tensor(
            f, T, chis_west[_next(r, end), c], Ds_west[r, _next(c, end)], chis_west[r, c]
        )

        corners[NORTHWEST, r, c] = _corner_tensor(
            f, T, chis_west[_next(r, end), c], chis_north[r, c]
        )
        corners[NORTHEAST, r, c] = _corner_tensor(
            f, T, chis_north[r, _prev(c, end)], chis_east[_next(r, end), c]
        )
        corners[SOUTHEAST, r, c] = _corner_tensor(
            f, T, chis_east[r, c], chis_south[r, _prev(c, end)]
        )
        corners[SOUTHWEST, r, c] = _corner_tensor(f, T, chis_south[r, c], chis_west[r, c])
    end

    corners[:, :, :] ./= norm.(corners[:, :, :])
    edges[:, :, :] ./= norm.(edges[:, :, :])
    return CTMRGEnv(corners, edges)
end
function CTMRGEnv(D_north::P, args...; kwargs...) where {P <: Union{Matrix{VectorSpace}, VectorSpace}}
    return CTMRGEnv(randn, ComplexF64, D_north, args...; kwargs...)
end

# expand physical edge spaces to unit cell size
function _fill_edge_physical_spaces(
        D_north::S, D_east::S = D_north; unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: VectorSpace}
    return fill(ProductSpace(D_north), unitcell), fill(ProductSpace(D_east), unitcell)
end

# expand virtual environment spaces to unit cell size
function _fill_environment_virtual_spaces(
        chis_north::S, chis_east::S = chis_north, chis_south::S = chis_north, chis_west::S = chis_north;
        unitcell::Tuple{Int, Int} = (1, 1)
    ) where {S <: ElementarySpace}
    return fill(chis_north, unitcell), fill(chis_east, unitcell), fill(chis_south, unitcell), fill(chis_west, unitcell)
end
function _fill_environment_virtual_spaces(
        chis_north::M, chis_east::M = chis_north, chis_south::M = chis_north, chis_west::M = chis_north;
        unitcell::Tuple{Int, Int} = (1, 1)
    ) where {M <: AbstractMatrix{<:ElementarySpace}}
    @assert size(chis_north) == size(chis_east) == size(chis_south) == size(chis_west) == unitcell "Incompatible size"
    return chis_north, chis_east, chis_south, chis_west
end

"""
    CTMRGEnv(
        [f=randn, T=ComplexF64], D_north::P, D_east::P, chi_north::S, [chi_east::S], [chi_south::S], [chi_west::S];
        unitcell::Tuple{Int,Int}=(1, 1),
    ) where {P<:VectorSpace,S<:ElementarySpace}

Construct a CTMRG environment by specifying the north and east virtual spaces of the
corresponding [`InfiniteSquareNetwork`](@ref) and the north, east, south and west virtual
spaces of the environment. The network unit cell can be specified by the `unitcell` keyword
argument. By default, the virtual environment spaces for all directions are taken to be the
same.

The environment virtual spaces for each site correspond to virtual space of the
corresponding edge tensor for each direction.
"""
function CTMRGEnv(
        f, T,
        D_north::S, D_east::S, virtual_spaces...; unitcell::Tuple{Int, Int} = (1, 1),
    ) where {S <: VectorSpace}
    return CTMRGEnv(
        f, T,
        _fill_edge_physical_spaces(D_north, D_east; unitcell)...,
        _fill_environment_virtual_spaces(virtual_spaces...; unitcell)...,
    )
end

# get edge physical spaces from network
function _north_edge_physical_spaces(network::InfiniteSquareNetwork)
    return map(ProductSpace ∘ _elementwise_dual ∘ north_virtualspace, unitcell(network))
end
function _east_edge_physical_spaces(network::InfiniteSquareNetwork)
    return map(ProductSpace ∘ _elementwise_dual ∘ east_virtualspace, unitcell(network))
end

"""
    CTMRGEnv(
        [f=randn, T=ComplexF64], network::InfiniteSquareNetwork, chis_north::A, [chis_east::A], [chis_south::A], [chis_west::A]
    ) where {A<:Union{AbstractMatrix{<:ElementarySpace}, ElementarySpace}}

Construct a CTMRG environment by specifying a corresponding
[`InfiniteSquareNetwork`](@ref), and the north, east, south and west virtual spaces of the
environment. The virtual spaces can either be specified as matrices of `ElementarySpace`s,
or as individual `ElementarySpace`s which are then filled to match the size of the unit
cell. Each respective matrix entry corresponds to a site in the unit cell. By default, the
virtual spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to the north or east virtual space
of the corresponding edge tensor for each direction. Specifically, for a given site
`(r, c)`, `chis_north[r, c]` corresponds to the east space of the north edge tensor,
`chis_east[r, c]` corresponds to the north space of the east edge tensor,
`chis_south[r, c]` corresponds to the east space of the south edge tensor, and
`chis_west[r, c]` corresponds to the north space of the west edge tensor.
"""
function CTMRGEnv(f, T, network::InfiniteSquareNetwork, virtual_spaces...)
    Ds_north = _north_edge_physical_spaces(network)
    Ds_east = _east_edge_physical_spaces(network)
    virtual_spaces = _fill_environment_virtual_spaces(virtual_spaces...; unitcell = size(network))
    return CTMRGEnv(f, T, Ds_north, Ds_east, virtual_spaces...)
end
function CTMRGEnv(network::Union{InfiniteSquareNetwork, InfinitePartitionFunction, InfinitePEPS}, virtual_spaces...)
    return CTMRGEnv(randn, scalartype(network), network, virtual_spaces...)
end

# allow constructing environments for implicitly defined contractible networks
function CTMRGEnv(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)
    return CTMRGEnv(f, T, InfiniteSquareNetwork(state), args...)
end

@non_differentiable CTMRGEnv(state::Union{InfinitePartitionFunction, InfinitePEPS}, args...)

# Custom adjoint for CTMRGEnv constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(::Type{CTMRGEnv}, corners, edges)
    ctmrgenv_pullback(ē) = NoTangent(), ē.corners, ē.edges
    return CTMRGEnv(corners, edges), ctmrgenv_pullback
end

# Custom adjoint for CTMRGEnv getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::CTMRGEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :corners
        function corner_pullback(Δcorners_)
            Δcorners = unthunk(Δcorners_)
            return NoTangent(), CTMRGEnv(Δcorners, zerovector.(e.edges)), NoTangent()
        end
        return result, corner_pullback
    elseif name === :edges
        function edge_pullback(Δedges_)
            Δedges = unthunk(Δedges_)
            return NoTangent(), CTMRGEnv(zerovector.(e.corners), Δedges), NoTangent()
        end
        return result, edge_pullback
    else
        # this should never happen because already errored in forwards pass
        throw(ArgumentError("No rrule for getproperty of $name"))
    end
end

Base.size(env::CTMRGEnv, args...) = size(env.corners, args...)
Base.axes(x::CTMRGEnv, args...) = axes(x.corners, args...)
function eachcoordinate(x::CTMRGEnv)
    return collect(Iterators.product(axes(x, 2), axes(x, 3)))
end
function eachcoordinate(x::CTMRGEnv, dirs)
    return collect(Iterators.product(dirs, axes(x, 2), axes(x, 3)))
end
Base.real(env::CTMRGEnv) = CTMRGEnv(real.(env.corners), real.(env.edges))
Base.complex(env::CTMRGEnv) = CTMRGEnv(complex.(env.corners), complex.(env.edges))

cornertype(env::CTMRGEnv) = cornertype(typeof(env))
cornertype(::Type{CTMRGEnv{C, E}}) where {C, E} = C
edgetype(env::CTMRGEnv) = edgetype(typeof(env))
edgetype(::Type{CTMRGEnv{C, E}}) where {C, E} = E

TensorKit.spacetype(::Type{E}) where {E <: CTMRGEnv} = spacetype(cornertype(E))

# In-place update of environment
function update!(env::CTMRGEnv{C, T}, env´::CTMRGEnv{C, T}) where {C, T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end

# Rotate corners & edges counter-clockwise
function Base.rotl90(env::CTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 4, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    for dir in 1:4
        dir2 = _prev(dir, 4)
        corners′[dir2, :, :] = rotl90(env.corners[dir, :, :])
        edges′[dir2, :, :] = rotl90(env.edges[dir, :, :])
    end
    return CTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges clockwise
function Base.rotr90(env::CTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 4, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    for dir in 1:4
        dir2 = _next(dir, 4)
        corners′[dir2, :, :] = rotr90(env.corners[dir, :, :])
        edges′[dir2, :, :] = rotr90(env.edges[dir, :, :])
    end
    return CTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges by 180 degrees
function Base.rot180(env::CTMRGEnv{C, T}) where {C, T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C, 3}(undef, 4, size(env.corners, 2), size(env.corners, 3))
    )
    edges′ = Zygote.Buffer(Array{T, 3}(undef, 4, size(env.edges, 2), size(env.edges, 3)))
    for dir in 1:4
        dir2 = _next(_next(dir, 4), 4)
        corners′[dir2, :, :] = rot180(env.corners[dir, :, :])
        edges′[dir2, :, :] = rot180(env.edges[dir, :, :])
    end
    return CTMRGEnv(copy(corners′), copy(edges′))
end

# Functions used for FP differentiation and by KrylovKit.linsolve
function Base.:+(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return CTMRGEnv(e₁.corners + e₂.corners, e₁.edges + e₂.edges)
end

function Base.:-(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return CTMRGEnv(e₁.corners - e₂.corners, e₁.edges - e₂.edges)
end

Base.:*(α::Number, e::CTMRGEnv) = CTMRGEnv(α * e.corners, α * e.edges)
Base.similar(e::CTMRGEnv) = CTMRGEnv(similar(e.corners), similar(e.edges))

# VectorInterface
# ---------------

# Note: the following methods consider the environment tensors as separate components of one
# big vector. In other words, the associated vector space is not the natural one associated
# to the original (physical) system, and addition, scaling, etc. are performed element-wise.

function VI.scalartype(::Type{CTMRGEnv{C, T}}) where {C, T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end

function VI.zerovector(env::CTMRGEnv, ::Type{S}) where {S <: Number}
    _zerovector = Base.Fix2(zerovector, S)
    return CTMRGEnv(map(_zerovector, env.corners), map(_zerovector, env.edges))
end
function VI.zerovector!(env::CTMRGEnv)
    foreach(zerovector!, env.corners)
    foreach(zerovector!, env.edges)
    return env
end
VI.zerovector!!(env::CTMRGEnv) = zerovector!(env)

function VI.scale(env::CTMRGEnv, α::Number)
    _scale = Base.Fix2(scale, α)
    return CTMRGEnv(map(_scale, env.corners), map(_scale, env.edges))
end
function VI.scale!(env::CTMRGEnv, α::Number)
    _scale! = Base.Fix2(scale!, α)
    foreach(_scale!, env.corners)
    foreach(_scale!, env.edges)
    return env
end
function VI.scale!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number)
    _scale!(x, y) = scale!(x, y, α)
    foreach(_scale!, env₁.corners, env₂.corners)
    foreach(_scale!, env₁.edges, env₂.edges)
    return env₁
end
VI.scale!!(env::CTMRGEnv, α::Number) = scale!(env, α)
VI.scale!!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number) = scale!(env₁, env₂, α)

function VI.add(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number)
    _add(x, y) = add(x, y, α, β)
    return CTMRGEnv(
        map(_add, env₁.corners, env₂.corners), map(_add, env₁.edges, env₂.edges)
    )
end
function VI.add!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number)
    _add!(x, y) = add!(x, y, α, β)
    foreach(_add!, env₁.corners, env₂.corners)
    foreach(_add!, env₁.edges, env₂.edges)
    return env₁
end
VI.add!!(env₁::CTMRGEnv, env₂::CTMRGEnv, α::Number, β::Number) = add!(env₁, env₂, α, β)

# Exploiting the fact that VectorInterface works for tuples:
function VI.inner(env₁::CTMRGEnv, env₂::CTMRGEnv)
    return inner((env₁.corners, env₁.edges), (env₂.corners, env₂.edges))
end
VI.norm(env::CTMRGEnv) = norm((env.corners, env.edges))

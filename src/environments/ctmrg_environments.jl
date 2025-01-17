"""
    struct CTMRGEnv{C,T}

Corner transfer-matrix environment containing unit-cell arrays of corner and edge tensors.
The last two indices of the arrays correspond to the row and column indices of the unit
cell, whereas the first index corresponds to the direction of the corner or edge tensor. The
directions are labeled in clockwise direction, starting from the north-west corner and north
edge respectively.

Given arrays of corners `c` and edges `t`, they connect to the partition function tensors
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
partition function being contracted.

# Fields
- `corners::Array{C,3}`: Array of corner tensors.
- `edges::Array{T,3}`: Array of edge tensors.
"""
struct CTMRGEnv{C,T}
    corners::Array{C,3}
    edges::Array{T,3}
end

const ElementarySpaceLike = Union{Int,ElementarySpace}
const ProductSpaceLike{N} = Union{NTuple{N,Int},NTuple{N,<:ElementarySpace}}
const SpaceLike = Union{ElementarySpaceLike,ProductSpaceLike}

_spacetype(::Int) = ComplexSpace
_spacetype(::S) where {S<:ElementarySpace} = S
_spacetype(s::ProductSpaceLike) = _spacetype(first(s))

_to_space(χ::Int) = ℂ^χ
_to_space(χ::ElementarySpace) = χ
_to_space(χ::ProductSpaceLike) = _to_space.(χ)

function _corner_tensor(
    f, ::Type{T}, left_vspace::S, right_vspace::S=left_vspace
) where {T,S<:ElementarySpaceLike}
    return TensorMap(f, T, _to_space(left_vspace) ← _to_space(right_vspace))
end

function _edge_tensor(
    f, ::Type{T}, left_vspace::S, pspaces::P, right_vspace::S=left_vspace
) where {T,S<:ElementarySpaceLike,P<:ProductSpaceLike}
    return TensorMap(
        f,
        T,
        prod([_to_space(left_vspace), _to_space.(pspaces)...]),
        _to_space(right_vspace),
    )
end

"""
    CTMRGEnv(
        [f=randn, ComplexF64], Ds_north::A, Ds_east::A, chis_north::B, [chis_east::B], [chis_south::B], [chis_west::B]
    ) where {A<:AbstractMatrix{<:SpaceLike}, B<:AbstractMatrix{<:ElementarySpaceLike}}

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
of the partition function, represented as a tuple of elementary spaces encoding a product
space. This can either contain a single elementary space for the case of a partition
function defined in terms of local rank-4 tensors, or a tuple of elementary spaces
representing a product space for the case of a partition function representing overlaps of
PEPSs and PEPOs.
"""
function CTMRGEnv(
    Ds_north::A,
    Ds_east::A,
    chis_north::B,
    chis_east::B=chis_north,
    chis_south::B=chis_north,
    chis_west::B=chis_north,
) where {A<:AbstractMatrix{<:ProductSpaceLike},B<:AbstractMatrix{<:ElementarySpaceLike}}
    return CTMRGEnv(
        randn,
        ComplexF64,
        N,
        Ds_north,
        Ds_east,
        chis_north,
        chis_east,
        chis_south,
        chis_west,
    )
end
function CTMRGEnv(
    f,
    T,
    Ds_north::A,
    Ds_east::A,
    chis_north::B,
    chis_east::B=chis_north,
    chis_south::B=chis_north,
    chis_west::B=chis_north,
) where {A<:AbstractMatrix{<:ProductSpaceLike},B<:AbstractMatrix{<:ElementarySpaceLike}}
    # no recursive broadcasting?
    Ds_south = (x -> adjoint.(x)).(circshift(Ds_north, (-1, 0)))
    Ds_west = (x -> adjoint.(x)).(circshift(Ds_east, (0, 1)))

    # do the whole thing
    N = length(first(Ds_north))
    st = _spacetype(first(Ds_north))
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

"""
    CTMRGEnv(
        [f=randn, ComplexF64], D_north::P, D_east::P, chi_north::S, [chi_east::S], [chi_south::S], [chi_west::S]; unitcell::Tuple{Int,Int}=(1, 1),
    ) where {P<:ProductSpaceLike,S<:ElementarySpaceLike}

Construct a CTMRG environment by specifying the north and east virtual spaces of the
corresponding [`InfinitePEPS`](@ref) and the north, east, south and west virtual spaces of
the environment. The PEPS unit cell can be specified by the `unitcell` keyword argument. By
default, the virtual environment spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to virtual space of the
corresponding edge tensor for each direction.
"""
function CTMRGEnv(
    D_north::P,
    D_east::P,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north;
    unitcell::Tuple{Int,Int}=(1, 1),
) where {P<:ProductSpaceLike,S<:Union{Int,ElementarySpace}}
    return CTMRGEnv(
        randn,
        ComplexF64,
        fill(D_north, unitcell),
        fill(D_east, unitcell),
        fill(chi_north, unitcell),
        fill(chi_east, unitcell),
        fill(chi_south, unitcell),
        fill(chi_west, unitcell),
    )
end
function CTMRGEnv(
    f,
    T,
    D_north::P,
    D_east::P,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north;
    unitcell::Tuple{Int,Int}=(1, 1),
) where {P<:ProductSpaceLike,S<:Union{Int,ElementarySpace}}
    return CTMRGEnv(
        f,
        T,
        N,
        fill(D_north, unitcell),
        fill(D_east, unitcell),
        fill(chi_north, unitcell),
        fill(chi_east, unitcell),
        fill(chi_south, unitcell),
        fill(chi_west, unitcell),
    )
end

"""
    CTMRGEnv(
        [f=randn, T=ComplexF64], peps::InfinitePEPS, chis_north::A, [chis_east::A], [chis_south::A], [chis_west::A]
    ) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Construct a CTMRG environment by specifying a corresponding [`InfinitePEPS`](@ref), and the
north, east, south and west virtual spaces of the environment as matrices. Each respective
matrix entry corresponds to a site in the unit cell. By default, the virtual spaces for all
directions are taken to be the same.

The environment virtual spaces for each site correspond to the north or east virtual space
of the corresponding edge tensor for each direction. Specifically, for a given site
`(r, c)`, `chis_north[r, c]` corresponds to the east space of the north edge tensor,
`chis_east[r, c]` corresponds to the north space of the east edge tensor,
`chis_south[r, c]` corresponds to the east space of the south edge tensor, and
`chis_west[r, c]` corresponds to the north space of the west edge tensor.
"""
function CTMRGEnv(
    peps::InfinitePEPS,
    chis_north::A,
    chis_east::A=chis_north,
    chis_south::A=chis_north,
    chis_west::A=chis_north,
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    Ds_north = map(peps.A) do t
        return (adjoint(space(t, 2)), space(t, 2))
    end
    Ds_east = map(peps.A) do t
        return (adjoint(space(t, 3)), space(t, 3))
    end
    return CTMRGEnv(
        randn,
        ComplexF64,
        Ds_north,
        Ds_east,
        _to_space.(chis_north),
        _to_space.(chis_east),
        _to_space.(chis_south),
        _to_space.(chis_west),
    )
end
function CTMRGEnv(
    f,
    T,
    peps::InfinitePEPS,
    chis_north::A,
    chis_east::A=chis_north,
    chis_south::A=chis_north,
    chis_west::A=chis_north,
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    Ds_north = map(peps.A) do t
        return (adjoint(space(t, 2)), space(t, 2))
    end
    Ds_east = map(peps.A) do t
        return (adjoint(space(t, 3)), space(t, 3))
    end
    return CTMRGEnv(
        f,
        T,
        Ds_north,
        Ds_east,
        _to_space.(chis_north),
        _to_space.(chis_east),
        _to_space.(chis_south),
        _to_space.(chis_west),
    )
end

"""
    CTMRGEnv(
        [f=randn, T=ComplexF64], partfunc::InfinitePartitionFunction, chis_north::A, [chis_east::A], [chis_south::A], [chis_west::A]
    ) where {A<:AbstractMatrix{<:Union{Int,ElementarySpace}}}

Construct a CTMRG environment by specifying a corresponding
[`InfinitePartitionFunction`](@ref), and the north, east, south and west virtual spaces of
the environment as matrices. Each respective matrix entry corresponds to a site in the unit
cell. By default, the virtual spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to the north or east virtual space
of the corresponding edge tensor for each direction. Specifically, for a given site
`(r, c)`, `chis_north[r, c]` corresponds to the east space of the north edge tensor,
`chis_east[r, c]` corresponds to the north space of the east edge tensor,
`chis_south[r, c]` corresponds to the east space of the south edge tensor, and
`chis_west[r, c]` corresponds to the north space of the west edge tensor.
"""
function CTMRGEnv(
    partfunc::InfinitePartitionFunction,
    chis_north::A,
    chis_east::A=chis_north,
    chis_south::A=chis_north,
    chis_west::A=chis_north,
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    Ds_north = map(partfunc.A) do t
        return (adjoint(space(t, 3)),)
    end
    Ds_east = map(partfunc.A) do t
        return (adjoint(space(t, 4)),)
    end
    return CTMRGEnv(
        randn,
        ComplexF64,
        Ds_north,
        Ds_east,
        _to_space.(chis_north),
        _to_space.(chis_east),
        _to_space.(chis_south),
        _to_space.(chis_west),
    )
end
function CTMRGEnv(
    f,
    T,
    partfunc::InfinitePartitionFunction,
    chis_north::A,
    chis_east::A=chis_north,
    chis_south::A=chis_north,
    chis_west::A=chis_north,
) where {A<:AbstractMatrix{<:ElementarySpaceLike}}
    Ds_north = map(partfunc.A) do t
        return (adjoint(space(t, 3)),)
    end
    Ds_east = map(partfunc.A) do t
        return (adjoint(space(t, 4)),)
    end
    return CTMRGEnv(
        f,
        T,
        N,
        Ds_north,
        Ds_east,
        _to_space.(chis_north),
        _to_space.(chis_east),
        _to_space.(chis_south),
        _to_space.(chis_west),
    )
end

"""
    CTMRGEnv(
        peps::InfinitePEPS, chi_north::S, [chi_east::S], [chi_south::S], [chi_west::S],
    ) where {S<:Union{Int,ElementarySpace}}

Construct a CTMRG environment by specifying a corresponding [`InfinitePEPS`](@ref), and the
north, east, south and west virtual spaces of the environment. By default, the virtual
spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to virtual space of the
corresponding edge tensor for each direction.
"""
function CTMRGEnv(
    peps::InfinitePEPS,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north,
) where {S<:ElementarySpaceLike}
    return CTMRGEnv(
        peps,
        fill(chi_north, size(peps)),
        fill(chi_east, size(peps)),
        fill(chi_south, size(peps)),
        fill(chi_west, size(peps)),
    )
end
function CTMRGEnv(
    f,
    T,
    peps::InfinitePEPS,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north,
) where {S<:ElementarySpaceLike}
    return CTMRGEnv(
        f,
        T,
        peps,
        fill(chi_north, size(peps)),
        fill(chi_east, size(peps)),
        fill(chi_south, size(peps)),
        fill(chi_west, size(peps)),
    )
end

"""
    CTMRGEnv(
        peps::InfinitePartitionFunction, chi_north::S, [chi_east::S], [chi_south::S], [chi_west::S],
    ) where {S<:ElementarySpaceLike

Construct a CTMRG environment by specifying a corresponding
[`InfinitePartitionFunction`](@ref), and the north, east, south and west virtual spaces of
the environment. By default, the virtual spaces for all directions are taken to be the same.

The environment virtual spaces for each site correspond to virtual space of the
corresponding edge tensor for each direction.
"""
function CTMRGEnv(
    partfunc::InfinitePartitionFunction,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north,
) where {S<:ElementarySpaceLike}
    return CTMRGEnv(
        partfunc,
        fill(chi_north, size(partfunc)),
        fill(chi_east, size(partfunc)),
        fill(chi_south, size(partfunc)),
        fill(chi_west, size(partfunc)),
    )
end
function CTMRGEnv(
    f,
    T,
    partfunc::InfinitePartitionFunction,
    chi_north::S,
    chi_east::S=chi_north,
    chi_south::S=chi_north,
    chi_west::S=chi_north,
) where {S<:ElementarySpaceLike}
    return CTMRGEnv(
        f,
        T,
        partfunc,
        fill(chi_north, size(partfunc)),
        fill(chi_east, size(partfunc)),
        fill(chi_south, size(partfunc)),
        fill(chi_west, size(partfunc)),
    )
end

@non_differentiable CTMRGEnv(peps::InfinitePEPS, args...)
@non_differentiable CTMRGEnv(peps::InfinitePartitionFunction, args...)

"""
    CTMRGEnv([f=randn, T=scalartype(env)], env::CTMRGEnv)

Copy constructor for easily initializing a similar environment, optionally providing a
function `f` for element generation with type `T`.
"""
function CTMRGEnv(env::CTMRGEnv)
    return CTMRTEnv(randn, scalartype(env), env)
end
function CTMRGEnv(f, T, env::CTMRGEnv)
    corners = map(env.corners) do corner
        return TensorMap(f, T, space(corner))
    end
    edges = map(env.edges) do edge
        return TensorMap(f, T, space(edge))
    end
    return CTMRGEnv(corners, edges)
end

# Custom adjoint for CTMRGEnv constructor, needed for fixed-point differentiation
function ChainRulesCore.rrule(::Type{CTMRGEnv}, corners, edges)
    ctmrgenv_pullback(ē) = NoTangent(), ē.corners, ē.edges
    return CTMRGEnv(corners, edges), ctmrgenv_pullback
end

# Custom adjoint for CTMRGEnv getproperty, to avoid creating named tuples in backward pass
function ChainRulesCore.rrule(::typeof(getproperty), e::CTMRGEnv, name::Symbol)
    result = getproperty(e, name)
    if name === :corners
        function corner_pullback(Δcorners)
            return NoTangent(), CTMRGEnv(Δcorners, zerovector.(e.edges)), NoTangent()
        end
        return result, corner_pullback
    elseif name === :edges
        function edge_pullback(Δedges)
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

# In-place update of environment
function update!(env::CTMRGEnv{C,T}, env´::CTMRGEnv{C,T}) where {C,T}
    env.corners .= env´.corners
    env.edges .= env´.edges
    return env
end

# Rotate corners & edges counter-clockwise
function Base.rotl90(env::CTMRGEnv{C,T}) where {C,T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C,3}(undef, 4, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T,3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    for dir in 1:4
        dir2 = _prev(dir, 4)
        corners′[dir2, :, :] = rotl90(env.corners[dir, :, :])
        edges′[dir2, :, :] = rotl90(env.edges[dir, :, :])
    end
    return CTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges clockwise
function Base.rotr90(env::CTMRGEnv{C,T}) where {C,T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C,3}(undef, 4, size(env.corners, 3), size(env.corners, 2))
    )
    edges′ = Zygote.Buffer(Array{T,3}(undef, 4, size(env.edges, 3), size(env.edges, 2)))
    for dir in 1:4
        dir2 = _next(dir, 4)
        corners′[dir2, :, :] = rotr90(env.corners[dir, :, :])
        edges′[dir2, :, :] = rotr90(env.edges[dir, :, :])
    end
    return CTMRGEnv(copy(corners′), copy(edges′))
end

# Rotate corners & edges by 180 degrees
function Base.rot180(env::CTMRGEnv{C,T}) where {C,T}
    # Initialize rotated corners & edges with rotated sizes
    corners′ = Zygote.Buffer(
        Array{C,3}(undef, 4, size(env.corners, 2), size(env.corners, 3))
    )
    edges′ = Zygote.Buffer(Array{T,3}(undef, 4, size(env.edges, 2), size(env.edges, 3)))
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

function LinearAlgebra.mul!(edst::CTMRGEnv, esrc::CTMRGEnv, α::Number)
    edst.corners .= α * esrc.corners
    edst.edges .= α * esrc.edges
    return edst
end

function LinearAlgebra.rmul!(e::CTMRGEnv, α::Number)
    rmul!.(e.corners, α)
    rmul!.(e.edges, α)
    return e
end

function LinearAlgebra.axpy!(α::Number, e₁::CTMRGEnv, e₂::CTMRGEnv)
    axpy!.(α, e₁.corners, e₂.corners)
    axpy!.(α, e₁.edges, e₂.edges)
    return e₂
end

function LinearAlgebra.axpby!(α::Number, e₁::CTMRGEnv, β::Number, e₂::CTMRGEnv)
    e₂.corners .= α * e₁.corners + β * e₂.corners
    e₂.edges .= α * e₁.edges + β * e₂.edges
    return e₂
end

function LinearAlgebra.dot(e₁::CTMRGEnv, e₂::CTMRGEnv)
    return dot(e₁.corners, e₂.corners) + dot(e₁.edges, e₂.edges)
end

# VectorInterface
# ---------------

# Note: the following methods consider the environment tensors as separate components of one
# big vector. In other words, the associated vector space is not the natural one associated
# to the original (physical) system, and addition, scaling, etc. are performed element-wise.

import VectorInterface as VI

function VI.scalartype(::Type{CTMRGEnv{C,T}}) where {C,T}
    S₁ = scalartype(C)
    S₂ = scalartype(T)
    return promote_type(S₁, S₂)
end

function VI.zerovector(env::CTMRGEnv, ::Type{S}) where {S<:Number}
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

# --------------------------------------------------------
# Sparse enlarged corner as building block for environment
# --------------------------------------------------------

"""
$(TYPEDEF)

Enlarged CTMRG corner tensor storage.

## Constructors

    EnlargedCorner(network::InfiniteSquareNetwork, env, coordinates)

Construct an enlarged corner with the correct row and column indices based on the given
`coordinates` which are of the form `(dir, row, col)`.

"""
struct EnlargedCorner{TC, TE, TA}
    C::TC
    E_1::TE
    E_2::TE
    A::TA
    dir::Int
end
function EnlargedCorner(network::InfiniteSquareNetwork, env, coordinates)
    dir, r, c = coordinates
    if dir == NORTHWEST
        return EnlargedCorner(
            env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
            env.edges[WEST, r, _prev(c, end)],
            env.edges[NORTH, _prev(r, end), c],
            network[r, c],
            dir,
        )
    elseif dir == NORTHEAST
        return EnlargedCorner(
            env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
            env.edges[NORTH, _prev(r, end), c],
            env.edges[EAST, r, _next(c, end)],
            network[r, c],
            dir,
        )
    elseif dir == SOUTHEAST
        return EnlargedCorner(
            env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
            env.edges[EAST, r, _next(c, end)],
            env.edges[SOUTH, _next(r, end), c],
            network[r, c],
            dir,
        )
    elseif dir == SOUTHWEST
        return EnlargedCorner(
            env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
            env.edges[SOUTH, _next(r, end), c],
            env.edges[WEST, r, _prev(c, end)],
            network[r, c],
            dir,
        )
    else
        throw(ArgumentError(lazy"Invalid direction $dir"))
    end
end

"""
    TensorMap(Q::EnlargedCorner)

Instantiate enlarged corner as a `TensorMap`.
"""
function TensorKit.TensorMap(Q::EnlargedCorner)
    if Q.dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.A)
    elseif Q.dir == NORTHEAST
        return enlarge_northeast_corner(Q.E_1, Q.C, Q.E_2, Q.A)
    elseif Q.dir == SOUTHEAST
        return enlarge_southeast_corner(Q.E_1, Q.C, Q.E_2, Q.A)
    elseif Q.dir == SOUTHWEST
        return enlarge_southwest_corner(Q.E_1, Q.C, Q.E_2, Q.A)
    else
        throw(ArgumentError(lazy"Invalid direction $dir"))
    end
end

function renormalize_northwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northwest_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A)
end
function renormalize_northeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northeast_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A)
end
function renormalize_southeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southeast_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A)
end
function renormalize_southwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southwest_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A)
end

# Wrapper around half_infinite_environment contraction using EnlargedCorners (used in ctmrg_projectors)
function half_infinite_environment(ec_1::EnlargedCorner, ec_2::EnlargedCorner)
    return HalfInfiniteEnv(ec_1, ec_2)
end

# Compute left and right projectors sparsely without constructing enlarged corners explicitly
function contract_projectors(U, S, V, Q::EnlargedCorner, Q_next::EnlargedCorner)
    Ar = _rotate_north_localsandwich(Q.A, _prev(Q.dir, 4))
    Ar_next = _rotate_north_localsandwich(Q_next.A, Q_next.dir)
    isqS = sdiag_pow(S, -0.5)
    P_left = left_projector(Q_next.E_1, Q_next.C, Q_next.E_2, V, isqS, Ar_next)
    P_right = right_projector(Q.E_1, Q.C, Q.E_2, U, isqS, Ar)
    return P_left, P_right
end

# --------------------------------
# Sparse half-infinite environment
# --------------------------------

"""
$(TYPEDEF)

Half-infinite CTMRG environment tensor storage.

## Fields

$(FIELDS)

## Constructors

    HalfInfiniteEnv(quadrant1::EnlargedCorner, quadrant2::EnlargedCorner)

Construct sparse half-infinite environment based on two sparse enlarged corners (quadrants).
"""
struct HalfInfiniteEnv{TC, TE, TA}  # TODO: subtype as AbstractTensorMap once TensorKit is updated
    C_1::TC
    C_2::TC
    E_1::TE
    E_2::TE
    E_3::TE
    E_4::TE
    A_1::TA
    A_2::TA
    A_1r::TA # prerotate
    A_2r::TA
    dir::Int
    function HalfInfiniteEnv(
            C_1::TC, C_2::TC, E_1::TE, E_2::TE, E_3::TE, E_4::TE, A_1::TA, A_2::TA, dir::Int
        ) where {TC, TE, TA}
        A_1r = _rotate_north_localsandwich(A_1, dir)
        A_2r = _rotate_north_localsandwich(A_2, dir)
        return new{TC, TE, TA}(C_1, C_2, E_1, E_2, E_3, E_4, A_1, A_2, A_1r, A_2r, dir)
    end
end
function HalfInfiniteEnv(quadrant1::EnlargedCorner, quadrant2::EnlargedCorner)
    return HalfInfiniteEnv(
        quadrant1.C, quadrant2.C,
        quadrant1.E_1, quadrant1.E_2, quadrant2.E_1, quadrant2.E_2,
        quadrant1.A, quadrant2.A,
        quadrant1.dir,
    )
end

"""
    TensorMap(env::HalfInfiniteEnv)

Instantiate half-infinite environment as `TensorMap` explicitly.
"""
function TensorKit.TensorMap(env::HalfInfiniteEnv)  # Dense operator
    return half_infinite_environment(
        env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, env.A_1r, env.A_2r
    )
end

"""
    (env::HalfInfiniteEnv)(x, ::Val{false}) 
    (env::HalfInfiniteEnv)(x, ::Val{true}) 

Contract half-infinite environment with a vector `x`, such that the environment acts as a
linear map or adjoint linear map on `x` if `Val(true)` or `Val(false)` is passed, respectively.
"""
function (env::HalfInfiniteEnv)(x, ::Val{false})  # Linear map: env() * x
    return half_infinite_environment(
        env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, x, env.A_1r, env.A_2r
    )
end
function (env::HalfInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return half_infinite_environment(
        x, env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, env.A_1r, env.A_2r
    )
end

function contract_projectors(U, S, V, env::HalfInfiniteEnv)
    Q = EnlargedCorner(env.C_1, env.E_1, env.E_2, env.A_1, env.dir)
    Q_next = EnlargedCorner(env.C_2, env.E_3, env.E_4, env.A_2, _next(env.dir, 4))
    return contract_projectors(U, S, V, Q, Q_next)
end

function contract_projectors(U, S, V, henv::HalfInfiniteEnv, henv_next::HalfInfiniteEnv)
    isqS = sdiag_pow(S, -0.5)
    P_left = left_projector(
        henv_next.E_1, henv_next.C_1, henv_next.E_2, henv_next.E_3, henv_next.C_2, henv_next.E_4,
        V, isqS,
        henv_next.A_1r, henv_next.A_2r
    )
    P_right = right_projector(
        henv.E_1, henv.C_1, henv.E_2, henv.E_3, henv.C_2, henv.E_4,
        U, isqS,
        henv.A_1r, henv.A_2r
    )
    return P_left, P_right
end

# -----------------------------------------------------
# AbstractTensorMap subtyping and IterSVD compatibility
# -----------------------------------------------------

function TensorKit.storagetype(::Type{HalfInfiniteEnv{TC, TE, TA}}) where {TC, TE, TA}
    return TensorKit.promote_storagetype(TC, TE, storagetype(TA))
end

function TensorKit.spacetype(::Type{HalfInfiniteEnv{TC, TE, TA}}) where {TC, TE, TA}
    return spacetype(TC)
end

function TensorKit.domain(env::HalfInfiniteEnv)
    return domain(env.E_4) * _elementwise_dual(south_virtualspace(env.A_2r))
end

function TensorKit.codomain(env::HalfInfiniteEnv)
    return first(codomain(env.E_1)) * south_virtualspace(env.A_1r)
end

function random_start_vector(env::HalfInfiniteEnv)
    return randn(storagetype(env), domain(env))
end

# --------------------------------
# Sparse full-infinite environment
# --------------------------------

"""
$(TYPEDEF)

Full-infinite CTMRG environment tensor storage.

## Fields

$(FIELDS)

## Constructors
    FullInfiniteEnv(
        quadrant1::E, quadrant2::E, quadrant3::E, quadrant4::E
    ) where {E<:EnlargedCorner}
    
Construct sparse full-infinite environment based on four sparse enlarged corners (quadrants).
"""
struct FullInfiniteEnv{TC, TE, TA}  # TODO: subtype as AbstractTensorMap once TensorKit is updated
    C_1::TC
    C_2::TC
    C_3::TC
    C_4::TC
    E_1::TE
    E_2::TE
    E_3::TE
    E_4::TE
    E_5::TE
    E_6::TE
    E_7::TE
    E_8::TE
    A_1::TA
    A_2::TA
    A_3::TA
    A_4::TA
    A_1r::TA
    A_2r::TA
    A_3r::TA
    A_4r::TA
    dir::Int
    function FullInfiniteEnv(
            C_1::TC, C_2::TC, C_3::TC, C_4::TC,
            E_1::TE, E_2::TE, E_3::TE, E_4::TE, E_5::TE, E_6::TE, E_7::TE, E_8::TE,
            A_1::TA, A_2::TA, A_3::TA, A_4::TA, dir::Int
        ) where {TC, TE, TA}
        A_1r = _rotate_north_localsandwich(A_1, dir)
        A_2r = _rotate_north_localsandwich(A_2, dir)
        A_3r = _rotate_north_localsandwich(A_3, dir)
        A_4r = _rotate_north_localsandwich(A_4, dir)
        return new{TC, TE, TA}(
            C_1, C_2, C_3, C_4,
            E_1, E_2, E_3, E_4, E_5, E_6, E_7, E_8,
            A_1, A_2, A_3, A_4,
            A_1r, A_2r, A_3r, A_4r,
            dir,
        )
    end
end
function FullInfiniteEnv(
        quadrant1::E, quadrant2::E, quadrant3::E, quadrant4::E
    ) where {E <: EnlargedCorner}
    return FullInfiniteEnv(
        quadrant1.C, quadrant2.C, quadrant3.C, quadrant4.C,
        quadrant1.E_1, quadrant1.E_2, quadrant2.E_1, quadrant2.E_2,
        quadrant3.E_1, quadrant3.E_2, quadrant4.E_1, quadrant4.E_2,
        quadrant1.A, quadrant2.A, quadrant3.A, quadrant4.A,
        quadrant1.dir,
    )
end

"""
    TensorMap(env::FullInfiniteEnv)

Instantiate full-infinite environment as `TensorMap` explicitly.
"""
function TensorKit.TensorMap(env::FullInfiniteEnv)  # Dense operator
    return full_infinite_environment(
        env.C_1, env.C_2, env.C_3, env.C_4,
        env.E_1, env.E_2, env.E_3, env.E_4, env.E_5, env.E_6, env.E_7, env.E_8,
        env.A_1r, env.A_2r, env.A_3r, env.A_4r,
    )
end

"""
    (env::FullInfiniteEnv)(x, ::Val{false}) 
    (env::FullInfiniteEnv)(x, ::Val{true}) 

Contract full-infinite environment with a vector `x`, such that the environment acts as a
linear map or adjoint linear map on `x` if `Val(true)` or `Val(false)` is passed, respectively.
"""
function (env::FullInfiniteEnv)(x, ::Val{false})  # Linear map: env() * x
    return full_infinite_environment(
        env.C_1, env.C_2, env.C_3, env.C_4,
        env.E_1, env.E_2, env.E_3, env.E_4, env.E_5, env.E_6, env.E_7, env.E_8,
        x,
        env.A_1r, env.A_2r, env.A_3r, env.A_4r,
    )
end
function (env::FullInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return full_infinite_environment(
        x,
        env.C_1, env.C_2, env.C_3, env.C_4,
        env.E_1, env.E_2, env.E_3, env.E_4, env.E_5, env.E_6, env.E_7, env.E_8,
        env.A_1r, env.A_2r, env.A_3r, env.A_4r,
    )
end

# Wrapper around full_infinite_environment contraction using EnlargedCorners (used in ctmrg_projectors)
function full_infinite_environment(
        ec_1::E, ec_2::E, ec_3::E, ec_4::E
    ) where {E <: EnlargedCorner}
    return FullInfiniteEnv(ec_1, ec_2, ec_3, ec_4)
end

function contract_projectors(U, S, V, env::FullInfiniteEnv)
    ndir = _next(env.dir, 4)
    nndir = _next(ndir, 4)
    henv = HalfInfiniteEnv(
        env.C_1, env.C_2,
        env.E_1, env.E_2, env.E_3, env.E_4,
        env.A_1, env.A_2, env.dir,
    )
    henv_next = HalfInfiniteEnv(
        env.C_3, env.C_4,
        env.E_5, env.E_6, env.E_7, env.E_8,
        env.A_3, env.A_4, nndir,
    )
    return contract_projectors(U, S, V, henv, henv_next)
end


# -----------------------------------------------------
# AbstractTensorMap subtyping and IterSVD compatibility
# -----------------------------------------------------

function TensorKit.storagetype(::Type{FullInfiniteEnv{TC, TE, TA}}) where {TC, TE, TA}
    return TensorKit.promote_storagetype(TC, TE, storagetype(TA))
end

function TensorKit.spacetype(::Type{FullInfiniteEnv{TC, TE, TA}}) where {TC, TE, TA}
    return spacetype(TC)
end

function TensorKit.domain(env::FullInfiniteEnv)
    return domain(env.E_8) * _elementwise_dual(north_virtualspace(env.A_4r))
end

function TensorKit.codomain(env::FullInfiniteEnv)
    return first(codomain(env.E_1)) * south_virtualspace(env.A_1r)
end

function random_start_vector(env::FullInfiniteEnv)
    return randn(storagetype(env), domain(env))
end

# -----------------------------
# Sparse column-enlarged corner
# -----------------------------

"""
$(TYPEDEF)

Column-enlarged CTMRG corner tensor storage.

## Constructors

    ColumnEnlargedCorner(env, coordinates)

Construct a column-enlarged corner with the correct row and column indices 
based on the given `coordinates` which are of the form `(dir, row, col)`.

```
    [NORTHWEST,r,c]         [NORTHEAST,r,c]

        c-1    c                 c     c+1
    r   C₁--←--E₁--←--      --←--E₂--←--C₂  r
        ↓      |                 |      ↑

        ↓      |                 |      ↑
    r   C₄--→--E₃--→--      --→--E₃--→--C₃  r
        c-1    c                 c     c+1

    [SOUTHWEST,r,c]         [SOUTHEAST,r,c]
```
"""
struct ColumnEnlargedCorner{TC, TE}
    C::TC
    E::TE
    dir::Int
end
function ColumnEnlargedCorner(env::CTMRGEnv, coordinates)
    dir, row, col = coordinates
    Nc = size(env, 3)
    if dir == NORTHWEST
        cm1 = _prev(col, Nc)
        return ColumnEnlargedCorner(
            env.corners[dir, row, cm1], env.edges[NORTH, row, col], dir
        )
    else
        error("Not implemented.")
    end
end

"""
    TensorMap(Q::ColumnEnlargedCorner)

Instantiate column-enlarged corner as a `TensorMap`.
"""
function TensorKit.TensorMap(Q::ColumnEnlargedCorner)
    if Q.dir == NORTHWEST
        return column_enlarge_northwest_corner(Q.C, Q.E)
    else
        error("Not implemented.")
    end
end

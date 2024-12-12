# --------------------------------------------------------
# Sparse enlarged corner as building block for environment
# --------------------------------------------------------

"""
    struct EnlargedCorner{Ct,E,A,A′}

Enlarged CTMRG corner tensor storage.
"""
struct EnlargedCorner{Ct,E,A,A′}
    C::Ct
    E_1::E
    E_2::E
    ket::A
    bra::A′
end

"""
    EnlargedCorner(state, envs, coordinates)

Construct an enlarged corner with the correct row and column indices based on the given
`coordinates` which are of the form `(dir, row, col)`.
"""
function EnlargedCorner(state, envs, coordinates)
    dir, r, c = coordinates
    if dir == NORTHWEST
        return EnlargedCorner(
            envs.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
            envs.edges[WEST, r, _prev(c, end)],
            envs.edges[NORTH, _prev(r, end), c],
            state[r, c],
            state[r, c],
        )
    elseif dir == NORTHEAST
        return EnlargedCorner(
            envs.corners[NORTHEAST, _prev(r, end), _next(c, end)],
            envs.edges[NORTH, _prev(r, end), c],
            envs.edges[EAST, r, _next(c, end)],
            state[r, c],
            state[r, c],
        )
    elseif dir == SOUTHEAST
        return EnlargedCorner(
            envs.corners[SOUTHEAST, _next(r, end), _next(c, end)],
            envs.edges[EAST, r, _next(c, end)],
            envs.edges[SOUTH, _next(r, end), c],
            state[r, c],
            state[r, c],
        )
    elseif dir == SOUTHWEST
        return EnlargedCorner(
            envs.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
            envs.edges[SOUTH, _next(r, end), c],
            envs.edges[WEST, r, _prev(c, end)],
            state[r, c],
            state[r, c],
        )
    end
end

"""
    TensorKit.TensorMap(Q::EnlargedCorner, dir::Int)

Instantiate enlarged corner as `TensorMap` where `dir` selects the correct contraction
direction, i.e. the way the environment and PEPS tensors connect.
"""
function TensorKit.TensorMap(Q::EnlargedCorner, dir::Int)
    if dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == NORTHEAST
        return enlarge_northeast_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == SOUTHEAST
        return enlarge_southeast_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == SOUTHWEST
        return enlarge_southwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    end
end

function renormalize_northwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northwest_corner(
        ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra
    )
end
function renormalize_northeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northeast_corner(
        ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra
    )
end
function renormalize_southeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southeast_corner(
        ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra
    )
end
function renormalize_southwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southwest_corner(
        ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra
    )
end

# Wrapper around half_infinite_environment contraction using EnlargedCorners (used in ctmrg_projectors)
function half_infinite_environment(ec_1::EnlargedCorner, ec_2::EnlargedCorner)
    return HalfInfiniteEnv(ec_1, ec_2)
end

# Compute left and right projectors sparsely without constructing enlarged corners explicitly 
function left_and_right_projector(U, S, V, Q::EnlargedCorner, Q_next::EnlargedCorner)
    isqS = sdiag_pow(S, -0.5)
    P_left = left_projector(Q.E_1, Q.C, Q.E_2, V, isqS, Q.ket, Q.bra)
    P_right = right_projector(
        Q_next.E_1, Q_next.C, Q_next.E_2, U, isqS, Q_next.ket, Q_next.bra
    )
    return P_left, P_right
end

# --------------------------------
# Sparse half-infinite environment
# --------------------------------

"""
    struct HalfInfiniteEnv{C,E,A,A′}

Half-infinite CTMRG environment tensor storage.
"""
struct HalfInfiniteEnv{C,E,A,A′}  # TODO: subtype as AbstractTensorMap once TensorKit is updated
    C_1::C
    C_2::C
    E_1::E
    E_2::E
    E_3::E
    E_4::E
    ket_1::A
    ket_2::A
    bra_1::A′
    bra_2::A′
end

# Construct environment from two enlarged corners
function HalfInfiniteEnv(quadrant1::EnlargedCorner, quadrant2::EnlargedCorner)
    return HalfInfiniteEnv(
        quadrant1.C,
        quadrant2.C,
        quadrant1.E_1,
        quadrant1.E_2,
        quadrant2.E_1,
        quadrant2.E_2,
        quadrant1.ket,
        quadrant2.ket,
        quadrant1.bra,
        quadrant2.bra,
    )
end

"""
    TensorKit.TensorMap(env::HalfInfiniteEnv)

Instantiate half-infinite environment as `TensorMap` explicitly.
"""
function TensorKit.TensorMap(env::HalfInfiniteEnv)  # Dense operator
    return half_infinite_environment(
        env.C_1,
        env.C_2,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        env.ket_1,
        env.ket_2,
        env.bra_1,
        env.bra_2,
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
        env.C_1,
        env.C_2,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        x,
        env.ket_1,
        env.ket_2,
        env.bra_1,
        env.bra_2,
    )
end
function (env::HalfInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return half_infinite_environment(
        x,
        env.C_1,
        env.C_2,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        env.ket_1,
        env.ket_2,
        env.bra_1,
        env.bra_2,
    )
end

# AbstractTensorMap subtyping and IterSVD compatibility
function TensorKit.domain(env::HalfInfiniteEnv)
    return domain(env.E_4) * domain(env.ket_2)[3] * domain(env.bra_2)[3]'
end

function TensorKit.codomain(env::HalfInfiniteEnv)
    return codomain(env.E_1)[1] * domain(env.ket_1)[3]' * domain(env.bra_1)[3]
end

function random_start_vector(env::HalfInfiniteEnv)
    return Tensor(randn, domain(env))
end

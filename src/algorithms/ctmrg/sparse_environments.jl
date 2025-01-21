# --------------------------------------------------------
# Sparse enlarged corner as building block for environment
# --------------------------------------------------------

"""
    struct EnlargedCorner{TC,TE,TA}

Enlarged CTMRG corner tensor storage.
"""
struct EnlargedCorner{TC,TE,TA}
    C::TC
    E_1::TE
    E_2::TE
    A::TA
end
function EnlargedCorner(
    C::CTMRGCornerTensor,
    E1::CTMRG_PEPS_EdgeTensor,
    E2::CTMRG_PEPS_EdgeTensor,
    ket::PEPSTensor,
    bra::PEPSTensor=ket,
)
    return EnlargedCorner(C, E1, E2, (bra, ket))
end
function EnlargedCorner(
    C::CTMRGCornerTensor, E1::CTMRG_PF_EdgeTensor, E2::CTMRG_PF_EdgeTensor, pf::PFTensor
)
    return EnlargedCorner(C, E1, E2, (pf,))
end

"""
    EnlargedCorner(network::InfiniteSquareNetwork, envs, coordinates)

Construct an enlarged corner with the correct row and column indices based on the given
`coordinates` which are of the form `(dir, row, col)`.
"""
function EnlargedCorner(network::InfiniteSquareNetwork, envs, coordinates)
    dir, r, c = coordinates
    if dir == NORTHWEST
        return EnlargedCorner(
            envs.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
            envs.edges[WEST, r, _prev(c, end)],
            envs.edges[NORTH, _prev(r, end), c],
            network[r, c],
        )
    elseif dir == NORTHEAST
        return EnlargedCorner(
            envs.corners[NORTHEAST, _prev(r, end), _next(c, end)],
            envs.edges[NORTH, _prev(r, end), c],
            envs.edges[EAST, r, _next(c, end)],
            network[r, c],
        )
    elseif dir == SOUTHEAST
        return EnlargedCorner(
            envs.corners[SOUTHEAST, _next(r, end), _next(c, end)],
            envs.edges[EAST, r, _next(c, end)],
            envs.edges[SOUTH, _next(r, end), c],
            network[r, c],
        )
    elseif dir == SOUTHWEST
        return EnlargedCorner(
            envs.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
            envs.edges[SOUTH, _next(r, end), c],
            envs.edges[WEST, r, _prev(c, end)],
            network[r, c],
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
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.A...)
    elseif dir == NORTHEAST
        return enlarge_northeast_corner(Q.E_1, Q.C, Q.E_2, Q.A...)
    elseif dir == SOUTHEAST
        return enlarge_southeast_corner(Q.E_1, Q.C, Q.E_2, Q.A...)
    elseif dir == SOUTHWEST
        return enlarge_southwest_corner(Q.E_1, Q.C, Q.E_2, Q.A...)
    end
end

function renormalize_northwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northwest_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A...)
end
function renormalize_northeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_northeast_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A...)
end
function renormalize_southeast_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southeast_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A...)
end
function renormalize_southwest_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_southwest_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.A...)
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
struct HalfInfiniteEnv{TC,TE,TA}  # TODO: subtype as AbstractTensorMap once TensorKit is updated
    C_1::TC
    C_2::TC
    E_1::TE
    E_2::TE
    E_3::TE
    E_4::TE
    A_1::TA
    A_2::TA
end
function HalfInfiniteEnv(
    C_1::CTMRGCornerTensor,
    C_2::CTMRGCornerTensor,
    E_1::CTMRG_PEPS_EdgeTensor,
    E_2::CTMRG_PEPS_EdgeTensor,
    E_3::CTMRG_PEPS_EdgeTensor,
    E_4::CTMRG_PEPS_EdgeTensor,
    ket_1::PEPSTensor,
    ket_2::PEPSTensor,
    bra_1::PEPSTensor=ket_1,
    bra_2::PEPSTensor=ket_2,
)
    return HalfInfiniteEnv(C_1, C_2, E_1, E_2, E_3, E_4, (bra_1, ket_1), (bra_2, ket_2))
end
function HalfInfiniteEnv(
    C_1::CTMRGCornerTensor,
    C_2::CTMRGCornerTensor,
    E_1::CTMRG_PF_EdgeTensor,
    E_2::CTMRG_PF_EdgeTensor,
    E_3::CTMRG_PF_EdgeTensor,
    E_4::CTMRG_PF_EdgeTensor,
    pf_1::PFTensor,
    pf_2::PFTensor,
)
    return HalfInfiniteEnv(C_1, C_2, E_1, E_2, E_3, E_4, (pf_1,), (pf_2,))
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
        quadrant1.A_1,
        quadrant2.A_2,
    )
end

"""
    TensorKit.TensorMap(env::HalfInfiniteEnv)

Instantiate half-infinite environment as `TensorMap` explicitly.
"""
function TensorKit.TensorMap(env::HalfInfiniteEnv)  # Dense operator
    return half_infinite_environment(
        env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, env.A_1..., env.A_2...
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
        env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, x, env.A_1..., env.A_2...
    )
end
function (env::HalfInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return half_infinite_environment(
        x, env.C_1, env.C_2, env.E_1, env.E_2, env.E_3, env.E_4, env.A_1..., env.A_2...
    )
end

# -----------------------------------------------------
# AbstractTensorMap subtyping and IterSVD compatibility
# -----------------------------------------------------

function _domain_space(A::NTuple{2,<:PEPSTensor}, n::Int)
    return domain(A[1])[n] * domain(A[2])[n]'
end
function _domain_space(A::NTuple{1,<:PFTensor}, n::Int)
    return domain(A[1])[n]
end

function _codomain_space(A::NTuple{2,<:PEPSTensor}, n::Int)
    return domain(A[1])[n]' * domain(A[2])[n]
end
function _codomain_space(A::NTuple{1,<:PFTensor}, n::Int)
    return domain(A[1])[n]'
end

function TensorKit.domain(env::HalfInfiniteEnv)
    return domain(env.E_4) * _domain_space(env.A_2, 3)
end

function TensorKit.codomain(env::HalfInfiniteEnv)
    return codomain(env.E_1)[1] * _codomain_space(env.A_1, 3)
end

function random_start_vector(env::HalfInfiniteEnv)
    return Tensor(randn, domain(env))
end

# --------------------------------
# Sparse full-infinite environment
# --------------------------------

"""
    struct FullInfiniteEnv{TC,TE,TA}

Full-infinite CTMRG environment tensor storage.
"""
struct FullInfiniteEnv{TC,TE,TA}  # TODO: subtype as AbstractTensorMap once TensorKit is updated
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
end

# Construct environment from two enlarged corners
function FullInfiniteEnv(
    quadrant1::E, quadrant2::E, quadrant3::E, quadrant4::E
) where {E<:EnlargedCorner}
    return FullInfiniteEnv(
        quadrant1.C,
        quadrant2.C,
        quadrant3.C,
        quadrant4.C,
        quadrant1.E_1,
        quadrant1.E_2,
        quadrant2.E_1,
        quadrant2.E_2,
        quadrant3.E_1,
        quadrant3.E_2,
        quadrant4.E_1,
        quadrant4.E_2,
        quadrant1.A,
        quadrant2.A,
        quadrant3.A,
        quadrant4.A,
    )
end

"""
    TensorKit.TensorMap(env::FullInfiniteEnv)

Instantiate full-infinite environment as `TensorMap` explicitly.
"""
function TensorKit.TensorMap(env::FullInfiniteEnv)  # Dense operator
    return full_infinite_environment(
        env.C_1,
        env.C_2,
        env.C_3,
        env.C_4,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        env.E_2,
        env.E_3,
        env.E_4,
        env.E_5,
        env.A_1...,
        env.A_2...,
        env.A_3...,
        env.A_4...,
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
        env.C_1,
        env.C_2,
        env.C_3,
        env.C_4,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        env.E_5,
        env.E_6,
        env.E_7,
        env.E_8,
        x,
        env.A_1...,
        env.A_2...,
        env.A_3...,
        env.A_4...,
    )
end
function (env::FullInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return full_infinite_environment(
        x,
        env.C_1,
        env.C_2,
        env.C_3,
        env.C_4,
        env.E_1,
        env.E_2,
        env.E_3,
        env.E_4,
        env.E_5,
        env.E_6,
        env.E_7,
        env.E_8,
        env.A_1...,
        env.A_2...,
        env.A_3...,
        env.A_4...,
    )
end

# Wrapper around full_infinite_environment contraction using EnlargedCorners (used in ctmrg_projectors)
function full_infinite_environment(
    ec_1::E, ec_2::E, ec_3::E, ec_4::E
) where {E<:EnlargedCorner}
    return FullInfiniteEnv(ec_1, ec_2, ec_3, ec_4)
end

# AbstractTensorMap subtyping and IterSVD compatibility
function TensorKit.domain(env::FullInfiniteEnv)
    return domain(env.E_8) * _domain_space(env.A_4, 3)
end

function TensorKit.codomain(env::FullInfiniteEnv)
    return codomain(env.E_1)[1] * _codomain_space(env.A_1, 3)
end

function random_start_vector(env::FullInfiniteEnv)
    return Tensor(randn, domain(env))
end

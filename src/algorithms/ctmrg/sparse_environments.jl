const SparseCTMRG = CTMRG{M,true} where {M}  # TODO: Is this really a good way to dispatch on the sparse CTMRG methods?

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
    (Q::EnlargedCorner)(::Val{<:Int})

Contract enlarged corner where `Val(1)` dispatches the north-west, `Val(2)` the north-east
`Val(3)` the south-east and `Val(4)` the south-west contraction.
"""
(Q::EnlargedCorner)(::Val{1}) = enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
(Q::EnlargedCorner)(::Val{2}) = enlarge_northeast_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
(Q::EnlargedCorner)(::Val{3}) = enlarge_southeast_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
(Q::EnlargedCorner)(::Val{4}) = enlarge_southwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)

"""
    enlarge_corner(::Val{<:Int}, (r, c), envs, state, alg::SparseCTMRG)

Enlarge corner but return as a `EnlargedCorner` struct used in sparse CTMRG.
"""
function enlarge_corner(::Val{1}, (r, c), envs, state, alg::SparseCTMRG)
    return EnlargedCorner(
        envs.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        envs.edges[WEST, r, _prev(c, end)],
        envs.edges[NORTH, _prev(r, end), c],
        state[r, c],
        state[r, c],
    )
end
function enlarge_corner(::Val{2}, (r, c), envs, state, alg::SparseCTMRG)
    return EnlargedCorner(
        envs.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        envs.edges[NORTH, _prev(r, end), c],
        envs.edges[EAST, r, _next(c, end)],
        state[r, c],
        state[r, c],
    )
end
function enlarge_corner(::Val{3}, (r, c), envs, state, alg::SparseCTMRG)
    return EnlargedCorner(
        envs.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        envs.edges[EAST, r, _next(c, end)],
        envs.edges[SOUTH, _next(r, end), c],
        state[r, c],
        state[r, c],
    )
end
function enlarge_corner(::Val{4}, (r, c), envs, state, alg::SparseCTMRG)
    return EnlargedCorner(
        envs.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        envs.edges[SOUTH, _next(r, end), c],
        envs.edges[WEST, r, _prev(c, end)],
        state[r, c],
        state[r, c],
    )
end

# Compute left & right projectors from enlarged corner struct
function build_projectors(
    U::AbstractTensorMap{E,3,1},
    S::AbstractTensorMap{E,1,1},
    V::AbstractTensorMap{E,1,3},
    Q::EnlargedCorner,
    Q_next::EnlargedCorner,
) where {E<:ElementarySpace}
    isqS = sdiag_inv_sqrt(S)
    P_left = left_projector(Q.E_1, Q.C, Q.E_2, V, isqS, Q.ket, Q.bra)
    P_right = right_projector(
        Q_next.E_1, Q_next.C, Q_next.E_2, U, isqS, Q_next.ket, Q_next.bra
    )
    return P_left, P_right
end

function renormalize_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra)
end

"""
    struct HalfInfiniteEnv{C,E,A,A′}

Half-infinite CTMRG environment tensor storage.
"""
struct HalfInfiniteEnv{C,E,A,A′}
    C_1::C
    C_2::C
    E_1::E
    E_2::E
    E_3::E
    E_4::E
    ket_1::A
    bra_1::A′
    ket_2::A
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
        quadrant1.ket_bra,
        quadrant2.ket_bra,
    )
end

"""
    (env::HalfInfiniteEnv)() 
    (env::HalfInfiniteEnv)(x) 

Contract half-infinite environment without or with a vector `x`.
"""
function (env::HalfInfiniteEnv)()
    return halfinfinite_environment(
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
function (env::HalfInfiniteEnv)(x)
    return halfinfinite_environment(
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

# TensorKit methods to make struct compatible with sparse SVD
TensorKit.InnerProductStyle(::HalfInfiniteEnv) = EuclideanProduct()
TensorKit.sectortype(::HalfInfiniteEnv) = Trivial
TensorKit.storagetype(env::HalfInfiniteEnv) = storagetype(env.ket_1)
TensorKit.spacetype(env::HalfInfiniteEnv) = spacetype(env.ket_1)

function TensorKit.blocks(env::HalfInfiniteEnv)
    return TensorKit.SingletonDict(Trivial() => env)
end
function TensorKit.blocksectors(::HalfInfiniteEnv)
    return TensorKit.OneOrNoneIterator{Trivial}(true, Trivial())
end

function TensorKit.MatrixAlgebra.svd!(env::HalfInfiniteEnv, args...)
    return TensorKit.MatrixAlgebra.svd!(env(), args...)
end

Base.eltype(env::HalfInfiniteEnv) = eltype(env.ket_1)
function Base.size(env::HalfInfiniteEnv)  # Treat environment as matrix
    χ_in = dim(space(env.E_1, 1))
    D_inabove = dim(space(env.ket_1, 2))
    D_inbelow = dim(space(env.bra_1, 2))
    χ_out = dim(space(env.E_4, 1))
    D_outabove = dim(space(env.ket_2, 2))
    D_outbelow = dim(space(env.bra_2, 2))
    return (χ_in * D_inabove * D_inbelow, χ_out * D_outabove * D_outbelow)
end
Base.size(env::HalfInfiniteEnv, i::Int) = size(env)[i]

# TODO: implement VectorInterface
VectorInterface.scalartype(env::HalfInfiniteEnv) = scalartype(env.ket_1)

# Wrapper around halfinfinite_environment contraction using EnlargedCorners (used in ctmrg_projectors)
function halfinfinite_environment(ec_1::EnlargedCorner, ec_2::EnlargedCorner)
    return HalfInfiniteEnv(
        ec_1.C,
        ec_2.C,
        ec_1.E_1,
        ec_1.E_2,
        ec_2.E_1,
        ec_2.E_2,
        ec_1.ket,
        ec_2.ket,
        ec_1.bra,
        ec_2.bra,
    )
end

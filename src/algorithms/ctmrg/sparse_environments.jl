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
    (Q::EnlargedCorner)(dir::Int)

Contract enlarged corner where `dir` selects the correct contraction direction,
i.e. the way the environment and PEPS tensors connect.
"""
function (Q::EnlargedCorner)(dir::Int)
    if dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    elseif dir == NORTHWEST
        return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
    end
end
function renormalize_corner(ec::EnlargedCorner, P_left, P_right)
    return renormalize_corner(ec.E_1, ec.C, ec.E_2, P_left, P_right, ec.ket, ec.bra)
end

# ------------------
# Sparse environment
# ------------------

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
    (env::HalfInfiniteEnv)(x, ::Val{false}) 
    (env::HalfInfiniteEnv)(x, ::Val{true}) 

Contract half-infinite environment. If a vector `x` is provided, the environment acts as a
linear map or adjoint linear map on `x` if `Val(true)` or `Val(false)` is passed, respectively.
"""
function (env::HalfInfiniteEnv)()  # Dense operator
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
function (env::HalfInfiniteEnv)(x, ::Val{false})  # Linear map: env() * x
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
function (env::HalfInfiniteEnv)(x, ::Val{true})  # Adjoint linear map: env()' * x
    return halfinfinite_environment(
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

# ------------------------------------------------------------------------
# Methods to make environment compatible with IterSVD and its reverse-rule
# ------------------------------------------------------------------------

TensorKit.InnerProductStyle(::HalfInfiniteEnv) = EuclideanProduct()
TensorKit.sectortype(::HalfInfiniteEnv) = Trivial
TensorKit.storagetype(env::HalfInfiniteEnv) = storagetype(env.ket_1)
TensorKit.spacetype(env::HalfInfiniteEnv) = spacetype(env.ket_1)

function TensorKit.domain(env::HalfInfiniteEnv)
    return domain(env.E_4) * domain(env.ket_2)[3] * domain(env.bra_2)[3]'
end
function TensorKit.codomain(env::HalfInfiniteEnv)
    return codomain(env.E_1)[1] * domain(env.ket_1)[3]' * domain(env.bra_1)[3]
end
function TensorKit.space(env::HalfInfiniteEnv)
    return codomain(env) ← domain(env)
end
function TensorKit.blocks(env::HalfInfiniteEnv)
    return TensorKit.SingletonDict(Trivial() => env)
end
function TensorKit.blocksectors(::HalfInfiniteEnv)
    return TensorKit.OneOrNoneIterator{Trivial}(true, Trivial())
end
function TensorKit.block(env::HalfInfiniteEnv, c::Sector)
    return env
end
function TensorKit.tsvd!(f::HalfInfiniteEnv; trunc=NoTruncation(), p::Real=2, alg=IterSVD())
    return _tsvd!(f, alg, trunc, p)
end
function TensorKit.MatrixAlgebra.svd!(env::HalfInfiniteEnv, args...)
    return TensorKit.MatrixAlgebra.svd!(env(), args...)  # Construct environment densely as fallback
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
VectorInterface.scalartype(env::HalfInfiniteEnv) = scalartype(env.ket_1)

function random_start_vector(env::HalfInfiniteEnv)
    return Tensor(randn, domain(env))
end

function Base.similar(env::HalfInfiniteEnv)
    return HalfInfiniteEnv(
        (similar(getfield(env, field)) for field in fieldnames(HalfInfiniteEnv))...
    )
end

function Base.copyto!(dest::HalfInfiniteEnv, src::HalfInfiniteEnv)
    for field in fieldnames(HalfInfiniteEnv)
        for (bd, bs) in zip(blocks(getfield(dest, field)), blocks(getfield(src, field)))
            copyto!(bd[2], bs[2])
        end
    end
    return dest
end

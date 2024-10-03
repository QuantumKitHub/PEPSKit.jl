"""
    struct HalfInfiniteEnv{A,C,E}

Half-infinite CTMRG environment tensor storage.
"""
struct HalfInfiniteEnv{A,A′,C,E}
    ket_1::A
    bra_1::A′
    ket_2::A
    bra_2::A′
    C_1::C
    C_2::C
    E_1::E
    E_2::E
    E_3::E
    E_4::E
end

# Construct environment from two enlarged corners
function HalfInfiniteEnv(quadrant1::EnlargedCorner, quadrant2::EnlargedCorner)
    return HalfInfiniteEnv(
        quadrant1.ket_bra,
        quadrant2.ket_bra,
        quadrant1.C,
        quadrant2.C,
        quadrant1.E_1,
        quadrant1.E_2,
        quadrant2.E_1,
        quadrant2.E_2,
    )
end

"""
    (env::HalfInfiniteEnv)() 
    (env::HalfInfiniteEnv)(x) 

Contract half-infinite environment without or with a vector `x`.
"""
function (env::HalfInfiniteEnv)()
    return halfinfinite_environment(
        env.E_1,
        env.C_1,
        env.E_2,
        env.E_3,
        env.C_2,
        env.E_4,
        env.ket_1,
        env.ket_2,
        env.bra_1,
        env.bra_2,
    )
end
function (env::HalfInfiniteEnv)(x)
    return halfinfinite_environment(
        env.E_1,
        env.C_1,
        env.E_2,
        env.E_3,
        env.C_2,
        env.E_4,
        x,
        env.ket_1,
        env.ket_2,
        env.bra_1,
        env.bra_2,
    )
end

"""
    struct EnlargedCorner{A,C,E}

Enlarged CTMRG corner tensor storage.

```
     C  --  E_2    --
     |       ||      
    E_1 == ket-bra ==
     |       ||      
```
"""
struct EnlargedCorner{A,A′,Ct,E}
    ket::A
    bra::A′
    C::Ct
    E_1::E
    E_2::E
end

# Contract enlarged corner (use NW corner as convention for connecting environment to PEPS tensor)
function (Q::EnlargedCorner)()
    return enlarge_northwest_corner(Q.E_1, Q.C, Q.E_2, Q.ket, Q.bra)
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

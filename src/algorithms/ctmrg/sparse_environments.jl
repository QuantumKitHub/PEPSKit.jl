"""
    struct HalfInfiniteEnv{A,C,E}

Half-infinite CTMRG environment tensor storage.

```
    C_2 --  E_2      --  E_3      -- C_3
     |       ||          ||           | 
    E_1 == ket_bra_1 == ket_bra_2 == E_4
     |       ||          ||           |
```
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

# Contract half-infinite environment
function (env::HalfInfiniteEnv)() end

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

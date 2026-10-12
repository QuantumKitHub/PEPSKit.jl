# Projector environments (green, teal, yellow, red)
# -------------------------------------------------

"""Collect the tensors surrounding a site in a split-CTMRG left move."""
function _split_local_environment((r, c), network, env::SplitCTMRGEnv)
    return (;
        C1 = corner(env, NORTHWEST, r - 1, c - 1),
        C2 = corner(env, NORTHEAST, r - 1, c + 1),
        C3 = corner(env, SOUTHEAST, r + 1, c + 1),
        C4 = corner(env, SOUTHWEST, r + 1, c - 1),
        N = edge(env, NORTH, r - 1, c), S = edge(env, SOUTH, r + 1, c),
        Wk = edge(env, :ket, WEST, r, c - 1), Wb = edge(env, :bra, WEST, r, c - 1),
        Ek = edge(env, :ket, EAST, r, c + 1), Eb = edge(env, :bra, EAST, r, c + 1),
        A = network[r, c],
    )
end

"""Build the two halves of the environment for the green projectors (Fig. 3)."""
function _split_green_environment((r, c), network, env::SplitCTMRGEnv)
    (; C1, C2, C3, C4, N, S, Wk, Wb, Ek, Eb, A) = _split_local_environment((r, c), network, env)
    top = @autoopt @tensor Q[χsw Dsk; χse Dsb] :=
        Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * C1[χnw; χnnw] * N[χnnw Dnk Dnb; χnne] *
        C2[χnne; χne] * Ek[χne Dek; χie] * Eb[χie Deb; χse] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
    bottom = @autoopt @tensor Q[χne Dnb; χnw Dnk] :=
        Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * C4[χssw; χsw] * S[χsse Dsk Dsb; χssw] *
        C3[χse; χsse] * Ek[χne Dek; χie] * Eb[χie Deb; χse] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
    return top, bottom
end

"""Build the two halves of the environment for the teal projectors (Fig. 7)."""
function _split_teal_environment((r, c), network, env::SplitCTMRGEnv, green_west, green_east)
    r = mod1(r, size(env, 2))
    l = _split_local_environment((r, c), network, env)
    h = _split_local_environment((r, c + 1), network, env)
    nextrow = _next(r, size(env, 2))
    NW = enlarge_northwest_corner((l.Wk, l.Wb), l.C1, l.N, l.A, green_west[2][nextrow])
    NE = enlarge_northeast_corner(h.N, h.C2, (h.Ek, h.Eb), h.A, green_east[1][nextrow])
    SW = enlarge_southwest_corner(l.S, l.C4, (l.Wk, l.Wb), l.A, green_west[1][r])
    SE = enlarge_southeast_corner((h.Ek, h.Eb), h.C3, h.S, h.A, green_east[2][r])
    return NW * NE, SE * SW
end

"""Build the two halves of the environment for the yellow projectors (Fig. 8)."""
function _split_yellow_environment((r, c), network, env::SplitCTMRGEnv)
    (; C1, C2, C3, C4, N, S, Wk, Wb, Ek, Eb, A) = _split_local_environment((r, c), network, env)
    top = @autoopt @tensor Q[χiw Dsk; χie Dnb d] :=
        Wk[χiw Dwk; χnw] * C1[χnw; χnnw] * N[χnnw Dnk Dnb; χnne] * C2[χnne; χne] *
        Ek[χne Dek; χie] * ket(A)[d; Dnk Dek Dsk Dwk]
    bottom = @autoopt @tensor Q[χie Dnb d; χiw Dsk] :=
        Wb[χsw Dwb; χiw] * C4[χssw; χsw] * S[χsse Dsk Dsb; χssw] * C3[χse; χsse] *
        Eb[χie Deb; χse] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
    return top, bottom
end

"""Build the two halves of the environment for the red projectors (Fig. 9)."""
function _split_red_environment((r, c), network, env::SplitCTMRGEnv, yellow_west, yellow_east)
    l = _split_local_environment((r, c), network, env)
    h = _split_local_environment((r, c + 1), network, env)
    Ywl, Ywr = yellow_west[1][r], yellow_west[2][r]
    Yel, Yer = yellow_east[1][r], yellow_east[2][r]
    C1, N, Wk, A = l.C1, l.N, l.Wk, l.A
    NW = @autoopt @tensor Q[χout d Dnb; χne Dek] :=
        Ywr[χout; χiw Dsk] * Wk[χiw Dwk; χnw] * C1[χnw; χnnw] * N[χnnw Dnk Dnb; χne] *
        ket(A)[d; Dnk Dek Dsk Dwk]
    C2, N, Ek, A = h.C2, h.N, h.Ek, h.A
    NE = @autoopt @tensor Q[χnw Dwk; χout d Dnb] :=
        N[χnw Dnk Dnb; χnne] * C2[χnne; χne] * Ek[χne Dek; χie] * Yel[χie Dsk; χout] *
        ket(A)[d; Dnk Dek Dsk Dwk]
    C4, S, Wb, A = l.C4, l.S, l.Wb, l.A
    SW = @autoopt @tensor Q[χse Deb; χout d Dnb] :=
        S[χse Dsk Dsb; χssw] * C4[χssw; χsw] * Wb[χsw Dwb; χiw] * Ywl[χiw Dsk; χout] *
        conj(bra(A)[d; Dnb Deb Dsb Dwb])
    C3, S, Eb, A = h.C3, h.S, h.Eb, h.A
    SE = @autoopt @tensor Q[χout d Dnb; χsw Dwb] :=
        Yer[χout; χie Dsk] * Eb[χie Deb; χse] * C3[χse; χsse] * S[χsse Dsk Dsb; χsw] *
        conj(bra(A)[d; Dnb Deb Dsb Dwb])
    return NW * NE, SE * SW
end

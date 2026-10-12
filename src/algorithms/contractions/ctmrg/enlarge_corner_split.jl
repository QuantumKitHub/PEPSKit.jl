# Corner enlargements
# -------------------

"""Project the ket cut of a northwest enlarged corner before contracting a full strip."""
function enlarge_northwest_corner((Wk, Wb)::Tuple, C, N, A::PEPSSandwich, G)
    return @autoopt @tensor Q[χout Dsb; χne Dek Deb] :=
        G[χout; χsw Dsk] * Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * C[χnw; χnnw] * N[χnnw Dnk Dnb; χne] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Project the ket cut of a northeast enlarged corner before contracting a full strip."""
function enlarge_northeast_corner(N, C, (Ek, Eb)::Tuple, A::PEPSSandwich, G)
    return @autoopt @tensor Q[χnw Dwk Dwb; χout Dsb] :=
        N[χnw Dnk Dnb; χnne] * C[χnne; χne] * Ek[χne Dek; χie] * Eb[χie Deb; χse] * G[χse Dsk; χout] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Project the ket cut of a southwest enlarged corner before contracting a full strip."""
function enlarge_southwest_corner(S, C, (Wk, Wb)::Tuple, A::PEPSSandwich, G)
    return @autoopt @tensor Q[χse Dek Deb; χout Dnb] :=
        S[χse Dsk Dsb; χssw] * C[χssw; χsw] * Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * G[χnw Dnk; χout] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Project the ket cut of a southeast enlarged corner before contracting a full strip."""
function enlarge_southeast_corner((Ek, Eb)::Tuple, C, S, A::PEPSSandwich, G)
    return @autoopt @tensor Q[χout Dnb; χsw Dwk Dwb] :=
        G[χout; χne Dnk] * Ek[χne Dek; χie] * Eb[χie Deb; χse] * C[χse; χsse] * S[χsse Dsk Dsb; χsw] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

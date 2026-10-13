# Corner enlargements
# -------------------

"""Build the northwest enlarged corner for the environment of the teal projectors."""
function enlarge_northwest_corner((Wk, Wb)::Tuple, C, N, A::PEPSSandwich, green)
    return @autoopt @tensor Q[χout Dsb; χne Dek Deb] :=
        green[χout; χsw Dsk] * Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * C[χnw; χnnw] * N[χnnw Dnk Dnb; χne] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Build the northeast enlarged corner for the environment of the teal projectors."""
function enlarge_northeast_corner(N, C, (Ek, Eb)::Tuple, A::PEPSSandwich, green)
    return @autoopt @tensor Q[χnw Dwk Dwb; χout Dsb] :=
        N[χnw Dnk Dnb; χnne] * C[χnne; χne] * Ek[χne Dek; χie] * Eb[χie Deb; χse] * green[χse Dsk; χout] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Build the southwest enlarged corner for the environment of the teal projectors."""
function enlarge_southwest_corner(S, C, (Wk, Wb)::Tuple, A::PEPSSandwich, green)
    return @autoopt @tensor Q[χse Dek Deb; χout Dnb] :=
        S[χse Dsk Dsb; χssw] * C[χssw; χsw] * Wb[χsw Dwb; χiw] * Wk[χiw Dwk; χnw] * green[χnw Dnk; χout] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

"""Build the southeast enlarged corner for the environment of the teal projectors."""
function enlarge_southeast_corner((Ek, Eb)::Tuple, C, S, A::PEPSSandwich, green)
    return @autoopt @tensor Q[χout Dnb; χsw Dwk Dwb] :=
        green[χout; χne Dnk] * Ek[χne Dek; χie] * Eb[χie Deb; χse] * C[χse; χsse] * S[χsse Dsk Dsb; χsw] *
        ket(A)[d; Dnk Dek Dsk Dwk] * conj(bra(A)[d; Dnb Deb Dsb Dwb])
end

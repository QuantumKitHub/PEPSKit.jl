# Renormalization
# ---------------

"""Apply the green and teal projectors sequentially when absorbing a north boundary edge."""
function renormalize_northwest_corner(C::CTMRGCornerTensor, N::CTMRG_PEPS_EdgeTensor, green, teal)
    @tensor tmp[χg Dnb; χe] := green[χg; χw Dnk] * C[χw; χn] * N[χn Dnk Dnb; χe]
    return @tensor χout[χw; χe] := teal[χw; χg Dnb] * tmp[χg Dnb; χe]
end

"""Apply the green and teal projectors sequentially when absorbing a south boundary edge."""
function renormalize_southwest_corner(C::CTMRGCornerTensor, S::CTMRG_PEPS_EdgeTensor, green, teal)
    @tensor tmp[χe Dnb; χg] := S[χe Dnk Dnb; χs] * C[χs; χw] * green[χw Dnk; χg]
    return @tensor χout[χe; χw] := tmp[χe Dnb; χg] * teal[χg Dnb; χw]
end

"""Absorb the ket layer and apply the split projectors to a west-edge factor."""
function renormalize_west_edge(K::CTMRGEdgeTensor{<:Any, <:Any, 2}, green, teal, yellow, red, A::PEPSTensor)
    return @autoopt @tensor E[χinew Dek; χnnew] :=
        K[χiold Dwk; χnold] * A[d; Dnk Dek Dsk Dwk] * green[χnold Dnk; χng] *
        yellow[χiy; χiold Dsk] * teal[χng Dnb; χnnew] * red[χinew; χiy d Dnb]
end

"""Absorb the bra layer and apply the split projectors to the other west-edge factor."""
function renormalize_west_edge(B::CTMRGEdgeTensor{<:Any, <:Any, 2}, green, teal, yellow, red, A::PEPSTensor, ::Val{:bra})
    @autoopt @tensor BY[χsg Dwb; χiy] :=
        green[χsg; χsold Dsk] * B[χsold Dwb; χiold] * yellow[χiold Dsk; χiy]
    @tensor BYT[χsnew Dwb Dsb; χiy] := teal[χsnew; χsg Dsb] * BY[χsg Dwb; χiy]
    @tensor BYTA[χsnew Dnb Deb; χiy d] := BYT[χsnew Dwb Dsb; χiy] * conj(A[d; Dnb Deb Dsb Dwb])
    return @tensor E[χsnew Deb; χinew] := BYTA[χsnew Dnb Deb; χiy d] * red[χiy d Dnb; χinew]
end

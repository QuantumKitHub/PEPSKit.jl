# Absorption of matrices into tensor legs
# ---------------------------------------

"""
    absorb_left(
        A::AbstractTensorMap{<:Any, S}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}

Absorb a matrix `C` into the left of a tensor map `A` by contracting the first index in the codomain of `A`
with the (only) index in the domain of `C`. This can be interpreted as contracting the first
leg of `A` with the last leg of `C`.
"""
function absorb_left(
        A::AbstractTensorMap{<:Any, S}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}
    pC = (codomainind(C), domainind(C))
    pA = ((codomainind(A)[1],), (codomainind(A)[2:end]..., domainind(A)...))
    pCA = (codomainind(A), domainind(A))
    return tensorcontract(C, pC, false, A, pA, false, pCA)
end
function absorb_left(
        P::AbstractTensorMap{<:Any, S, 1, N}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S, N}
    return twistnondual(C, 2) * P
end

"""
    absorb_right(
        A::AbstractTensorMap{<:Any, S}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}

Absorb a matrix `C` into the right of a tensor map `A` by contracting the first index in
the domain of `A` with the (only) index in the codomain of `C`. In the case where `A` has
only one space in its domain, this can be interpreted as contracting the last leg of `A`
with the first leg of `C`.
"""
function absorb_right(
        A::AbstractTensorMap{<:Any, S}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}
    pA = ((codomainind(A)..., domainind(A)[2:end]...), (domainind(A)[1],))
    pC = (codomainind(C), domainind(C))
    pAC = (codomainind(A), (domainind(A)[end], domainind(A)[1:(end - 1)]...))
    return tensorcontract(A, pA, false, C, pC, false, pAC)
end
function absorb_right(
        E::AbstractTensorMap{<:Any, S, N, 1}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S, N}
    return E * twistdual(C, 1)
end

"""
    absorb_left_right(
        A::AbstractTensorMap{<:Any, S},
        CL::AbstractTensorMap{<:Any, S, 1, 1},
        CR::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}

Absorb matrices `CL` and `CR` into the left and right of a tensor map `A` by contracting the
first leg of `A` with the second leg of `CL` and the last leg of `A` with the first leg of
`CR`.
"""
function absorb_left_right(
        A::AbstractTensorMap{<:Any, S},
        CL::AbstractTensorMap{<:Any, S, 1, 1},
        CR::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}
    return absorb_right(absorb_left(A, CL), CR)
end

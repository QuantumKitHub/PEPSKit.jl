# Absorption of matrices into tensor legs
# ---------------------------------------

"""
    absorb_left(
        A::AbstractTensorMap{<:Any, S}, C::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}

Absorb a matrix `C` into the left of a tensor map `A` by contracting the first index of `A` with the last index of `C`.
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

Absorb a matrix `C` into the right of a tensor map `A` by contracting the first index in the domain of `A` with the first index of `C`.
In the case where `A` has only one space in its domain, this can be interpreted as contracting the last leg of `A` with the first leg of `C`.
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

Absorb matrices `CL` and `CR` into the left and right of a tensor map `A` by contracting the first leg of `A` with the second leg of `CL` and the first domain index of `A` with the first leg of `CR`.
"""
function absorb_left_right(
        A::AbstractTensorMap{<:Any, S},
        CL::AbstractTensorMap{<:Any, S, 1, 1},
        CR::AbstractTensorMap{<:Any, S, 1, 1}
    ) where {S}
    return absorb_right(absorb_left(A, CL), CR)
end

"""
    absorb_left_right(A, CL, CR, allocator = DefaultAllocator())

Absorb both matrices into a tensor map with one domain leg and at least one codomain
leg. Use the allocator for intermediates of the combined contraction.
"""
@generated function absorb_left_right(
        A::AbstractTensorMap{<:Any, S, N, 1},
        CL::AbstractTensorMap{<:Any, S, 1, 1},
        CR::AbstractTensorMap{<:Any, S, 1, 1},
        allocator = DefaultAllocator(),
    ) where {S, N}
    physical = ntuple(virtuallabel, N - 1)
    output = tensorexpr(:result, (envlabel(:out), physical...), (envlabel(:in),))
    input = tensorexpr(:A, (envlabel(1), physical...), (envlabel(2),))
    left = tensorexpr(:CL, (envlabel(:out),), (envlabel(1),))
    right = tensorexpr(:CR, (envlabel(2),), (envlabel(:in),))
    return macroexpand(
        @__MODULE__, :(return @autoopt @tensor allocator = allocator $output := $left * $input * $right)
    )
end

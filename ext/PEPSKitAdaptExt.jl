module PEPSKitAdaptExt

using PEPSKit
using Adapt

function Adapt.adapt_structure(to, x::PEPSKit.LocalOperator{T, S}) where {T, S}
    terms′ = Dict(k => adapt(to, v) for (k, v) in x.terms)
    return PEPSKit.LocalOperator{valtype(terms′)}(x.lattice, terms′)
end

function Adapt.adapt_structure(to, x::PEPSKit.InfinitePEPS{T}) where {T}
    A′ = map(a -> adapt(to, a), x.A)
    return InfinitePEPS{eltype(A′)}(A′)
end

end

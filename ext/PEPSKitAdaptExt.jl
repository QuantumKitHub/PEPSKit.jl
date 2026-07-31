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

function Adapt.adapt_structure(to, x::PEPSKit.InfinitePEPO{T}) where {T}
    A′ = map(a -> adapt(to, a), x.A)
    return InfinitePEPO{eltype(A′)}(A′)
end

function Adapt.adapt_structure(to, x::PEPSKit.CTMRGEnv{C, T}) where {C, T}
    C′ = map(c -> adapt(to, c), x.corners)
    T′ = map(t -> adapt(to, t), x.edges)
    return CTMRGEnv{eltype(C′), eltype(T′)}(C′, T′)
end

end

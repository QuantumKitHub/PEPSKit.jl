module PEPSKitAdaptExt

using PEPSKit
using Adapt

function Adapt.adapt_structure(to, x::PEPSKit.LocalOperator{T, S}) where {T, S}
    terms′ = Dict(k=>adapt(to, v) for (k, v) in x.terms)
    return PEPSKit.LocalOperator{valtype(terms′)}(x.lattice, terms′)
end

end

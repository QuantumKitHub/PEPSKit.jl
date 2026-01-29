module PEPSKitAdaptExt

using PEPSKit
using Adapt

function Adapt.adapt_structure(to, x::PEPSKit.LocalOperator{T, S}) where {T, S}
    terms′ = map(t->(t[1]=>adapt(to, t[2])), x.terms)
    return PEPSKit.LocalOperator{typeof(terms′), S}(x.lattice, terms′)
end

#=function Adapt.adapt_structure(to, x::AdjointTensorMap)
    return adjoint(adapt(to, parent(x)))
end
function Adapt.adapt_structure(to, x::DiagonalTensorMap)
    data′ = adapt(to, x.data)
    return DiagonalTensorMap(data′, x.domain)
end=#

end

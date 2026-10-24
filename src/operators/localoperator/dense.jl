# Dense AbstractTensorMap terms

add_term!(operator::LocalOperator, inds::Tuple, term::AbstractTensorMap; kwargs...) = add_term!(operator, collect(inds), term; kwargs...)
add_term!(operator::LocalOperator, inds::AbstractVector, term::AbstractTensorMap; kwargs...) = add_term!(operator, CartesianIndex{2}[CartesianIndex{2}(ind) for ind in inds], term; kwargs...)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::AbstractTensorMap;
        atol = zero(real(scalartype(term))),
    )
    # input checks
    length(inds) == numin(term) == numout(term) || throw(ArgumentError("Incompatible number of indices and tensor legs"))
    allunique(inds) || throw(ArgumentError("`inds` should not contain repeated coordinates."))
    for (i, ind) in enumerate(inds)
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        physicalspace(operator, ind_translated) == domain(term)[i] == codomain(term)[i] ||
            throw(SpaceMismatch("Incompatible physical spaces"))
    end
    norm(term) <= atol && return operator # skip adding negligible terms

    # permute input
    if !issorted(inds)
        I = sortperm(inds)
        inds = inds[I]
        term = permute(term, (Tuple(I), Tuple(I) .+ numout(term)))
    end

    # translate coordinates
    inds = _shift_into_unitcell!(copy(inds), size(operator))

    if haskey(operator.terms, inds)
        operator.terms[inds] isa MPOTerm &&
            throw(ArgumentError("Accumulating terms with the same sites is not implemented for MPO or tensor product terms."))
        operator.terms[inds] = VI.add!!(operator.terms[inds], term)
    else
        operator.terms[inds] = term
    end

    return operator
end

"""
Detect an identically zero dense tensor.
"""
_local_term_iszero(term::AbstractTensorMap) = iszero(norm(term))

_scale_local_term(term::AbstractTensorMap, α::Number) = VI.scale(term, α)

"""
    TensorProductTerm

A single term of a [`LocalOperator`](@ref) given as a tensor product of one rank-2 operator
per site acted on, rather than as the rank-`2N` tensor product itself.

Only operators whose terms are individually tensor products can be represented this way. In
particular several such terms acting on the same set of sites cannot be combined, since a
sum of tensor products is generally no longer a tensor product.

The factors are rank-2, carrying no bond indices, which is what distinguishes this from the
more general [`MPOTerm`](@ref): both are vectors of tensors, but a tensor product term is the
narrower type and therefore wins on dispatch wherever it applies.

When added to a `LocalOperator`, the factors are promoted to a common scalar type.
"""
const TensorProductTerm{T} = AbstractVector{T} where {T <: AbstractTensorMap{<:Number, <:IndexSpace, 1, 1}}

add_term!(operator::LocalOperator, inds::Tuple, term::TensorProductTerm; kwargs...) =
    add_term!(operator, collect(inds), term; kwargs...)
add_term!(operator::LocalOperator, inds::AbstractVector, term::TensorProductTerm; kwargs...) =
    add_term!(operator, CartesianIndex{2}[CartesianIndex{2}(ind) for ind in inds], term; kwargs...)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::TensorProductTerm;
        atol = 0,
    )
    # input checks
    isempty(term) && throw(ArgumentError("A tensor product term should contain at least one tensor."))
    term = _promote_term_scalars(term)
    length(inds) == length(term) ||
        throw(ArgumentError("Incompatible number of indices and tensor product factors"))
    allunique(inds) || throw(ArgumentError("`inds` should not contain repeated coordinates."))
    for (i, ind) in enumerate(inds)
        numin(term[i]) == numout(term[i]) == 1 ||
            throw(ArgumentError("Tensor product factors should be single-site operators"))
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        physicalspace(operator, ind_translated) == domain(term[i])[1] == codomain(term[i])[1] ||
            throw(SpaceMismatch("Incompatible physical spaces"))
    end
    prod(norm, term) <= atol && return operator # skip adding negligible terms

    # permute input, which for a product just reorders the factors along with their sites
    if !issorted(inds)
        I = sortperm(inds)
        inds = inds[I]
        term = term[I]
    end

    # translate coordinates
    inds = _shift_into_unitcell!(copy(inds), size(operator))

    # a sum of tensor products is not a tensor product, so terms cannot be accumulated here
    haskey(operator.terms, inds) && throw(
        ArgumentError(
            "A term acting on $inds is already present. Tensor product terms cannot be \
            summed, since a sum of tensor products is not itself a tensor product."
        )
    )
    operator.terms[inds] = collect(term)

    return operator
end

function _fuse_charge(term::TensorProductTerm, charges::Tuple)
    return map(fuse_charge, term, collect(charges))
end

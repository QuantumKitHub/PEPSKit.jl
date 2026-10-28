"""
    MPOTerm{T}

A single term of a [`LocalOperator`](@ref) given as a matrix product operator with one
tensor per site acted on, rather than as the dense rank-`2N` tensor.

The factors are ordered along the chain and follow the usual MPO convention, rank-3 at the ends and rank-4 in the bulk:

    W₁  : P₁ ← P₁ ⊗ B₁
    Wᵢ  : Bᵢ₋₁ ⊗ Pᵢ ← Pᵢ ⊗ Bᵢ
    W_N : B_{N-1} ⊗ P_N ← P_N

so the physical index in the codomain is the bra index and the one in the domain is the ket
index, matching the convention of a dense term, and the bonds run left to right. This is what
[`gate_to_mpo`](@ref) produces, which is the way to obtain an `MPOTerm` from a dense operator.
A single-site term contains only one rank-2 operator.

When added to a `LocalOperator`, the factors are promoted to a common scalar type.

A [`TensorProductTerm`](@ref) is the special case in which every bond is trivial, so its
factors carry no virtual bond indices and are rank-2 throughout. That is exactly what separates the
two on dispatch: a vector of rank-2 operators is a tensor product term, anything else is an MPO.
"""
const MPOTerm{T} = AbstractVector{T} where {T <: AbstractTensorMap}

"""
Validate an ordered MPO's sites, tensor partitions, physical spaces, and adjacent bond spaces.
"""
function _validate_mpo_term(inds, term::MPOTerm, lattice::AbstractMatrix{<:ElementarySpace})
    isempty(term) && throw(ArgumentError("An MPO term should contain at least one tensor."))
    length(inds) == length(term) ||
        throw(ArgumentError("Incompatible number of indices and MPO factors"))
    allunique(inds) || throw(ArgumentError("`inds` should not contain repeated coordinates."))
    n = length(inds)
    for (i, ind) in enumerate(inds)
        # a factor carries its physical pair plus a bond towards each neighbour it has, so it
        # is rank-2 for a lone site, rank-3 at the ends of a chain and rank-4 in the bulk
        nout = (n == 1 || i == 1) ? 1 : 2
        nin = (n == 1 || i == n) ? 1 : 2
        (numout(term[i]) == nout && numin(term[i]) == nin) || throw(
            ArgumentError(
                "MPO factor $i of $n should have $nout index(es) out and $nin in, got \
                $(numout(term[i])) and $(numin(term[i]))."
            )
        )
        # the bra index is the physical one in the codomain: last for a bulk factor, which
        # carries the incoming bond first, and only for an end factor
        bra = codomain(term[i])[nout]
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(lattice)))
        lattice[ind_translated] == bra == domain(term[i])[1] ||
            throw(SpaceMismatch("Incompatible physical spaces"))
    end
    for i in 1:(n - 1)
        domain(term[i])[numin(term[i])] == codomain(term[i + 1])[1] ||
            throw(SpaceMismatch("Incompatible MPO bond spaces between tensors $i and $(i + 1)."))
    end
    return nothing
end

# TODO: Narrow AbstractVector to MPOTerm once TensorMap's scalar type is restricted to Number.
add_term!(operator::LocalOperator, inds::Tuple, term::AbstractVector) =
    add_term!(operator, collect(inds), term)
add_term!(operator::LocalOperator, inds::AbstractVector, term::AbstractVector) =
    add_term!(operator, CartesianIndex{2}[CartesianIndex{2}(ind) for ind in inds], term)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::AbstractVector
    )
    isempty(term) && throw(ArgumentError("An MPO term should contain at least one tensor."))
    term = _promote_term_scalars(term)
    _validate_mpo_term(inds, term, physicalspace(operator))
    _local_term_iszero(term) && return operator

    # NOTE: `inds` is deliberately *not* sorted here, unlike for dense and tensor product
    # terms, since permuting MPOs is not straightforward.

    # translate coordinates
    inds = _shift_into_unitcell!(copy(inds), size(operator))

    # as for tensor products, a sum of MPOs of fixed bond dimension is not one of the same
    # bond dimension, so terms are not accumulated here
    haskey(operator.terms, inds) && throw(
        ArgumentError(
            "A term acting on $inds is already present. MPO terms cannot be summed in \
            place; combine the operators before splitting them."
        )
    )
    operator.terms[inds] = collect(term)

    return operator
end

"""
Detect an MPO containing a zero factor.
"""
_local_term_iszero(term::MPOTerm) = any(_local_term_iszero, term)

# handles both MPOTerm and TensorProductTerm
function _scale_local_term(term::MPOTerm, α::Number)
    # Every factor is brought to the common scalar type, so the resulting container has a
    # concrete scalar type even though the factors differ in rank. Leaving the untouched
    # factors alone instead would give a container whose element type varies in its scalar
    # parameter, which `scalartype` cannot resolve.
    T = promote_type(typeof(α), map(scalartype, term)...)
    return map(enumerate(term)) do (i, tensor)
        i == 1 && return VI.scale(tensor, α)
        return scalartype(tensor) === T ? tensor : VI.scale(tensor, one(T))
    end
end

function _fuse_charge(term::MPOTerm, charges::Tuple)
    return map(_fuse_mpo_charge, term, collect(charges))
end

"""
Fuse an auxiliary charge into an MPO factor's physical legs while preserving its virtual bond spaces.
"""
function _fuse_mpo_charge(tensor::AbstractTensorMap, charge::Sector)
    nout, nin = numout(tensor), numin(tensor)
    P = codomain(tensor)[nout]
    C = spacetype(tensor)(charge => 1)
    A = storagetype(tensor)
    fuser = isomorphism(A, fuse(P, C) ← P ⊗ C)
    left = nout == 1 ? fuser : id(A, codomain(tensor)[1]) ⊗ fuser
    right = nin == 1 ? fuser : fuser ⊗ id(A, domain(tensor)[2])
    # Move the auxiliary input next to the physical input, including any required braiding.
    order = (
        ntuple(identity, nout + 1),
        (nout + 2, nout + nin + 2, ntuple(i -> nout + 2 + i, nin - 1)...),
    )
    return left * permute(tensor ⊗ id(A, C), order) * right'
end

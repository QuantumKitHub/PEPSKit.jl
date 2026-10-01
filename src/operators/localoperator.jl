# Hamiltonian consisting of local terms
# -------------------------------------
"""
An open-boundary MPO represented by an ordered vector of tensor maps.
The endpoint partitions are `(1, 2)` and `(2, 1)`, with `(2, 2)` tensors in between; a one-site MPO has partition `(1, 1)`.
Dense tensors can be converted with `gate_to_mpo`.
Evaluation of MPO terms and vectors of tensor-product factors is not implemented.
"""
const MPOTerm{T} = AbstractVector{T} where {T <: AbstractTensorMap}

"""
$(TYPEDEF)

A sum of local operators acting on a lattice.
The lattice is stored as a matrix of vector spaces, and the terms are stored as a `Dict` of indices mapping to operators.
Terms can be dense tensor maps or `MPOTerm`s, whose site and factor ordering is preserved.

## Fields

$(TYPEDFIELDS)
- `lattice::Matrix{S}`: The lattice on which the operator acts.
- `terms::Dict{Vector{CartesianIndex{2}}, O}`: The terms of the operator, mapping coordinates to operators

## Constructors

    LocalOperator(lattice::Matrix{S}, terms::Pair...)
    LocalOperator{T, S}(lattice::Matrix{S}, terms::T) where {T,S}

## Examples

```julia
lattice = fill(ℂ^2, 1, 1) # single-site unitcell
O1 = LocalOperator(lattice, [(1, 1),] => σx, [(1, 1), (1, 2)] => σx ⊗ σx, [(1, 1), (2, 1)] => σx ⊗ σx)
```
"""
struct LocalOperator{O, S}
    "lattice of physical spaces on which the gates act"
    lattice::Matrix{S}

    "list of `sites => term` pairs that make up the operator"
    terms::Dict{Vector{CartesianIndex{2}}, O}

    LocalOperator{O, S}(lattice::Matrix{S}) where {O, S} =
        new{O, S}(lattice, Dict{Vector{CartesianIndex{2}}, O}())
end

LocalOperator{O}(lattice::Matrix{<:ElementarySpace}) where {O} =
    LocalOperator{O, eltype(lattice)}(lattice)
LocalOperator{O}(lattice, terms::Pair...) where {O} = LocalOperator{O}(lattice, terms)

function LocalOperator{O}(lattice, terms) where {O}
    operator = LocalOperator{O}(lattice)
    for (inds, term) in terms
        add_term!(operator, inds, term)
    end
    return operator
end

# Default to Any for eltype: needs to be abstract anyways so not that much to gain
LocalOperator(lattice, terms) = LocalOperator{Any}(lattice, terms)
LocalOperator(lattice, terms::Pair...) = LocalOperator(lattice, terms)

"""
Sort operator sites using the default `CartesianIndex` ordering.
When the order changes, permute the corresponding output and input physical legs by the same ordering.
"""
function _sort_op_sites(sites::Vector{CartesianIndex{2}}, op::AbstractTensorMap)
    issorted(sites) && return sites, op
    order = sortperm(sites)
    sites′ = sites[order]
    op′ = permute(op, (Tuple(order), Tuple(order) .+ numout(op)))
    return sites′, op′
end

add_term!(operator::LocalOperator, inds::Tuple, term::Union{AbstractTensorMap, MPOTerm}; kwargs...) =
    add_term!(operator, collect(inds), term; kwargs...)
add_term!(operator::LocalOperator, inds::AbstractVector, term::Union{AbstractTensorMap, MPOTerm}; kwargs...) =
    add_term!(operator, CartesianIndex{2}[CartesianIndex{2}(ind) for ind in inds], term; kwargs...)
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

    inds, term = _sort_op_sites(copy(inds), term)

    # translate coordinates
    _shift_into_unitcell!(inds, size(operator))

    if haskey(operator.terms, inds)
        operator.terms[inds] isa MPOTerm &&
            throw(ArgumentError("Accumulating terms with the same sites is not implemented for MPO terms."))
        operator.terms[inds] = VI.add!!(operator.terms[inds], term)
    else
        operator.terms[inds] = term
    end

    return operator
end

"""
Validate an ordered MPO's sites, tensor partitions, physical spaces, and adjacent bond spaces.
"""
function _validate_mpo_term(sites, term::MPOTerm, lattice::AbstractMatrix{<:ElementarySpace})
    isempty(term) && throw(ArgumentError("An MPO term should contain at least one tensor."))
    length(sites) == length(term) ||
        throw(ArgumentError("The MPO should contain one tensor for every operator site."))
    allunique(sites) || throw(ArgumentError("`inds` should not contain repeated coordinates."))

    N = length(term)
    for (k, op) in enumerate(term)
        expected = N == 1 ? (1, 1) : k == 1 ? (1, 2) : k == N ? (2, 1) : (2, 2)
        (numout(op), numin(op)) == expected ||
            throw(ArgumentError("MPO tensor $k should have partition $expected."))
        site = sites[k]
        physical = lattice[mod1(site[1], size(lattice, 1)), mod1(site[2], size(lattice, 2))]
        physical == codomain(op)[numout(op)] == domain(op)[1] ||
            throw(SpaceMismatch("MPO physical space does not match lattice site $site."))
    end
    for k in 1:(N - 1)
        domain(term[k])[numin(term[k])] == codomain(term[k + 1])[1] ||
            throw(SpaceMismatch("Incompatible MPO bond spaces between tensors $k and $(k + 1)."))
    end
    return nothing
end

"""
Insert an ordered MPO term, copying its coordinate and factor containers before translation.
"""
function add_term!(operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::MPOTerm)
    _validate_mpo_term(inds, term, physicalspace(operator))
    _local_term_iszero(term) && return operator
    inds = _shift_into_unitcell!(copy(inds), size(operator))
    haskey(operator.terms, inds) &&
        throw(ArgumentError("Accumulating terms with the same sites is not implemented for MPO terms."))
    operator.terms[inds] = collect(term)
    return operator
end

"""
Detect an identically zero dense tensor or an MPO containing a zero factor.
"""
_local_term_iszero(term::AbstractTensorMap) = iszero(norm(term))
_local_term_iszero(term::MPOTerm) = any(_local_term_iszero, term)


"""
    checklattice(Bool, args...)
    checklattice(args...)

Helper function for checking lattice compatibility. The first version returns a boolean,
while the second version throws an error if the lattices do not match.
"""
function checklattice(args...)
    return checklattice(Bool, args...) || throw(ArgumentError("Lattice mismatch."))
end
checklattice(::Type{Bool}, arg) = true
function checklattice(::Type{Bool}, arg1, arg2, args...)
    return checklattice(Bool, arg1, arg2) && checklattice(Bool, arg2, args...)
end
function checklattice(::Type{Bool}, H1::LocalOperator, H2::LocalOperator)
    return physicalspace(H1) == physicalspace(H2)
end
function checklattice(::Type{Bool}, peps::InfinitePEPS, O::LocalOperator)
    return physicalspace(peps) == physicalspace(O)
end
function checklattice(::Type{Bool}, H::LocalOperator, peps::InfinitePEPS)
    return checklattice(Bool, peps, H)
end
function checklattice(::Type{Bool}, pepo::InfinitePEPO, O::LocalOperator)
    return size(pepo, 3) == 1 && physicalspace(pepo) == physicalspace(O)
end
function checklattice(::Type{Bool}, O::LocalOperator, pepo::InfinitePEPO)
    return checklattice(Bool, pepo, O)
end
@non_differentiable checklattice(args...)

function Base.similar(operator::LocalOperator, lattice::Matrix{<:ElementarySpace})
    return similar(operator, eltype(operator), lattice)
end
function Base.similar(
        operator::LocalOperator, ::Type{O} = eltype(operator), lattice::Matrix{<:ElementarySpace} = physicalspace(operator)
    ) where {O}
    return LocalOperator{O}(lattice)
end

function Base.repeat(operator::LocalOperator, m::Int, n::Int)
    operator_repeated = similar(operator, repeat(physicalspace(operator), m, n))
    for i in 1:m, j in 1:n
        offset = CartesianIndex((i - 1) * size(operator, 1), (j - 1) * size(operator, 2))
        for (inds, term) in operator.terms
            add_term!(operator_repeated, inds .+ offset, term)
        end
    end
    return operator_repeated
end

"""
    physicalspace(O::LocalOperator)

Return lattice of physical spaces on which the `LocalOperator` is defined.
"""
physicalspace(O::LocalOperator) = O.lattice
Base.@propagate_inbounds physicalspace(O::LocalOperator, I...) =
    periodic_getindex(O, O.lattice, I)

Base.size(O::LocalOperator, args...) = size(physicalspace(O), args...)
Base.eltype(::Type{LocalOperator{O, S}}) where {O, S} = O

# Real and imaginary part
# -----------------------
function Base.real(O::LocalOperator)
    any(term -> term isa MPOTerm, values(O.terms)) &&
        throw(ArgumentError("Taking the real part is not implemented for MPO terms."))
    return LocalOperator(O.lattice, (sites => real(op) for (sites, op) in O.terms)...)
end
function Base.imag(O::LocalOperator)
    any(term -> term isa MPOTerm, values(O.terms)) &&
        throw(ArgumentError("Taking the imaginary part is not implemented for MPO terms."))
    return LocalOperator(O.lattice, (sites => imag(op) for (sites, op) in O.terms)...)
end

# Linear Algebra
# --------------
"""
Scale a local term, applying an MPO's scalar to its first factor only.
"""
_scale_local_term(α::Number, term::AbstractTensorMap) = α * term
_scale_local_term(α::Number, term::MPOTerm) =
    AbstractTensorMap[k == 1 ? α * tensor : tensor for (k, tensor) in enumerate(term)]

Base.:*(α::Number, O::LocalOperator) =
    LocalOperator(physicalspace(O), inds => _scale_local_term(α, term) for (inds, term) in O.terms)
Base.:*(O::LocalOperator, α::Number) = α * O

Base.:/(O::LocalOperator, α::Number) = O * inv(α)
Base.:\(α::Number, O::LocalOperator) = inv(α) * O

function Base.:+(O1::LocalOperator, O2::LocalOperator)
    checklattice(O1, O2)
    return LocalOperator(physicalspace(O1), mergewith(_add_local_terms, O1.terms, O2.terms))
end

"""
Accumulate dense terms while rejecting addition involving an MPO at the same sites.
"""
function _add_local_terms(term1, term2)
    (term1 isa MPOTerm || term2 isa MPOTerm) &&
        throw(ArgumentError("Accumulating terms with the same sites is not implemented for MPO terms."))
    return VI.add(term1, term2)
end

Base.:-(O::LocalOperator) = -1 * O
Base.:-(O1::LocalOperator, O2::LocalOperator) = O1 + (-O2)

# VectorInterface
# ---------------

# Since we allow abstract types in T, value and type domain might not match
function VI.scalartype(operator::LocalOperator)
    return promote_type((_local_term_scalartype(term) for term in values(operator.terms))...)
end

"""
Return the promoted scalar type of a dense tensor or the factors of an MPO.
"""
_local_term_scalartype(term::AbstractTensorMap) = scalartype(term)
_local_term_scalartype(term::MPOTerm) = promote_type((scalartype(tensor) for tensor in term)...)

# Equivalence
# -----------

Base.:(==)(O₁::LocalOperator, O₂::LocalOperator) =
    physicalspace(O₁) == physicalspace(O₂) && O₁.terms == O₂.terms

# Rotation
# ----------------------

# rotation of a lattice site
# (copy logic from Base.rotl90, Base.rotr90, Base.rot180)
function siterotl90(site::CartesianIndex{2}, unitcell::NTuple{2, Int})
    return CartesianIndex(unitcell[2] + 1 - site[2], site[1])
end
function siterotr90(site::CartesianIndex{2}, unitcell::NTuple{2, Int})
    return CartesianIndex(site[2], unitcell[1] + 1 - site[1])
end
function siterot180(site::CartesianIndex{2}, unitcell::NTuple{2, Int})
    return CartesianIndex(unitcell[1] + 1 - site[1], unitcell[2] + 1 - site[2])
end

function Base.rotr90(H::LocalOperator)
    Hsize = size(H)
    lattice2 = rotr90(physicalspace(H))
    terms2 = (siterotr90.(inds, Ref(Hsize)) => term for (inds, term) in H.terms)
    return LocalOperator(lattice2, terms2)
end
function Base.rotl90(H::LocalOperator)
    Hsize = size(H)
    lattice2 = rotl90(physicalspace(H))
    terms2 = (siterotl90.(inds, Ref(Hsize)) => term for (inds, term) in H.terms)
    return LocalOperator(lattice2, terms2)
end
function Base.rot180(H::LocalOperator)
    Hsize = size(H)
    lattice2 = rot180(physicalspace(H))
    terms2 = (siterot180.(inds, Ref(Hsize)) => term for (inds, term) in H.terms)
    return LocalOperator(lattice2, terms2)
end

# Charge shifting
# ---------------
TensorKit.spacetype(::Type{<:LocalOperator{<:Any, S}}) where {S} = S

"""
    add_physical_charge(H::LocalOperator, charges::AbstractMatrix{<:Sector})

Change the spaces of a `LocalOperator` by fusing in an auxiliary charge into the domain of
the operator on every site, according to a given matrix of 'auxiliary' physical charges.
"""
function MPSKit.add_physical_charge(H::LocalOperator, charges::AbstractMatrix{<:Sector})
    size(H) == size(charges) ||
        throw(ArgumentError("Incompatible lattice and auxiliary charge sizes"))
    sectortype(H) === eltype(charges) ||
        throw(SectorMismatch("Incompatible lattice and auxiliary charge sizes"))

    # auxiliary spaces will be fused into codomain, so need to dualize the space to fuse
    # the charge into the domain as desired
    dual_charges = map(dual, charges)
    periodic_charges = PeriodicArray(dual_charges)

    # new physical spaces
    Pspaces = map(physicalspace(H), dual_charges) do P, charge
        return fuse(P, spacetype(H)(charge => 1))
    end

    return LocalOperator(
        Pspaces,
        inds => fuse_charge(op, Tuple(map(Base.Fix1(getindex, periodic_charges), inds))) for (inds, op) in H.terms
    )
end

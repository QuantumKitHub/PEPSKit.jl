# Hamiltonian consisting of local terms
# -------------------------------------
"""
$(TYPEDEF)

A sum of local operators acting on a lattice.
The lattice is stored as a matrix of vector spaces, and the terms are stored as a `Dict` of indices mapping to operators.

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
# TODO: add terms beyond AbstractTensorMap
# e.g. tensor product of 1-site operators, MPOs
add_term!(operator::LocalOperator, inds::Tuple, term::AbstractTensorMap) = add_term!(operator, collect(inds), term)
add_term!(operator::LocalOperator, inds::Vector, term::AbstractTensorMap) = add_term!(operator, map(CartesianIndex{2}, inds), term)
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
    _shift_into_unitcell!(inds, size(operator))

    if haskey(operator.terms, inds)
        operator.terms[inds] = VI.add!!(operator.terms[inds], term)
    else
        operator.terms[inds] = term
    end

    return operator
end

# Tensor product terms
# --------------------
# A term given as one rank-2 operator per site, i.e. an explicit tensor product which is
# never actually formed.

"""
    TensorProductTerm

A single term of a [`LocalOperator`](@ref) given as a tensor product of one rank-2 operator
per site acted on, rather than as the rank-`2N` tensor product itself.

Only operators whose terms are individually tensor products can be represented this way. In
particular several such terms acting on the same set of sites cannot be combined, since a
sum of tensor products is not itself a tensor product.

The factors are rank-2, carrying no bond indices, which is what distinguishes this from the
more general [`MPOTerm`](@ref): both are vectors of tensors, but a tensor product term is the
narrower type and therefore wins on dispatch wherever it applies.
"""
const TensorProductTerm{T} = AbstractVector{T} where {T <: AbstractTensorMap{<:Any, <:Any, 1, 1}}

add_term!(operator::LocalOperator, inds::Tuple, term::TensorProductTerm) =
    add_term!(operator, collect(inds), term)
add_term!(operator::LocalOperator, inds::Vector, term::TensorProductTerm) =
    add_term!(operator, map(CartesianIndex{2}, inds), term)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::TensorProductTerm;
        atol = zero(real(scalartype(first(term)))),
    )
    # input checks
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
    _shift_into_unitcell!(inds, size(operator))

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

# MPO terms
# ---------
# A term given as a chain of MPO tensors, one per site, linked by virtual bonds which are
# contracted directly between neighbours rather than being fused into the PEPS bonds. This
# generalizes a tensor product term, which is the special case of bond dimension 1.

"""
    MPOTerm{T}

A single term of a [`LocalOperator`](@ref) given as a matrix product operator with one
tensor per site acted on, rather than as the dense rank-`2N` tensor.

The factors are ordered along the chain and carry the index conventions

    W₁ : P₁ ← P₁ ⊗ B₁
    Wᵢ : Pᵢ ← Bᵢ₋₁' ⊗ Pᵢ ⊗ Bᵢ
    W_N : P_N ← B_{N-1}' ⊗ P_N

so that the first index of each factor is the bra index and the physical index in the domain
is the ket index, matching the convention of a dense term. A single-site term is just a
rank-2 operator.

Unlike a [`TensorProductTerm`](@ref), an `MPOTerm` can represent any operator: the bond
dimension is the operator's Schmidt rank across each cut. It is worth using in place of the
dense form when that rank is small compared to `d^2`, which is the case for the sums of few
product terms that physical Hamiltonians are built from.

A [`TensorProductTerm`](@ref) is the special case in which every bond has dimension 1, so its
factors carry no bond indices and are rank-2. That is exactly what separates the two on
dispatch: a vector of rank-2 operators is a tensor product term, anything else is an MPO. The
tensor product form is preferred where it applies, since it can be absorbed into the ket
instead of inserted into the contraction.

Construct one from a dense operator with [`mpo_term`](@ref).
"""
const MPOTerm{T} = AbstractVector{T} where {T <: AbstractTensorMap}

"""
$(SIGNATURES)

Bond dimension of an [`MPOTerm`](@ref), i.e. the largest Schmidt rank across its cuts.
"""
function mpobond(term::MPOTerm)
    length(term) == 1 && return 1
    return maximum(1:(length(term) - 1)) do i
        W = term[i]
        return dim(space(W, numind(W)))   # the outgoing bond is the last domain index
    end
end

"""
$(SIGNATURES)

Split a dense `N`-site operator into an [`MPOTerm`](@ref) by successive SVDs along the chain,
distributing the singular values symmetrically over each cut.

Singular values below `rtol` relative to the largest are discarded, so that the bond carries
only the operator's actual Schmidt rank rather than the full `d^2` a compact SVD would
return.
"""
function mpo_term(O::AbstractTensorMap{T, S, N, N}; rtol = 1.0e-12) where {T, S, N}
    N == 1 && return [O]
    factors = Vector{Any}(undef, N)

    # split site 1 off: group (bra₁, ket₁) against everything else
    cod = (1, N + 1)
    dom = (ntuple(k -> 1 + k, N - 1)..., ntuple(k -> N + 1 + k, N - 1)...)
    U, Σ, V = svd_trunc(permute(O, (cod, dom)); trunc = trunctol(; rtol))
    sΣ = sqrt(Σ)
    factors[1] = permute(U * sΣ, ((1,), (2, 3)))
    rest = sΣ * V

    # `rest` always holds: index 1 the bond, then the bra sides of the remaining m sites,
    # then their ket sides
    for i in 2:(N - 1)
        m = N - i + 1
        cod = (1, 2, m + 2)
        dom = (ntuple(k -> 2 + k, m - 1)..., ntuple(k -> m + 2 + k, m - 1)...)
        U, Σ, V = svd_trunc(permute(rest, (cod, dom)); trunc = trunctol(; rtol))
        sΣ = sqrt(Σ)
        factors[i] = permute(U * sΣ, ((2,), (1, 3, 4)))
        rest = sΣ * V
    end

    factors[N] = permute(rest, ((2,), (1, 3)))
    return [factors...]
end

add_term!(operator::LocalOperator, inds::Tuple, term::MPOTerm) =
    add_term!(operator, collect(inds), term)
add_term!(operator::LocalOperator, inds::Vector, term::MPOTerm) =
    add_term!(operator, map(CartesianIndex{2}, inds), term)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::MPOTerm
    )
    # input checks
    length(inds) == length(term) ||
        throw(ArgumentError("Incompatible number of indices and MPO factors"))
    allunique(inds) || throw(ArgumentError("`inds` should not contain repeated coordinates."))
    issorted(inds) || throw(
        ArgumentError(
            "`inds` should be sorted: the MPO factors are ordered along the chain, so \
            reordering the sites would require re-splitting the operator."
        )
    )
    for (i, ind) in enumerate(inds)
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        physicalspace(operator, ind_translated) == codomain(term[i])[1] ||
            throw(SpaceMismatch("Incompatible physical spaces"))
    end

    # translate coordinates
    _shift_into_unitcell!(inds, size(operator))

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
"""
$(SIGNATURES)

Take the real part of a single term of a [`LocalOperator`](@ref).

A [`TensorProductTerm`](@ref) is made real factor by factor, so that the real part of a
tensor product is the tensor product of the real parts of its factors. An [`MPOTerm`](@ref)
is treated the same way, factor by factor.
"""
_real_local_term(term) = real(term)
_real_local_term(term::MPOTerm) = map(real, term)

function Base.real(O::LocalOperator)
    return LocalOperator(
        O.lattice, (sites => _real_local_term(op) for (sites, op) in O.terms)...
    )
end
function Base.imag(O::LocalOperator)
    return LocalOperator(O.lattice, (sites => imag(op) for (sites, op) in O.terms)...)
end

# Linear Algebra
# --------------
"""
$(SIGNATURES)

Scale a single term of a [`LocalOperator`](@ref) by `α`.

Terms which are not plain tensors need not scale by plain multiplication: an
[`MPOTerm`](@ref) - and hence also a [`TensorProductTerm`](@ref) - is scaled by scaling one
of its factors, since multiplying every factor would scale the term it represents by `α^n`.
"""
_scale_local_term(term, α::Number) = α * term
function _scale_local_term(term::MPOTerm, α::Number)
    scaled = collect(term)
    scaled[1] = α * scaled[1]
    return scaled
end

Base.:*(α::Number, O::LocalOperator) = LocalOperator(
    physicalspace(O), inds => _scale_local_term(operator, α) for (inds, operator) in O.terms
)
Base.:*(O::LocalOperator, α::Number) = α * O

Base.:/(O::LocalOperator, α::Number) = O * inv(α)
Base.:\(α::Number, O::LocalOperator) = inv(α) * O

function Base.:+(O1::LocalOperator, O2::LocalOperator)
    checklattice(O1, O2)
    return LocalOperator(physicalspace(O1), mergewith(VI.add, O1.terms, O2.terms))
end

Base.:-(O::LocalOperator) = -1 * O
Base.:-(O1::LocalOperator, O2::LocalOperator) = O1 + (-O2)

# VectorInterface
# ---------------

# Since we allow abstract types in T, value and type domain might not match
function VI.scalartype(operator::LocalOperator)
    return promote_type((scalartype(term[2]) for term in operator.terms)...)
end


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

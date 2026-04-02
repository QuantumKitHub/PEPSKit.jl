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
O1 = LocalOperator(lattice, ((1, 1),) => σx, ((1, 1), (1, 2)) => σx ⊗ σx, ((1, 1), (2, 1)) => σx ⊗ σx)
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

add_term!(operator::LocalOperator, inds::Tuple, term::AbstractTensorMap) = add_term!(operator, collect(inds), term)
add_term!(operator::LocalOperator, inds::Vector, term::AbstractTensorMap) = add_term!(operator, map(CartesianIndex{2}, inds), term)
function add_term!(
        operator::LocalOperator, inds::Vector{CartesianIndex{2}}, term::AbstractTensorMap;
        atol = zero(real(scalartype(term))),
    )
    # input checks
    length(inds) == numin(term) == numout(term) || throw(ArgumentError("Incompatible number of indices and tensor legs"))
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
    I1 = first(inds)
    I1_mod = CartesianIndex(mod1.(Tuple(I1), size(operator)))
    inds .-= (I1 - I1_mod)

    if haskey(operator.terms, inds)
        operator.terms[inds] = VI.add!!(operator.terms[inds], term)
    else
        operator.terms[inds] = term
    end

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
physicalspace(O::LocalOperator, args...) = physicalspace(O)[args...]

Base.size(O::LocalOperator, args...) = size(physicalspace(O), args...)
Base.eltype(::Type{LocalOperator{O, S}}) where {O, S} = O

# Real and imaginary part
# -----------------------
function Base.real(O::LocalOperator)
    return LocalOperator(O.lattice, (sites => real(op) for (sites, op) in O.terms)...)
end
function Base.imag(O::LocalOperator)
    return LocalOperator(O.lattice, (sites => imag(op) for (sites, op) in O.terms)...)
end

# Linear Algebra
# --------------
Base.:*(α::Number, O::LocalOperator) =
    LocalOperator(physicalspace(O), inds => α * operator for (inds, operator) in O.terms)
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

@generated function _fuse_isomorphisms(
        op::AbstractTensorMap{<:Any, S, N, N}, fs::Vector{<:AbstractTensorMap{<:Any, S, 1, 2}}
    ) where {S, N}
    op_out_e = tensorexpr(:op_out, -(1:N), -((1:N) .+ N))
    op_e = tensorexpr(:op, 1:3:(3 * N), 2:3:(3 * N))
    f_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(fs[$i]), -i, (j, j + 2))
    end
    f_dag_es = map(1:N) do i
        j = 3 * (i - 1) + 1
        return tensorexpr(:(twistdual(fs[$i]', 1:2)), (j + 1, j + 2), -(N + i))
    end
    multiplication_ex = Expr(
        :call, :*, op_e, f_es..., f_dag_es...
    )
    return macroexpand(@__MODULE__, :(return @tensor $op_out_e := $multiplication_ex))
end

"""
    _fuse_ids(op::AbstractTensorMap{T, S, N, N}, [Ps::NTuple{N, S}]) where {T, S, N}

Fuse identities on auxiliary physical spaces `Ps` into a given operator `op`.
When `Ps` is not specified, it defaults to the domain spaces of `op`.
"""
function _fuse_ids(op::AbstractTensorMap{T, S, N, N}, Ps::NTuple{N, S}) where {T, S, N}
    # make isomorphisms
    fs = map(1:N) do i
        return isomorphism(fuse(space(op, i), Ps[i]), space(op, i) ⊗ Ps[i])
    end
    # and fuse them into the operator
    return _fuse_isomorphisms(op, fs)
end
function _fuse_ids(op::AbstractTensorMap{T, S, N, N}) where {T, S, N}
    return _fuse_ids(op, Tuple(domain(op)))
end

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
    # also, make indexing periodic for convenience
    Paux = PeriodicArray(map(c -> spacetype(H)(c => 1)', charges))

    # new physical spaces
    Pspaces = map(fuse, physicalspace(H), Paux)

    return LocalOperator(
        Pspaces,
        inds => _fuse_ids(op, Tuple(map(Base.Fix1(getindex, Paux), inds))) for (inds, op) in H.terms
    )
end

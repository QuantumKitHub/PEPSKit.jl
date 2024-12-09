# Hamiltonian consisting of local terms
# -------------------------------------
"""
    struct LocalOperator{T<:Tuple,S}

A sum of local operators acting on a lattice. The lattice is stored as a matrix of vector spaces,
and the terms are stored as a tuple of pairs of indices and operators.

# Fields

- `lattice::Matrix{S}`: The lattice on which the operator acts.
- `terms::T`: The terms of the operator, stored as a tuple of pairs of indices and operators.

# Constructors

    LocalOperator(lattice::Matrix{S}, terms::Pair...)
    LocalOperator{T,S}(lattice::Matrix{S}, terms::T) where {T,S} # expert mode

# Examples

```julia
lattice = fill(ℂ^2, 1, 1) # single-site unitcell
O1 = LocalOperator(lattice, ((1, 1),) => σx, ((1, 1), (1, 2)) => σx ⊗ σx, ((1, 1), (2, 1)) => σx ⊗ σx)
```
"""
struct LocalOperator{T<:Tuple,S}
    lattice::Matrix{S}
    terms::T
    function LocalOperator{T,S}(lattice::Matrix{S}, terms::T) where {T,S}
        plattice = PeriodicArray(lattice)
        # Check if the indices of the operator are valid with themselves and the lattice
        for (inds, operator) in terms
            @assert operator isa AbstractTensorMap
            @assert numout(operator) == numin(operator) == length(inds)
            @assert spacetype(operator) == S

            for i in 1:length(inds)
                @assert space(operator, i) == plattice[inds[i]]
            end
        end
        return new{T,S}(lattice, terms)
    end
end
function LocalOperator(
    lattice::Matrix,
    terms::Pair...;
    atol=maximum(x -> eps(real(scalartype(x[2])))^(3 / 4), terms),
)
    allinds = getindex.(terms, 1)
    alloperators = getindex.(terms, 2)

    relevant_terms = []
    for inds in unique(allinds)
        operator = sum(alloperators[findall(==(inds), allinds)])
        norm(operator) > atol && push!(relevant_terms, inds => operator)
    end

    terms_tuple = Tuple(relevant_terms)
    return LocalOperator{typeof(terms_tuple),eltype(lattice)}(lattice, terms_tuple)
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
function checklattice(::Type{Bool}, H1::LocalOperator, H2::LocalOperator)
    return H1.lattice == H2.lattice
end
function checklattice(::Type{Bool}, peps::InfinitePEPS, O::LocalOperator)
    return size(peps) == size(O.lattice)
end
function checklattice(::Type{Bool}, H::LocalOperator, peps::InfinitePEPS)
    return checklattice(Bool, peps, H)
end
@non_differentiable checklattice(args...)

function Base.repeat(O::LocalOperator, m::Int, n::Int)
    lattice = repeat(O.lattice, m, n)
    terms = []
    for (inds, operator) in O.terms, i in 1:m, j in 1:n
        offset = CartesianIndex((i - 1) * size(O.lattice, 1), (j - 1) * size(O.lattice, 2))
        push!(terms, (inds .+ Ref(offset)) => operator)
    end
    return LocalOperator(lattice, terms...)
end

# Linear Algebra
# --------------
function Base.:*(α::Number, O::LocalOperator)
    scaled_terms = map(((inds, operator),) -> (inds => α * operator), O.terms)
    return LocalOperator{typeof(scaled_terms),eltype(O.lattice)}(O.lattice, scaled_terms)
end
Base.:*(O::LocalOperator, α::Number) = α * O

Base.:/(O::LocalOperator, α::Number) = O * inv(α)
Base.:\(α::Number, O::LocalOperator) = inv(α) * O

function Base.:+(O1::LocalOperator, O2::LocalOperator)
    checklattice(O1, O2)
    return LocalOperator(O1.lattice, O1.terms..., O2.terms...)
end

Base.:-(O::LocalOperator) = -1 * O
Base.:-(O1::LocalOperator, O2::LocalOperator) = O1 + (-O2)

# Rotation and mirroring
# ----------------------

"""
Get the position of `site` after reflection about the anti-diagonal line
"""
function _mirror_antidiag_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 - c + Ncol, 1 - r + Nrow)
end

"""
Get the position of `site` after clockwise (right) rotation by 90 degrees
"""
function _rotr90_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(c, 1 + Nrow - r)
end

"""
Get the position of `site` after counter-clockwise (left) rotation by 90 degrees
"""
function _rotl90_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 + Ncol - c, r)
end

"""
Get the position of `site` after rotation by 180 degrees
"""
function _rot180_site(
    site::S, (Nrow, Ncol)::NTuple{2,Int}
) where {S<:Union{CartesianIndex{2},NTuple{2,Int}}}
    r, c = site[1], site[2]
    return CartesianIndex(1 + Nrow - r, 1 + Ncol - c)
end

function mirror_antidiag(H::LocalOperator)
    lattice2 = mirror_antidiag(H.lattice)
    terms2 = (
        (Tuple(_mirror_antidiag_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rotr90(H::LocalOperator)
    lattice2 = rotr90(H.lattice)
    terms2 = (
        (Tuple(_rotr90_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rotl90(H::LocalOperator)
    lattice2 = rotl90(H.lattice)
    terms2 = (
        (Tuple(_rotl90_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

function Base.rot180(H::LocalOperator)
    lattice2 = rot180(H.lattice)
    terms2 = (
        (Tuple(_rot180_site(site, size(H.lattice)) for site in sites) => op) for
        (sites, op) in H.terms
    )
    return LocalOperator(lattice2, terms2...)
end

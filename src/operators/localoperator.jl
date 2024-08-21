
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
end
function LocalOperator(lattice::Matrix{S}, terms::Pair... ; min_norm_operators = eps()^(3/4)) where {S}
    lattice′ = PeriodicArray(lattice)
    relevant_terms = []
    for (inds, operator) in terms
        # Check if the indices of the operator are valid with themselves and the lattice
        @assert operator isa AbstractTensorMap
        @assert numout(operator) == numin(operator) == length(inds)
        for i in 1:length(inds)
            @assert space(operator, i) == lattice′[inds[i]]
        end        
        # Check if we already have an operator acting on the coordinates
        i = findfirst(existing_inds -> existing_inds == inds, map(first, relevant_terms))
        if !isnothing(i) # We are adding to an existing operator
            new_operator = relevant_terms[i][2] + operator
            if norm(new_operator) > min_norm_operators
                relevant_terms[i] = (inds => new_operator)
            else
                deleteat!(relevant_terms, i)
            end
        else # It's a new operator, add it if its norm is large enough
            norm(operator) > min_norm_operators && push!(relevant_terms, inds => operator)
        end
    end
    relevant_terms = Tuple(relevant_terms)
    return LocalOperator{typeof(relevant_terms), S}(lattice, relevant_terms)
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

function nearest_neighbour_hamiltonian(
    lattice::Matrix{S}, h::AbstractTensorMap{S,2,2}
) where {S}
    terms = []
    for I in eachindex(IndexCartesian(), lattice)
        J1 = I + CartesianIndex(1, 0)
        J2 = I + CartesianIndex(0, 1)
        push!(terms, (I, J1) => h)
        push!(terms, (I, J2) => h)
    end
    return LocalOperator(lattice, terms...)
end

function Base.repeat(O::LocalOperator, m::Int, n::Int)
    lattice = repeat(O.lattice, m, n)
    terms = []
    for (inds, operator) in O.terms, i in 1:m, j in 1:n
        offset = CartesianIndex((i - 1) * size(O.lattice, 1), (j - 1) * size(O.lattice, 2))
        push!(terms, (inds .+ Ref(offset)) => operator)
    end
    return LocalOperator(lattice, terms...)
end

function Base.:*(α::Number, O2::LocalOperator)
    return LocalOperator(O2.lattice, map(t -> (t[1] => α * t[2]), O2.terms)...)
end

function Base.:+(O1::LocalOperator, O2::LocalOperator)
    checklattice(O1, O2) 
    return LocalOperator(O1.lattice, O1.terms..., O2.terms...)
end

function Base.:-(O1::LocalOperator, O2::LocalOperator)
    return O1 + (-1) * O2
end

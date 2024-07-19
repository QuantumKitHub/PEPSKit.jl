
# Hamiltonian consisting of local terms
# -------------------------------------
struct LocalOperator{T<:Tuple,S}
    lattice::Matrix{S}
    terms::T
end
function LocalOperator(lattice::Matrix{S}, terms::Pair...) where {S}
    lattice′ = PeriodicArray(lattice)
    for (inds, operator) in terms
        @assert operator isa AbstractTensorMap
        @assert numout(operator) == numin(operator) == length(inds)
        for i in 1:length(inds)
            @assert space(operator, i) == lattice′[inds[i]]
        end
    end
    return LocalOperator{typeof(terms),S}(lattice, terms)
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

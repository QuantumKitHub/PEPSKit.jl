# Hamiltonian consisting of local terms
# -------------------------------------
"""
$(TYPEDEF)

A sum of local operators acting on a lattice. The lattice is stored as a matrix of vector spaces,
and the terms are stored as a tuple of pairs of indices and operators.

## Fields

- `lattice::Matrix{S}`: The lattice on which the operator acts.
- `terms::T`: The terms of the operator, stored as a tuple of pairs of indices and operators.

## Constructors

    LocalOperator(lattice::Matrix{S}, terms::Pair...)
    LocalOperator{T,S}(lattice::Matrix{S}, terms::T) where {T,S}

## Examples

```julia
lattice = fill(ℂ^2, 1, 1) # single-site unitcell
O1 = LocalOperator(lattice, ((1, 1),) => σx, ((1, 1), (1, 2)) => σx ⊗ σx, ((1, 1), (2, 1)) => σx ⊗ σx)
```
"""
struct LocalOperator{T <: Tuple, S}
    lattice::Matrix{S}
    terms::T
    function LocalOperator{T, S}(lattice::Matrix{S}, terms::T) where {T, S}
        plattice = PeriodicArray(lattice)
        # Check if the indices of the operator are valid with themselves and the lattice
        for (inds, operator) in terms
            @assert operator isa AbstractTensorMap
            @assert eltype(inds) <: CartesianIndex
            @assert numout(operator) == numin(operator) == length(inds)
            @assert spacetype(operator) == S

            for i in 1:length(inds)
                @assert space(operator, i) == plattice[inds[i]]
            end
        end
        return new{T, S}(lattice, terms)
    end
end
function LocalOperator(
        lattice::Matrix, terms::Pair...;
        atol = maximum(x -> eps(real(scalartype(x[2])))^(3 / 4), terms),
    )
    allinds = getindex.(terms, 1)
    alloperators = getindex.(terms, 2)

    relevant_terms = []
    for inds in unique(allinds)
        operator = sum(alloperators[findall(==(inds), allinds)])
        cinds = if !(eltype(inds) <: CartesianIndex) # force indices to be CartesianIndices
            map(CartesianIndex, inds)
        else
            inds
        end
        norm(operator) > atol && push!(relevant_terms, cinds => operator)
    end

    terms_tuple = Tuple(relevant_terms)
    return LocalOperator{typeof(terms_tuple), eltype(lattice)}(lattice, terms_tuple)
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

function Base.repeat(O::LocalOperator, m::Int, n::Int)
    lattice = repeat(O.lattice, m, n)
    terms = []
    for (inds, operator) in O.terms, i in 1:m, j in 1:n
        offset = CartesianIndex((i - 1) * size(O.lattice, 1), (j - 1) * size(O.lattice, 2))
        push!(terms, (inds .+ Ref(offset)) => operator)
    end
    return LocalOperator(lattice, terms...)
end

"""
    physicalspace(O::LocalOperator)

Return lattice of physical spaces on which the `LocalOperator` is defined.
"""
function physicalspace(O::LocalOperator)
    return O.lattice
end

Base.size(O::LocalOperator) = size(physicalspace(O))

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
function Base.:*(α::Number, O::LocalOperator)
    scaled_terms = map(((inds, operator),) -> (inds => α * operator), O.terms)
    return LocalOperator{typeof(scaled_terms), eltype(O.lattice)}(O.lattice, scaled_terms)
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

# Rotation
# ----------------------

# rotation of a lattice site
# TODO: type piracy
Base.rotl90(site::CartesianIndex{2}) = CartesianIndex(2 - site[2], site[1])
Base.rotr90(site::CartesianIndex{2}) = CartesianIndex(site[2], 2 - site[1])
Base.rot180(site::CartesianIndex{2}) = CartesianIndex(2 - site[1], 2 - site[2])

function Base.rotr90(H::LocalOperator)
    lattice2 = rotr90(H.lattice)
    terms2 = ((Tuple(rotr90(site) for site in sites) => op) for (sites, op) in H.terms)
    return LocalOperator(lattice2, terms2...)
end

function Base.rotl90(H::LocalOperator)
    lattice2 = rotl90(H.lattice)
    terms2 = ((Tuple(rotl90(site) for site in sites) => op) for (sites, op) in H.terms)
    return LocalOperator(lattice2, terms2...)
end

function Base.rot180(H::LocalOperator)
    lattice2 = rot180(H.lattice)
    terms2 = ((Tuple(rot180(site) for site in sites) => op) for (sites, op) in H.terms)
    return LocalOperator(lattice2, terms2...)
end

# Charge shifting
# ---------------
TensorKit.spacetype(::Type{T}) where {S, T <: LocalOperator{<:Any, S}} = S

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
        return tensorexpr(:(fs[$i]), -(N + i), (j + 1, j + 2))
    end
    multiplication_ex = Expr(
        :call, :*, op_e, f_es..., map(x -> Expr(:call, :conj, x), f_dag_es)...
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
    size(physicalspace(H)) == size(charges) ||
        throw(ArgumentError("Incompatible lattice and auxiliary charge sizes"))
    sectortype(H) === eltype(charges) ||
        throw(SectorMismatch("Incompatible lattice and auxiliary charge sizes"))

    # auxiliary spaces will be fused into codomain, so need to dualize the space to fuse
    # the charge into the domain as desired
    # also, make indexing periodic for convenience
    Paux = PeriodicArray(map(c -> Vect[typeof(c)](c => 1)', charges))

    # new physical spaces
    Pspaces = map(fuse, physicalspace(H), Paux)

    new_terms = map(H.terms) do (sites, op)
        Paux_slice = map(Base.Fix1(getindex, Paux), sites)
        return sites => _fuse_ids(op, Paux_slice)
    end
    H´ = LocalOperator(Pspaces, new_terms...)

    return H´
end

"""
$(TYPEDEF)

Circuit consisting of local gates and MPOs.

## Fields

$(TYPEDFIELDS)
"""
struct LocalCircuit{O, S}
    "lattice of physical spaces on which the gates act"
    lattice::Matrix{S}

    "list of `sites => gate` pairs that make up the circuit"
    gates::Vector{Pair{Vector{CartesianIndex{2}}, O}}

    LocalCircuit{O, S}(lattice::Matrix{S}) where {O, S} =
        new{O, S}(lattice, Vector{Pair{Vector{CartesianIndex{2}}, O}}())
end

LocalCircuit{O}(lattice::Matrix{<:ElementarySpace}) where {O} =
    LocalCircuit{O, eltype(lattice)}(lattice)
LocalCircuit{O}(lattice, gates::Pair...) where {O} = LocalCircuit{O}(lattice, gates)

function LocalCircuit{O}(lattice, terms) where {O}
    operator = LocalCircuit{O}(lattice)
    for (inds, term) in terms
        add_factor!(operator, inds, term)
    end
    return operator
end

# Default to Any for eltype: needs to be abstract anyways so not that much to gain
LocalCircuit(lattice, terms) = LocalCircuit{Any}(lattice, terms)
LocalCircuit(lattice, terms::Pair...) = LocalCircuit(lattice, terms)

add_factor!(operator::LocalCircuit, inds::Tuple, term::AbstractTensorMap) = add_factor!(operator, collect(inds), term)
add_factor!(operator::LocalCircuit, inds::Vector, term::AbstractTensorMap) = add_factor!(operator, map(CartesianIndex{2}, inds), term)
# for AbstractTensorMap term
function add_factor!(operator::LocalCircuit, inds::Vector{CartesianIndex{2}}, term::AbstractTensorMap)
    # input checks
    length(inds) == numin(term) == numout(term) || throw(ArgumentError("Incompatible number of indices and tensor legs"))
    for (i, ind) in enumerate(inds)
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        physicalspace(operator, ind_translated) == domain(term)[i] == codomain(term)[i] ||
            throw(SpaceMismatch("Incompatible physical spaces at $(ind)."))
    end
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
    push!(operator.gates, inds => term)
    return operator
end
# for MPO term
# TODO: consider directly use MPSKit.FiniteMPO
function add_factor!(
        operator::LocalCircuit, inds::Vector{CartesianIndex{2}}, term::Vector{M}
    ) where {M <: AbstractTensorMap}
    # input checks
    length(inds) >= 2 || throw(ArgumentError("Gate MPO must act on 2 or more sites."))
    length(inds) == length(term) || throw(ArgumentError("Incompatible number of indices and length of gate MPO."))
    allunique(inds) || throw(ArgumentError("`inds` should not contain repeated coordinates."))
    for (i, (ind, t)) in enumerate(zip(inds, term))
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        out_ax = (i == 1) ? 1 : 2
        in_ax = (i == 1) ? 2 : 3
        physicalspace(operator, ind_translated) == space(t, out_ax) == space(t, in_ax)' ||
            throw(SpaceMismatch("Incompatible physical spaces at $(ind)."))
        if i >= 2
            ind_prev = inds[i - 1]
            sum(Tuple(ind - ind_prev) .^ 2) == 1 || throw(ArgumentError("Two consecutive sites in `inds` must be nearest neighbours for MPO terms."))
        end
    end
    # for MPO term, `inds` should not be sorted
    push!(operator.gates, inds => term)
    return operator
end

function checklattice(::Type{Bool}, H1::LocalCircuit, H2::LocalCircuit)
    return physicalspace(H1) == physicalspace(H2)
end
function checklattice(::Type{Bool}, peps::InfinitePEPS, O::LocalCircuit)
    return physicalspace(peps) == physicalspace(O)
end
function checklattice(::Type{Bool}, H::LocalCircuit, peps::InfinitePEPS)
    return checklattice(Bool, peps, H)
end
function checklattice(::Type{Bool}, pepo::InfinitePEPO, O::LocalCircuit)
    return size(pepo, 3) == 1 && physicalspace(pepo) == physicalspace(O)
end
function checklattice(::Type{Bool}, O::LocalCircuit, pepo::InfinitePEPO)
    return checklattice(Bool, pepo, O)
end

"""
    physicalspace(gates::LocalCircuit)

Return lattice of physical spaces on which the `LocalCircuit` is defined.
"""
physicalspace(gates::LocalCircuit) = gates.lattice
physicalspace(gates::LocalCircuit, args...) = physicalspace(gates)[args...]
Base.size(gates::LocalCircuit) = size(physicalspace(gates))

# Equality
# -----------

Base.:(==)(O₁::LocalCircuit, O₂::LocalCircuit) =
    physicalspace(O₁) == physicalspace(O₂) && O₁.gates == O₂.gates

# Rotation
# ----------------------

function Base.rotr90(H::LocalCircuit)
    Hsize = size(H)
    lattice2 = rotr90(physicalspace(H))
    terms2 = (siterotr90.(inds, Ref(Hsize)) => term for (inds, term) in H.gates)
    return LocalCircuit(lattice2, terms2)
end
function Base.rotl90(H::LocalCircuit)
    Hsize = size(H)
    lattice2 = rotl90(physicalspace(H))
    terms2 = (siterotl90.(inds, Ref(Hsize)) => term for (inds, term) in H.gates)
    return LocalCircuit(lattice2, terms2)
end
function Base.rot180(H::LocalCircuit)
    Hsize = size(H)
    lattice2 = rot180(physicalspace(H))
    terms2 = (siterot180.(inds, Ref(Hsize)) => term for (inds, term) in H.gates)
    return LocalCircuit(lattice2, terms2)
end

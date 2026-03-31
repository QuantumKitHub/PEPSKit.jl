"""
$(TYPEDEF)

Collection of Trotter evolution gates and MPOs obtained from
a Hamiltonian containing long-range or multi-site terms.

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
function add_factor!(operator::LocalCircuit, inds::Vector{CartesianIndex{2}}, term::AbstractTensorMap)
    # input checks
    length(inds) == numin(term) == numout(term) || throw(ArgumentError("Incompatible number of indices and tensor legs"))
    for (i, ind) in enumerate(inds)
        ind_translated = CartesianIndex(mod1.(Tuple(ind), size(operator)))
        physicalspace(operator, ind_translated) == domain(term)[i] == codomain(term)[i] ||
            throw(SpaceMismatch("Incompatible physical spaces"))
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

const NNGate{T, S} = AbstractTensorMap{T, S, 2, 2}

"""
Convert an N-site gate to MPO form by SVD, 
in which the axes are ordered as
```
    site 1      mid sites      site N
    2               3               3
    ↓               ↓               ↓
    g1 ←- 3    1 ←- g ←- 4    1 ←- gN
    ↓               ↓               ↓
    1               2               2
```
"""
function gate_to_mpo(
        gate::AbstractTensorMap{<:Any, <:Any, N, N};
        trunc = trunctol(; atol = MPSKit.Defaults.tol)
    ) where {N}
    N == 1 && return gate
    Os = MPSKit.decompose_localmpo(MPSKit.add_util_leg(gate), trunc)
    return map(1:N) do i
        if i == 1
            return removeunit(Os[1], 1)
        elseif i == N
            return removeunit(Os[N], 4)
        else
            return Os[i]
        end
    end
end
gate_to_mpo(x; kwargs...) = x

"""
    _check_hamiltonian_for_trotter(H::LocalOperator)

Assert that operator `H` contains only one-site and two-site terms.
Returns the maximum squared distance covered by a two-site term in `H`.
On the square lattice, the neighbor distances are
```
    1st nb.     2nd nb.     3rd nb.

                    o
                    |
    o---o       o---o       o---o---o

    dist² = 1   dist² = 2   dist² = 4
```
"""
function _check_hamiltonian_for_trotter(H::LocalOperator)
    dist = 0
    for (sites, op) in H.terms
        @assert numin(op) <= 2 "Hamiltonians containing multi-site (> 2) terms are not currently supported."
        if numin(op) == 2
            dist = max(dist, sum(Tuple(sites[1] - sites[2]) .^ 2))
        end
    end
    @assert dist <= 2 "Hamiltonians with 2-site terms on beyond 2nd-neighbor bonds are not currently supported."
    return dist
end

"""
Trotterize a trivial Hamiltonian `H` containing only 1-site terms.
"""
function _trotterize_1site!(gates::Vector, H::LocalOperator, dt::Number)
    for x in CartesianIndices(size(H))
        coord = [x]
        haskey(H.terms, coord) || continue
        gate = exp(H.terms[coord] * -dt)
        push!(gates, coord => gate)
        used_terms += 1
    end
    return gates
end

"""
Trotterize nearest neighbor terms (grouped with 1-site terms)
in the Hamiltonian `H`.

Gate order: `(d, c, r)`
- d = 1: horizontal bond ((r, c), (r, c+1))
- d = 2:   vertical bond ((r, c), (r-1, c))
"""
function _trotterize_nn2site!(gates::Vector, H::LocalOperator, dt::Number)
    # 2-site gates: horizontal nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(0, 1)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate = exp(H.terms[coord] * -dt)
        push!(gates, coord => gate)
    end

    # 2-site gates: vertical nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(1, 0)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate = exp(H.terms[coord] * -dt)
        push!(gates, coord => gate)
    end

    return gates
end

"""
Trotterize a next-nearest neighbor terms in a Hamiltonian.

Gate order: `(c, r, d)`
- d = 1 (NORTHWEST), ..., 4 (SOUTHWEST) labels the triangular 3-site clusters.
"""
function _trotterize_nnn2site!(gates::Vector, H::LocalOperator, dt::Number)
    T = eltype(H)
    # 2-site gates: ⌞ next-nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(1, 1)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate_mpo = gate_to_mpo(exp(H.terms[coord] * -dt))
        b = TensorKit.BraidingTensor{T}(
            left_virtualspace(gate_mpo[2])', physicalspace(H, x + CartesianIndex(1, 0))
        )
        insert!(gate_mpo, 2, TensorMap(b))
        push!(gates, coord => gate)
    end

    # 2-site gates: ⌜ next-nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(-1, 1)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate_mpo = gate_to_mpo(exp(H.terms[coord] * -dt))
        b = TensorKit.BraidingTensor{T}(
            left_virtualspace(gate_mpo[2])', physicalspace(H, x + CartesianIndex(-1, 0))
        )
        insert!(gate_mpo, 2, TensorMap(b))
        push!(gates, coord => gate_mpo)
    end

    # 2-site gates: ⌝ next-nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(1, 1)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate_mpo = gate_to_mpo(exp(H.terms[coord] * -dt))
        b = TensorKit.BraidingTensor{T}(
            left_virtualspace(gate_mpo[2])', physicalspace(H, x + CartesianIndex(0, 1))
        )
        insert!(gate_mpo, 2, TensorMap(b))
        push!(gates, coord => gate)
    end

    # 2-site gates: ⌟ next-nearest-neighbour
    for x in CartesianIndices(size(H))
        y = x + CartesianIndex(-1, 1)
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate_mpo = gate_to_mpo(exp(H.terms[coord] * -dt))
        b = TensorKit.BraidingTensor{T}(
            left_virtualspace(gate_mpo[2])', physicalspace(H, x + CartesianIndex(0, 1))
        )
        insert!(gate_mpo, 2, TensorMap(b))
        push!(gates, coord => gate_mpo)
    end

    return gates
end

"""
Trotterize the evolution operator `exp(-H * dt)`.
Currently, `H` can only contain the following terms:

- 1-site terms
- 2-site nearest neighbor (NN) terms
- 2-site next-nearest neighbor (NNN) terms

## Keyword arguments

- `symmetrize_gates::Bool`: if true, use second-order Trotter decomposition.
- `force_mpo::Bool`: if true, 2-site nearest-neighbor gates are also decomposed to MPOs.
"""
function trotterize(
        H::LocalOperator, dt::Number;
        symmetrize_gates::Bool = false, force_mpo::Bool = false
    )
    for coords in keys(H.terms)
        @assert length(coords) <= 2 "Hamiltonians containing multi-site (>2) terms are not yet supported"
        @assert (length(coords) == 1 || max(abs.(Tuple(coords[1] - coords[2]))...) == 1) "Hamiltonians containing beyond next-nearest neighbour terms are not yet supported"
    end

    dt′ = symmetrize_gates ? (dt / 2) : dt

    gates = Vector{Pair{Vector{CartesianIndex{2}}, Any}}()

    _trotterize_1site!(gates, H, dt′)
    _trotterize_nn2site!(gates, H, dt′)
    _trotterize_nnn2site!(gates, H, dt′)

    if force_mpo
        gates = map(gates) do (coords, gate)
            return coords => gate_to_mpo(gate)
        end
    end

    @assert length(H.terms) == length(gates) "Not all terms were handled"

    symmetrize_gates && append!(gates, reverse(gates))

    return LocalCircuit(physicalspace(H), gates)
end

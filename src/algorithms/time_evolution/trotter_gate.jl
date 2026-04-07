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

const NNGate{T, S} = AbstractTensorMap{T, S, 2, 2}

"""
Convert an N-site gate (N ≥ 2) to MPO by SVD, 
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
    @assert N >= 2
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
    end
    return gates
end

"""
Trotterize nearest neighbor terms (grouped with 1-site terms)
in the Hamiltonian `H`.
"""
function _trotterize_nn2site!(
        gates::Vector, H::LocalOperator, dt::Number; force_mpo::Bool = false
    )
    vs = [CartesianIndex(0, 1), CartesianIndex(1, 0)]
    for x in CartesianIndices(size(H)), v in vs
        y = x + v
        coord = [x, y]
        haskey(H.terms, coord) || continue
        gate = exp(H.terms[coord] * -dt)
        force_mpo && (gate = gate_to_mpo(gate))
        push!(gates, coord => gate)
    end
    return gates
end

"""
Trotterize next-nearest neighbor terms in a Hamiltonian,
converting them to 3-site MPO gates. 
For each gate, the order of sites is
```
    2---3   1---2
    |           |
    1           3

    1           3
    |           |
    2---3   1---2
```
"""
function _trotterize_nnn2site!(gates::Vector, H::LocalOperator, dt::Number)
    T = scalartype(H)
    vs = [
        # ⌞ next-nearest-neighbour
        (CartesianIndex(1, 0), CartesianIndex(1, 1)),
        # ⌜ next-nearest-neighbour
        (CartesianIndex(-1, 0), CartesianIndex(-1, 1)),
        # ⌝ next-nearest-neighbour
        (CartesianIndex(0, 1), CartesianIndex(1, 1)),
        # ⌟ next-nearest-neighbour
        (CartesianIndex(0, 1), CartesianIndex(-1, 1)),
    ]
    for x1 in CartesianIndices(size(H)), v in vs
        x2, x3 = x1 + v[1], x1 + v[2]
        coord = [x1, x3]
        haskey(H.terms, coord) || continue
        gate = gate_to_mpo(exp(H.terms[coord] * -dt / 2))
        x2′ = CartesianIndex(mod1.(Tuple(x2), size(H)))
        b = TensorKit.BraidingTensor{T}(
            physicalspace(H, x2′), left_virtualspace(gate[2])
        )
        insert!(gate, 2, TensorMap(b))
        push!(gates, [x1, x2, x3] => gate)
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
    dist = _check_hamiltonian_for_trotter(H)

    dt′ = symmetrize_gates ? (dt / 2) : dt

    gates = Vector{Pair{Vector{CartesianIndex{2}}, Any}}()

    dist >= 0 && _trotterize_1site!(gates, H, dt′)
    dist >= 1 && _trotterize_nn2site!(gates, H, dt′; force_mpo)
    dist >= 2 && _trotterize_nnn2site!(gates, H, dt′)

    symmetrize_gates && append!(gates, reverse(gates))

    return LocalCircuit(physicalspace(H), gates)
end

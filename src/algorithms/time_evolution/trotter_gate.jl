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

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
    # use MPS convention of domain/codomain
    Os = map(MPSKit.decompose_localmpo(MPSKit.add_util_leg(gate), trunc)) do O
        return permute(O, ((1, 2, 3), (4,)))
    end
    # convert to Vidal gauge
    _cluster_truncate!(Os, fill(notrunc(), N - 1))
    # evenly distribute the (Inf) norm
    nrms = norm.(Os, Inf)
    fac = prod(nrms)^(1 / N)
    for (i, nrm) in enumerate(nrms)
        Os[i] *= fac / nrm
    end
    # remove trivial legs in first/last tensor, and restore MPO convention
    return map(enumerate(Os)) do (i, O)
        if i == 1
            return permute(removeunit(O, 1), ((1,), (2, 3)))
        elseif i == N
            return permute(removeunit(O, 4), ((1, 2), (3,)))
        else
            return permute(O, ((1, 2), (3, 4)))
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
    all(size(H) .>= 2) || error("Unit cell size of the Hamiltonian cannot be smaller than (2, 2).")
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
Trotterize nearest neighbor terms in the Hamiltonian `H`.
"""
function _trotterize_nn2site!(
        gates::Vector, H::LocalOperator, dt::Number; force_mpo::Bool = false
    )
    Nr, Nc = size(H)
    # horizontal bonds, column by column
    # within group `g`, all gates commute
    period = iseven(Nc) ? 2 : 3
    for g in 1:period, c in 1:Nc, r in 1:Nr
        mod1(c, period) == g || continue
        x = CartesianIndex(r, c)
        coord = [x, x + CartesianIndex(0, 1)]
        haskey(H.terms, coord) || continue
        gate = exp(H.terms[coord] * -dt)
        force_mpo && (gate = gate_to_mpo(gate))
        push!(gates, coord => gate)
    end
    # vertical bonds, row by row
    # within group `g`, all gates commute
    period = iseven(Nr) ? 2 : 3
    for g in 1:period, r in 1:Nr, c in 1:Nc
        mod1(r, period) == g || continue
        x = CartesianIndex(r, c)
        coord = [x, x + CartesianIndex(1, 0)]
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
For each gate, the sites are in counter-clockwise order
```
    2---1   3---2
    |           |
    3           1

    1           3
    |           |
    2---3   1---2
```
"""
function _trotterize_nnn2site!(gates::Vector, H::LocalOperator, dt::Number)
    T = scalartype(H)
    origin = CartesianIndex(0, 0)
    vs = (
        # ⌜ northwest next-nearest-neighbour
        (CartesianIndex(-1, 1), CartesianIndex(-1, 0), origin),
        # ⌝ northeast next-nearest-neighbour
        (CartesianIndex(1, 1), CartesianIndex(0, 1), origin),
        # ⌟ southeast next-nearest-neighbour
        (origin, CartesianIndex(0, 1), CartesianIndex(-1, 1)),
        # ⌞ southwest next-nearest-neighbour
        (origin, CartesianIndex(1, 0), CartesianIndex(1, 1)),
    )
    Nr = size(H, 1)
    for (dir, v) in enumerate(vs), x in CartesianIndices(size(H))
        x′ = if dir == NORTHEAST || dir == SOUTHWEST
            x
        else
            CartesianIndex(mod1(x[1] + 1, Nr), x[2])
        end
        x1, x2, x3 = x′ + v[1], x′ + v[2], x′ + v[3]
        coord = [x1, x3]
        rev = !issorted(coord)
        coord′ = rev ? reverse!(coord) : coord
        haskey(H.terms, coord′) || continue
        term = H.terms[coord′]
        if rev
            term = permute(term, ((2, 1), (4, 3)))
        end
        gate = gate_to_mpo(exp(term * -dt / 2))
        b = TensorKit.BraidingTensor{T}(physicalspace(H, x2), left_virtualspace(gate[2]))
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

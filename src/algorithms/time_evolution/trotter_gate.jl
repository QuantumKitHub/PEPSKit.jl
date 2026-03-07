"""
    struct TrotterGates{T <: Vector}

Collection of Trotter evolution gates and MPOs obtained from
a Hamiltonian containing long-range or multi-site terms.
Each item in `data` is a pair `sites => gate`, where `sites` is a
vector of `CartesianIndex`s storing the sites on which the
Trotter `gate` acts.
"""
struct TrotterGates{T <: Vector}
    data::T
end

const NNGate{T, S} = AbstractTensorMap{T, S, 2, 2}

"""
    is_equivalent_site(
        site1::CartesianIndex{2}, site2::CartesianIndex{2},
        (Nrow, Ncol)::NTuple{2, Int}
    )

Check if two lattice sites are related by a (periodic) lattice translation.
"""
function is_equivalent_site(
        site1::CartesianIndex{2}, site2::CartesianIndex{2},
        (Nrow, Ncol)::NTuple{2, Int}
    )
    shift = site1 - site2
    return mod(shift[1], Nrow) == 0 && mod(shift[2], Ncol) == 0
end

"""
    _get_site_term(ham::LocalOperator, site::CartesianIndex{2})

Get the sum of all 1-site terms at `site` in `ham`.
If there are no such terms, return the zero operator at `site`.
"""
function _get_site_term(ham::LocalOperator, site::CartesianIndex{2})
    r, c = mod1.(Tuple(site), size(ham))
    V = physicalspace(ham)[r, c]
    term = zeros(scalartype(ham), V ← V)
    for (sites, op) in ham.terms
        length(sites) != 1 && continue
        if is_equivalent_site(sites[1], site, size(ham))
            term = term + op
        end
    end
    return term
end

"""
    is_equivalent_bond(
        bond1::NTuple{2, CartesianIndex{2}}, bond2::NTuple{2, CartesianIndex{2}},
        (Nrow, Ncol)::NTuple{2, Int},
    )

Check if two 2-site bonds are related by a (periodic) lattice translation.
"""
function is_equivalent_bond(
        bond1::NTuple{2, CartesianIndex{2}}, bond2::NTuple{2, CartesianIndex{2}},
        (Nrow, Ncol)::NTuple{2, Int},
    )
    r1 = bond1[1] - bond1[2]
    r2 = bond2[1] - bond2[2]
    shift_row = bond1[1][1] - bond2[1][1]
    shift_col = bond1[1][2] - bond2[1][2]
    return r1 == r2 && mod(shift_row, Nrow) == 0 && mod(shift_col, Ncol) == 0
end

"""
    _get_bond_term(ham::LocalOperator, bond::NTuple{2, CartesianIndex{2}})

Get the sum of all 2-site terms on `bond` in `ham`.
If there are no such terms, return the zero operator on `bond`.
"""
function _get_bond_term(ham::LocalOperator, bond::NTuple{2, CartesianIndex{2}})
    # create zero operator
    r1, c1 = mod1.(Tuple(bond[1]), size(ham))
    r2, c2 = mod1.(Tuple(bond[2]), size(ham))
    V1 = physicalspace(ham)[r1, c1]
    V2 = physicalspace(ham)[r2, c2]
    term = zeros(scalartype(ham), V1 ⊗ V2 ← V1 ⊗ V2)
    for (sites, op) in ham.terms
        length(sites) != 2 && continue
        if is_equivalent_bond(sites, bond, size(ham))
            term += op
        elseif is_equivalent_bond(sites, reverse(bond), size(ham))
            op′ = permute(op, ((2, 1), (4, 3)); copy = true)
            term += op′
        end
    end
    return term
end

"""
Get coordinates of sites in the 3-site triangular cluster
used in Trotter evolution with next-nearest neighbor gates,
with southwest corner at `[row, col]`.
```
    NORTHWEST   NORTHEAST
        2---1   3---2
        |           |
        3           1

        1           3
        |           |
        2---3   1---2
    SOUTHWEST   SOUTHEAST
```
"""
function _nnn_cluster_sites(dir::Int, row::Int, col::Int)
    @assert 1 <= dir <= 4
    return if dir == NORTHWEST
        map(CartesianIndex, [(row - 1, col + 1), (row - 1, col), (row, col)])
    elseif dir == NORTHEAST
        map(CartesianIndex, [(row, col + 1), (row - 1, col + 1), (row - 1, col)])
    elseif dir == SOUTHEAST
        map(CartesianIndex, [(row, col), (row, col + 1), (row - 1, col + 1)])
    else # dir == SOUTHWEST
        map(CartesianIndex, [(row - 1, col), (row, col), (row, col + 1)])
    end
end

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
        gate::AbstractTensorMap{T, S, N, N}, trunc = trunctol(; atol = MPSKit.Defaults.tol)
    ) where {T <: Number, S <: ElementarySpace, N}
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

function _trotterize_1site!(gates::Vector, H::LocalOperator, dt::Number; atol::Real)
    for site in CartesianIndices(size(H))
        gate = _get_site_term(H, site)
        (norm(gate) <= atol) && continue
        push!(gates, [site] => exp(-dt * gate))
    end
    return gates
end

function _trotterize_nn2site!(gates::Vector, H::LocalOperator, dt::Number; atol::Real, force_mpo::Bool = false)
    Nr, Nc = size(H)
    T = scalartype(H)
    for (d, c, r) in Iterators.product(1:2, 1:Nc, 1:Nr)
        site1 = CartesianIndex(r, c)
        site2 = (d == 1) ? CartesianIndex(r, c + 1) : CartesianIndex(r - 1, c)
        # group with 1-site terms
        s1term = _get_site_term(H, site1)
        unit1 = TensorKit.id(T, space(s1term, 1))
        s2term = _get_site_term(H, site2)
        unit2 = TensorKit.id(T, space(s2term, 1))
        gate = _get_bond_term(H, (site1, site2))
        gate = gate + (s1term ⊗ unit2 + unit1 ⊗ s2term) / 4
        (norm(gate) <= atol) && continue
        gate = exp(-dt * gate)
        force_mpo && (gate = gate_to_mpo(gate))
        push!(gates, [site1, site2] => gate)
    end
    return gates
end

function _trotterize_nnn2site!(gates::Vector, H::LocalOperator, dt::Number; atol::Real)
    Nr, Nc = size(H)
    T = scalartype(H)
    for (c, r, d) in Iterators.product(1:Nc, 1:Nr, 1:4)
        sites = _nnn_cluster_sites(d, r, c)
        gate = _get_bond_term(H, (sites[1], sites[3]))
        (norm(gate) <= atol) && continue
        gate = exp(-(dt / 2) * gate) # account for double counting
        # combine with identity at sites[2]
        r2, c2 = mod1(sites[2][1], Nr), mod1(sites[2][2], Nc)
        id_ = TensorKit.id(T, physicalspace(H)[r2, c2])
        gate = permute(gate ⊗ id_, ((1, 3, 2), (4, 6, 5)))
        push!(gates, sites => gate_to_mpo(gate))
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
    T = scalartype(H)
    atol = eps(real(T))^(3 / 4)
    dt′ = symmetrize_gates ? (dt / 2) : dt
    gates = Vector{Pair{Any, Any}}()

    # TODO: order of gates is fixed for more tight control.
    # Consider directly iterating over H.terms in the future.

    # 1-site gates are only constructed when H only has 1-site terms
    dist == 0 && _trotterize_1site!(gates, H, dt′; atol)
    #= 
    2-site NN gates on bonds [d, r, c], grouped with 1-site terms
    - d = 1: horizontal bond ((r, c), (r, c+1))
    - d = 2:   vertical bond ((r, c), (r-1, c))
    =#
    dist >= 1 && _trotterize_nn2site!(gates, H, dt′; atol, force_mpo)
    #= 
    2-site NNN gates converted to 3-site MPOs on triangular clusters [d, r, c]
    - d = 1 (NORTHWEST), ..., 4 (SOUTHWEST)
    =#
    dist >= 2 && _trotterize_nnn2site!(gates, H, dt′; atol)

    symmetrize_gates && push!(gates, reverse(gates)...)
    return TrotterGates(gates)
end

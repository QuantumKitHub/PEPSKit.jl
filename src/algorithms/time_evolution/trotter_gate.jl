"""
$(TYPEDEF)

Abstract super type for the collection of 2-body Trotter evolution gates.
"""
abstract type TrotterGates end

Base.getindex(gates::TrotterGates, args...) = Base.getindex(gates.gates, args...)

"""
Collection of 1st (nearest) neighbor 2-body Trotter gates.

Before exponentiating, terms in the Hamiltonian are organized as
```
    H = ∑ᵢⱼ (Xᵢⱼ + Yᵢⱼ)
```
where each `Xᵢⱼ` (or `Yᵢⱼ`) acts on a horizontal (or vertical) bond.
The Trotter gates are `exp(-dt * Xᵢⱼ)`, `exp(-dt * Yᵢⱼ)`.
"""
struct TrotterGates1stNeighbor{G} <: TrotterGates
    gates::G
end

function TrotterGates1stNeighbor(H::LocalOperator, dt::Number)
    Nr, Nc = size(H)
    gates = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        # d = 1: horizontal bond; d = 2: vertical bond
        site1 = CartesianIndex(r, c)
        site2 = (d == 1) ?  CartesianIndex(r, c + 1) : CartesianIndex(r - 1, c)
        term = _get_bond_term(H, (site1, site2))
        return exp(-dt * term)
    end
    return TrotterGates1stNeighbor(gates)
end

"""
    is_nearest_neighbour(H::LocalOperator)

Check if an operator `H` contains only nearest neighbor terms.
"""
function is_nearest_neighbour(H::LocalOperator)
    return all(H.terms) do (sites, op)
        return numin(op) == 2 && sum(abs, Tuple(sites[2] - sites[1])) == 1
    end
end

"""
    is_equivalent_bond(bond1::NTuple{2,CartesianIndex{2}}, bond2::NTuple{2,CartesianIndex{2}}, (Nrow, Ncol)::NTuple{2,Int})

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

Get the 2-site term on `bond` in `ham`.
"""
function _get_bond_term(ham::LocalOperator, bond::NTuple{2, CartesianIndex{2}})
    bonds = findall(p -> is_equivalent_bond(p.first, bond, size(ham.lattice)), ham.terms)
    if length(bonds) == 0
        # try reversed site order
        bonds = findall(
            p -> is_equivalent_bond(p.first, reverse(bond), size(ham.lattice)), ham.terms
        )
        if length(bonds) == 1
            return permute(ham.terms[bonds[1]].second, ((2, 1), (4, 3)))
        elseif length(bonds) == 0
            # if term not found, return the zero operator on this bond
            dtype = scalartype(ham)
            r1, c1 = (mod1(bond[1][i], n) for (i, n) in zip(1:2, size(ham)))
            r2, c2 = (mod1(bond[2][i], n) for (i, n) in zip(1:2, size(ham)))
            V1 = physicalspace(ham)[r1, c1]
            V2 = physicalspace(ham)[r2, c2]
            return zeros(dtype, V1 ⊗ V2 ← V1 ⊗ V2)
        else
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        end
    else
        (length(bonds) == 1) ||
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        return ham.terms[bonds[1]].second
    end
end

"""
    _get_se3site_term(ham::LocalOperator, row::Int, col::Int)

Construct the term acting on the southeast 3-site cluster in `ham`.
```
    r-1        g3
                |
                ↓
    r   g1 -←- g2
        c      c+1
```
"""
function _get_se3site_term(ham::LocalOperator, row::Int, col::Int)
    Nr, Nc = size(ham)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    sites = [
        CartesianIndex(row, col),
        CartesianIndex(row, col + 1),
        CartesianIndex(row - 1, col + 1),
    ]
    nb1x = _get_bond_term(ham, (sites[1], sites[2]))
    nb1y = _get_bond_term(ham, (sites[2], sites[3]))
    nb2 = _get_bond_term(ham, (sites[1], sites[3]))
    # identity operator at each site
    units = map(sites) do site
        site_ = CartesianIndex(mod1(site[1], Nr), mod1(site[2], Nc))
        return id(physicalspace(ham)[site_])
    end
    # when iterating through ┘, └, ┌, ┐ clusters in the unit cell,
    # NN / NNN bonds are counted 4 / 2 times, respectively.
    @tensor term[i' j' k'; i j k] :=
        (nb1x[i' j'; i j] * units[3][k' k] + units[1][i'; i] * nb1y[j' k'; j k]) / 4 +
        (nb2[i' k'; i k] * units[2][j'; j]) / 2
    return term
end

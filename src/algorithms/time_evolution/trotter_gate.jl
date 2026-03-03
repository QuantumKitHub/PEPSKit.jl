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
    T = scalartype(H)
    gates = map(Iterators.product(1:2, 1:Nr, 1:Nc)) do (d, r, c)
        # d = 1: horizontal bond; d = 2: vertical bond
        site1 = CartesianIndex(r, c)
        site2 = (d == 1) ? CartesianIndex(r, c + 1) : CartesianIndex(r - 1, c)
        s1term = _get_site_term(H, site1)
        unit1 = TensorKit.id(T, space(s1term, 1))
        s2term = _get_site_term(H, site2)
        unit2 = TensorKit.id(T, space(s2term, 1))
        b_term = _get_bond_term(H, (site1, site2))
        # When iterating through horizontal and vertical bonds in the unit cell,
        # each site / NN-bond is counted 4 / 1 times, respectively.
        term = b_term + (s1term ⊗ unit2 + unit1 ⊗ s2term) / 4
        return exp(-dt * term)
    end
    return TrotterGates1stNeighbor(gates)
end

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
    _get_se3site_term(ham::LocalOperator, row::Int, col::Int)

Construct the term acting on the southeast 3-site cluster in `ham`.
```
    r-1     3
            ↓
    r   1-←-2
        c   c+1
```
"""
function _get_se3site_term(ham::LocalOperator, row::Int, col::Int)
    Nr, Nc = size(ham)
    T = scalartype(ham)
    @assert 1 <= row <= Nr && 1 <= col <= Nc
    sites = [
        CartesianIndex(row, col),
        CartesianIndex(row, col + 1),
        CartesianIndex(row - 1, col + 1),
    ]
    ss = map(sites) do site
        _get_site_term(ham, site)
    end
    nb1x = _get_bond_term(ham, (sites[1], sites[2]))
    nb1y = _get_bond_term(ham, (sites[2], sites[3]))
    nb2 = _get_bond_term(ham, (sites[1], sites[3]))
    # identity operator at each site
    units = map(sites) do site
        site_ = CartesianIndex(mod1(site[1], Nr), mod1(site[2], Nc))
        return id(T, physicalspace(ham)[site_])
    end
    # When iterating through ┘, └, ┌, ┐ clusters in the unit cell,
    # each site / NN-bond / NNN-bond is counted 12 / 4 / 2 times, respectively.
    term_site = (
        ss[1] ⊗ units[2] ⊗ units[3] +
            units[1] ⊗ ss[2] ⊗ units[3] +
            units[1] ⊗ units[2] ⊗ ss[3]
    ) / 12
    @tensor term_nb1[i' j' k'; i j k] :=
        (nb1x[i' j'; i j] * units[3][k' k] + units[1][i'; i] * nb1y[j' k'; j k]) / 4
    @tensor term_nb2[i' j' k'; i j k] := (nb2[i' k'; i k] * units[2][j'; j]) / 2
    return term_site + term_nb1 + term_nb2
end

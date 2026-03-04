"""
Collection of nearest neighbor 2-body Trotter gates.

Before exponentiating, terms in the Hamiltonian are organized as
```
    H = ∑ᵢⱼ (Xᵢⱼ + Yᵢⱼ)
```
where each `Xᵢⱼ` (or `Yᵢⱼ`) acts on a horizontal (or vertical) bond.
The Trotter gates are `exp(-dt * Xᵢⱼ)`, `exp(-dt * Yᵢⱼ)`.
"""
struct TrotterNNGates{G} <: TrotterNNGates
    gates::G
end
Base.getindex(gates::TrotterNNGates, args...) = Base.getindex(gates.gates, args...)

function TrotterNNGates(H::LocalOperator, dt::Number)
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
    return TrotterNNGates(gates)
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

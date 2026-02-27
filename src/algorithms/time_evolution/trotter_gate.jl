"""
    get_expham(H::LocalOperator, dt::Number)

Compute `exp(-dt * op)` for each term `op` in `H`,
and combine them into a new LocalOperator.
Each `op` in `H` must be a single `TensorMap`.
"""
function get_expham(H::LocalOperator, dt::Number)
    return LocalOperator(
        physicalspace(H), (sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
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
    get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})

Get the term of a 2-site gate acting on a certain bond.
Input `gate` should only include one term for each nearest neighbor bond.
"""
function get_gateterm(gate::LocalOperator, bond::NTuple{2, CartesianIndex{2}})
    bonds = findall(p -> is_equivalent_bond(p.first, bond, size(gate.lattice)), gate.terms)
    if length(bonds) == 0
        # try reversed site order
        bonds = findall(
            p -> is_equivalent_bond(p.first, reverse(bond), size(gate.lattice)), gate.terms
        )
        if length(bonds) == 1
            return permute(gate.terms[bonds[1]].second, ((2, 1), (4, 3)))
        elseif length(bonds) == 0
            # if term not found, return the zero operator on this bond
            dtype = scalartype(gate)
            r1, c1 = (mod1(bond[1][i], n) for (i, n) in zip(1:2, size(gate)))
            r2, c2 = (mod1(bond[2][i], n) for (i, n) in zip(1:2, size(gate)))
            V1 = physicalspace(gate)[r1, c1]
            V2 = physicalspace(gate)[r2, c2]
            return zeros(dtype, V1 ⊗ V2 ← V1 ⊗ V2)
        else
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        end
    else
        (length(bonds) == 1) ||
            error("There are multiple terms in `gate` corresponding to the bond $(bond).")
        return gate.terms[bonds[1]].second
    end
end

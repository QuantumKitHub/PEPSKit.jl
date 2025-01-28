"""
    get_gate(dt::Float64, H::LocalOperator)

Compute `exp(-dt * H)` from the nearest neighbor Hamiltonian `H`.
"""
function get_gate(dt::Float64, H::LocalOperator)
    @assert all([
        numin(op) == 2 && norm(Tuple(terms[2] - terms[1])) == 1.0 for (terms, op) in H.terms
    ]) "Only nearest-neighbour terms allowed"
    return LocalOperator(
        H.lattice, Tuple(sites => exp(-dt * op) for (sites, op) in H.terms)...
    )
end

"""
    is_equivalent(bond1::NTuple{2,CartesianIndex{2}}, bond2::NTuple{2,CartesianIndex{2}}, (Nrow, Ncol)::NTuple{2,Int})

Check if two 2-site bonds are related by a (periodic) lattice translation.
"""
function is_equivalent(
    bond1::NTuple{2,CartesianIndex{2}},
    bond2::NTuple{2,CartesianIndex{2}},
    (Nrow, Ncol)::NTuple{2,Int},
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
function get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})
    label = findall(p -> is_equivalent(p.first, bond, size(gate.lattice)), gate.terms)
    if length(label) == 0
        # try reversed site order
        label = findall(
            p -> is_equivalent(p.first, reverse(bond), size(gate.lattice)), gate.terms
        )
        @assert length(label) == 1
        return permute(gate.terms[label[1]].second, ((2, 1), (4, 3)))
    else
        @assert length(label) == 1
        return gate.terms[label[1]].second
    end
end

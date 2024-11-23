"""
Convert Hamiltonian `H` with nearest neighbor terms to `exp(-dt * H)`
"""
function get_gate(dt::Float64, H::LocalOperator)
    return LocalOperator(H.lattice, Tuple(ind => exp(-dt * op) for (ind, op) in H.terms)...)
end

"""
Get the term of a 2-site gate acting on a certain bond.
Input `gate` should only include one term for each nearest neighbor bond.
"""
function get_gateterm(gate::LocalOperator, bond::NTuple{2,CartesianIndex{2}})
    label = findall(p -> p.first == bond, gate.terms)
    if length(label) == 0
        # try reversed site order
        label = findall(p -> p.first == reverse(bond), gate.terms)
        @assert length(label) == 1
        return permute(gate.terms[label[1]].second, ((2, 1), (4, 3)))
    else
        @assert length(label) == 1
        return gate.terms[label[1]].second
    end
end

#
# Observables evaluated using a symmetric boundary MPS environment
#

"""
    network_value(network::InfiniteSquareNetwork, env::SymmetricBoundaryMPSEnv)

Return the value (per unit cell) of a contractible network contracted using a symmetric
boundary MPS environment.
"""
function network_value(network::InfiniteSquareNetwork, env::SymmetricBoundaryMPSEnv)
    size(network) == (1, 1) ||
        throw(ArgumentError("symmetric boundary MPS environments require a single-site unit cell"))
    AC = get_AC(env)
    AC´ = PEPS_AC_Hamiltonian(env.GL, network[1, 1], env.GR) * AC
    return dot(AC, AC´)
end
function network_value(state, env::SymmetricBoundaryMPSEnv)
    return network_value(InfiniteSquareNetwork(state), env)
end

function LinearAlgebra.norm(peps::InfinitePEPS, env::SymmetricBoundaryMPSEnv)
    return network_value(InfiniteSquareNetwork(peps), env)
end

## Partition function tensor insertions

function contract_local_tensor(
        ::Union{CartesianIndex{2}, Tuple{Int, Int}}, O::PartitionFunctionTensor,
        env::SymmetricBoundaryMPSEnv,
    )
    # the index is irrelevant here: symmetric boundary MPS environments are single-site
    return _contract_site(get_AC(env), env.GL, env.GR, O)
end

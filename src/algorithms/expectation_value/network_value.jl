# Network values
# --------------

"""
    network_value(network::InfiniteSquareNetwork, env::CTMRGEnv)

Return the value (per unit cell) of a given contractible network contracted using a given
CTMRG environment.
"""
function network_value(network::InfiniteSquareNetwork, env::AbstractCTMRGEnv, allocator; scheduler = Defaults.scheduler[])
    return tmapreduce(*, Iterators.product(axes(network)...); scheduler) do (r, c)
        return _contract_site((r, c), network, env, allocator) * _contract_corners((r, c), env, allocator) /
            _contract_vertical_edges((r, c), env, allocator) / _contract_horizontal_edges((r, c), env, allocator)
    end
end
network_value(state, env::AbstractCTMRGEnv, allocator; kwargs...) = network_value(InfiniteSquareNetwork(state), env, allocator; kwargs...)

function LinearAlgebra.norm(peps::InfinitePEPS, env::AbstractCTMRGEnv; scheduler = Defaults.scheduler[], allocator = default_allocator(peps, scheduler))
    return network_value(InfiniteSquareNetwork(peps), env, allocator; scheduler)
end

# Local tensor insertions
# -----------------------

"""
    contract_local_tensor(inds, O::PFTensor, env, allocator)

Contract a local tensor `O` inserted into a partition function `pf` at position `inds`,
using the environment `env`.
"""
function contract_local_tensor(
        inds::Tuple{Int, Int}, O::PFTensor, env::AbstractCTMRGEnv{C, <:CTMRG_PF_EdgeTensor}, allocator
    ) where {C}
    r, c = inds
    return _contract_site(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r + 1, c + 1),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, NORTH, r - 1, c), edge(env, EAST, r, c + 1),
        edge(env, SOUTH, r + 1, c), edge(env, WEST, r, c - 1),
        O, allocator
    )
end

"""
    contract_local_tensor(inds, O::PEPOTensor, network, env, allocator)

Contract a local tensor `O` inserted into the PEPO of a given `network` at position `inds`,
using the environment `env`.
"""
function contract_local_tensor(
        ind::Tuple{Int, Int, Int},
        O::PEPOTensor,
        network::InfiniteSquareNetwork{<:PEPOSandwich},
        env::AbstractCTMRGEnv,
        allocator,
    )
    r, c, h = ind
    sandwich´ = Base.setindex(network[r, c], O, h + 2)
    return _contract_site(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r + 1, c + 1),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, NORTH, r - 1, c), edge(env, EAST, r, c + 1),
        edge(env, SOUTH, r + 1, c), edge(env, WEST, r, c - 1),
        sandwich´,
        allocator,
    )
end

function contract_local_tensor(inds::CartesianIndex, O::AbstractTensorMap, env::AbstractCTMRGEnv, allocator)
    return contract_local_tensor(Tuple(inds), O, env, allocator)
end

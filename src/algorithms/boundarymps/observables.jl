#
# Observables evaluated using boundary MPS environments
#

"""
    network_value(network::InfiniteSquareNetwork, env::C4vBoundaryMPSEnv)
    network_value(network::InfiniteSquareNetwork, env::BoundaryMPSEnv)

Return the value (per unit cell) of a contractible network contracted using a boundary MPS
environment.

For a C4v environment, this is the ratio of the eigenvalues of the effective site and
bond operators. For an asymmetric environment, it is the product over the rows `r` of the
network of the ratio ``⟨t_{r-1}| T_r |b_{r+1}⟩ / ⟨t_r | b_{r+1}⟩`` of the leading eigenvalues of
the mixed transfer matrices through the whole row with and without the network, where
``t_r`` is the top MPS below row `r` and ``b_{r+1}`` the bottom MPS above row `r + 1`. The
first is the product of the eigenvalues of the mixed transfer matrices through the sites of
the row, which follow from the normalized mixed environments.
"""
function network_value(network::InfiniteSquareNetwork, env::C4vBoundaryMPSEnv)
    size(network) == (1, 1) ||
        throw(ArgumentError("C4v boundary MPS environments require a single-site unit cell"))
    AC = _center_tensor(env)
    λAC = dot(AC, ∂AC(AC, env.GL, network[1, 1], env.GR))
    λC = dot(env.C, ∂C(env.C, env.GL, env.GR))
    return λAC / λC
end
function network_value(network::InfiniteSquareNetwork, env::BoundaryMPSEnv)
    size(network) == size(env) ||
        throw(ArgumentError("the environment and the network have different unit cells"))
    rows, cols = size(network)
    return prod(1:rows) do r
        λ_network = prod(1:cols) do c
            GL′ = edge_transfer_left(
                _at(env.GLm, r, c - 1), network[r, c], _at(env.ALt, r - 1, c), _at(env.ARb, r + 1, c)
            )
            return dot(env.GLm[r, c], GL′) / dot(env.GLm[r, c], env.GLm[r, c])
        end
        At, Ab = env.ALt[r, 1], _at(env.ARb, r + 1, 1)
        x₀ = randn(scalartype(At), right_virtualspace(Ab) ← left_virtualspace(At))
        λs, = eigsolve(x₀, 1, :LM) do x
            return _through(x, Tuple(1:cols)) do y, c
                return edge_transfer_left(y, env.ALt[r, c], _at(env.ARb, r + 1, c))
            end
        end
        return λ_network / first(λs)
    end
end
network_value(state, env::BoundaryMPSEnvs) = network_value(InfiniteSquareNetwork(state), env)

function LinearAlgebra.norm(peps::InfinitePEPS, env::BoundaryMPSEnvs)
    return network_value(InfiniteSquareNetwork(peps), env)
end

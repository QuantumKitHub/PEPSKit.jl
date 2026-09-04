# Expectation value of a LocalOperator
# ------------------------------------

"""
    expectation_value(state, O::LocalOperator, env)
    expectation_value(bra, O::LocalOperator, ket, env)

Compute the expectation value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state) of a [`LocalOperator`](@ref) `O`.
This can be done either for a PEPS, or alternatively for a density matrix PEPO.
In the latter case the first signature corresponds to a single layer PEPO contraction, while
the second signature yields a bilayer contraction instead.
"""
function MPSKit.expectation_value(
        bra::S, O::LocalOperator, ket::S, env
    ) where {S <: InfiniteState}
    checklattice(bra, O, ket)
    term_vals = dtmap(collect(O.terms)) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        return local_expectation_value(inds, bra, operator, ket, env)
    end
    return sum(term_vals)
end
MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env) = expectation_value(peps, O, peps, env)
function MPSKit.expectation_value(state::InfinitePEPO, O::LocalOperator, env)
    checklattice(state, O)
    term_vals = dtmap(collect(O.terms)) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        return local_expectation_value(inds, state, operator, env)
    end
    return sum(term_vals)
end


# Expectation value of an individual local term
# ---------------------------------------------

"""
    local_expectation_value(inds, bra, operator, ket, env)
    local_expectation_value(inds, state, operator, env)

Compute the contribution of a single term of a [`LocalOperator`](@ref) to the expectation
value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ or tr(O * state) / tr(state), where `operator` is the local
term acting on the sites `inds`.

The implementation is overloaded based on the type of operator to be evaluated
"""
function local_expectation_value end

# AbstractTensorMap evaluation goes through reduced density matrix
function local_expectation_value(inds, bra, operator::AbstractTensorMap, ket, env)
    ρ = reduced_densitymatrix(inds, ket, bra, env)
    return trmul(operator, ρ)
end
function local_expectation_value(inds, state, operator::AbstractTensorMap, env)
    ρ = reduced_densitymatrix(inds, state, env)
    return trmul(operator, ρ)
end


# Local patch contractions
# ------------------------

@doc """
    reduced_densitymatrix(inds, ket::InfinitePEPS, bra::InfinitePEPS = ket, env)
    reduced_densitymatrix(inds, ket::InfinitePEPO, bra::InfinitePEPO, env)
    reduced_densitymatrix(inds, state::InfinitePEPO, env)

Construct the reduced density matrix `ρ` of `|ket⟩⟨bra|`, where both `ket` and `bra`
correspond to either a PEPS or a PEPO representing a PEPS with ancillary legs.
Alternatively, construct the reduced density matrix `ρ` of a mixed state specified by the 
density matrix PEPO `state`. The reduced density matrix is contracted around the open
indices `inds` using the environment `env`, and is normalized such that `str(ρ) = 1`.

See also [`str`](@ref).
""" reduced_densitymatrix

# PEPS case has fast-path specializations
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env
    )
    length(inds) == 1 && return reduced_densitymatrix1x1(only(inds), ket, bra, env)

    if length(inds) == 2
        if inds[2] - inds[1] == CartesianIndex(1, 0)
            return reduced_densitymatrix2x1(inds[1], ket, bra, env)
        elseif inds[2] - inds[1] == CartesianIndex(0, 1)
            return reduced_densitymatrix1x2(inds[1], ket, bra, env)
        end
    end

    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, (ket, bra), env)
end
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, state::InfinitePEPO, env
    )
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, state, env)
end
function reduced_densitymatrix(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_densitymatrix(static_inds, (ket, bra), env)
end
reduced_densitymatrix(inds, ket::InfinitePEPS, env) =
    reduced_densitymatrix(inds, ket, ket, env)

# handle deprecations of Tuple inds specifications
Base.@deprecate(
    reduced_densitymatrix(
        inds::NTuple{N, CartesianIndex{2}}, args...
    ) where {N},
    reduced_densitymatrix(collect(inds), args...)
)
Base.@deprecate(
    reduced_densitymatrix(
        inds::NTuple{N, Tuple{Int, Int}}, args...
    ) where {N},
    reduced_densitymatrix(collect(CartesianIndex.(inds)), args...)
)

"""
    contract_local_operator(inds, O, ket::InfinitePEPS, bra::InfinitePEPS = ket, env)
    contract_local_operator(inds, O, ket::InfinitePEPO, bra::InfinitePEPO, env)
    contract_local_operator(inds, O, state::InfinitePEPO, env)

Contract a local operator `O` between `ket` and `bra` states, computing `⟨bra|O|ket⟩`, where
`ket` and `bra` correspond to either a PEPS or a PEPO representing a PEPS with ancillary
legs. Alternatively, contract a local operator `O` with a density matrix PEPO `state`,
computing `tr(O * state)`. `O` is applied to the open indices `inds`, and the result is
contracted using the environment `env`.
"""
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O,
        ket::InfinitePEPS, bra::InfinitePEPS, env,
    )
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, (ket, bra), env)
end
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O, state::InfinitePEPO, env
    )
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, state, env)
end
function contract_local_operator(
        inds::Vector{CartesianIndex{2}}, O, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_operator(static_inds, O, (ket, bra), env)
end
function contract_local_operator(inds::Vector{Tuple{Int, Int}}, O, args...)
    return contract_local_operator(CartesianIndex.(inds), O, args...)
end

Base.@deprecate(
    contract_local_operator(
        inds::NTuple, args...
    ),
    contract_local_operator(collect(inds), args...)
)

"""
    contract_local_norm(inds, ket::InfinitePEPS, bra::InfinitePEPS = ket, env)
    contract_local_norm(inds, ket::InfinitePEPO, bra::InfinitePEPO, env)
    contract_local_norm(inds, state::InfinitePEPO, env)

Contract a local norm corresponding to the overlap `ket` and `bra` states, computing a patch
of `⟨bra|ket⟩`, where `ket` and `bra` correspond to either a PEPS or a PEPO representing a
PEPS with ancillary legs. Alternatively, contract a local norm patch of a density matrix
PEPO `state`, computing a patch of `tr(state)`. The norm patch is contracted around the open
indices `inds` using the environment `env`.
"""
function contract_local_norm(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPS, bra::InfinitePEPS, env
    )
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, (ket, bra), env)
end
function contract_local_norm(inds::Vector{CartesianIndex{2}}, state::InfinitePEPO, env)
    size(state, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, state, env)
end
function contract_local_norm(
        inds::Vector{CartesianIndex{2}}, ket::InfinitePEPO, bra::InfinitePEPO, env
    )
    size(ket) == size(bra) || throw(DimensionMismatch("incompatible bra and ket dimensions"))
    size(ket, 3) == 1 || throw(DimensionMismatch("only single-layer densitymatrices are supported"))
    static_inds = Tuple(Val.(inds))
    return _contract_local_norm(static_inds, (ket, bra), env)
end
function contract_local_norm(inds::Vector{Tuple{Int, Int}}, args...)
    return contract_local_norm(CartesianIndex.(inds), args...)
end

Base.@deprecate(
    contract_local_norm(inds::NTuple, ket::InfinitePEPS, bra::InfinitePEPS, env),
    contract_local_norm(collect(inds), ket, bra, env)
)


# Expectation value of a local partition function tensor
# ------------------------------------------------------

"""
    expectation_value(pf::InfinitePartitionFunction, inds => O, env::CTMRGEnv)

Compute the expectation value corresponding to inserting a local tensor(s) `O` at
position `inds` in the partition function `pf` and contracting the whole using a given CTMRG
environment `env`.

Here `inds` can be specified as either a `Tuple{Int,Int}` or a `CartesianIndex{2}`, and `O`
should be a rank-4 tensor conforming to the [`PartitionFunctionTensor`](@ref) indexing
convention.
"""
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction,
        op::Pair{CartesianIndex{2}, <:AbstractTensorMap{T, S, 2, 2}},
        env,
    ) where {T, S}
    return contract_local_tensor(op[1], op[2], env) /
        contract_local_tensor(op[1], pf[op[1]], env)
end
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction, op::Pair{Tuple{Int, Int}}, env
    )
    return expectation_value(pf, CartesianIndex(op[1]) => op[2], env)
end

# Network values
# --------------

"""
    network_value(network::InfiniteSquareNetwork, env::CTMRGEnv)

Return the value (per unit cell) of a given contractible network contracted using a given
CTMRG environment.
"""
function network_value(network::InfiniteSquareNetwork, env::CTMRGEnv)
    return prod(Iterators.product(axes(network)...)) do (r, c)
        return _contract_site((r, c), network, env) * _contract_corners((r, c), env) /
            _contract_vertical_edges((r, c), env) / _contract_horizontal_edges((r, c), env)
    end
end
network_value(state, env::CTMRGEnv) = network_value(InfiniteSquareNetwork(state), env)

"""
    _contract_site(ind::Tuple{Int,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv)

Contract around a single site `ind` of a square network using a given CTMRG environment.
"""
function _contract_site(ind::Tuple{Int, Int}, network::InfiniteSquareNetwork, env::CTMRGEnv)
    r, c = ind
    return _contract_site(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r + 1, c + 1),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, NORTH, r - 1, c), edge(env, EAST, r, c + 1),
        edge(env, SOUTH, r + 1, c), edge(env, WEST, r, c - 1),
        network[r, c],
    )
end

"""
    _contract_corners(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract all corners around the south-east at position `ind` of the CTMRG
environment `env`.
"""
function _contract_corners(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    return _contract_corners(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c),
        corner(env, SOUTHEAST, r, c),
        corner(env, SOUTHWEST, r, c - 1),
    )
end

"""
    _contract_vertical_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the vertical edges and corners around the east edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_vertical_edges(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    return _contract_vertical_edges(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c),
        corner(env, SOUTHEAST, r + 1, c),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, EAST, r, c),
        edge(env, WEST, r, c - 1),
    )
end

"""
    _contract_horizontal_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the horizontal edges and corners around the south edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_horizontal_edges(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    return _contract_horizontal_edges(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r, c + 1),
        corner(env, SOUTHWEST, r, c - 1),
        edge(env, NORTH, r - 1, c),
        edge(env, SOUTH, r, c),
    )
end

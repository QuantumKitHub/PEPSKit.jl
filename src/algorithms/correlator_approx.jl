# Approximate finite-window two-site correlators
# ----------------------------------------------

"""
$(SIGNATURES)

Approximately measure a dense two-site operator on one or more ordered pairs of sites in a single-layer PEPO.
The first and second operator legs act on the first and second sites of each pair, respectively.
Multiple pairs are evaluated in one shared window and reuse open-string boundary contractions.
Ordered pairs in a batched call must be unique.
"""
function correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap, bond::Tuple, env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1, direction::Symbol = :auto,
    )
    return only(
        correlator_approx(ρ, op, [bond], env; trunc, maxiter, direction)
    )
end

function correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap, bonds::AbstractVector,
        env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1, direction::Symbol = :auto,
    )
    bonds′ = _approx_twosite_bonds(bonds)
    return _correlator_approx(
        ρ, op, bonds′, env,
        WindowApprox(Zipup(; trunc), _approx_dmrg(maxiter)), direction
    )
end

"""
Validate and regularize a nonempty collection of ordered two-site bonds to Cartesian indices.
"""
function _approx_twosite_bonds(bonds)
    isempty(bonds) && throw(ArgumentError("correlator_approx requires at least one bond"))
    bonds′ = NTuple{2, CartesianIndex{2}}[]
    sizehint!(bonds′, length(bonds))
    for bond in bonds
        length(bond) == 2 || throw(ArgumentError("each bond should contain two sites"))
        first_site = _mpo_observable_site(bond[1])
        second_site = _mpo_observable_site(bond[2])
        first_site != second_site ||
            throw(ArgumentError("the sites of a bond should be distinct"))
        push!(bonds′, (first_site, second_site))
    end
    allunique(bonds′) || throw(ArgumentError("bonds should be unique"))
    return bonds′
end

"""
Return the row and column ranges enclosing every endpoint in a collection of bonds.
"""
function _window_ranges(bonds::Vector{NTuple{2, CartesianIndex{2}}})
    return _window_ranges(Iterators.flatten(bonds))
end

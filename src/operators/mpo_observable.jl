"""
$(TYPEDEF)

An open-boundary MPO embedded along a non-self-intersecting nearest-neighbor path on the
square lattice. The tensor `mpo[k]` acts on `sites[k]`; `path` also contains intermediate
sites that only carry the MPO string.

The first and last MPO tensors use the reduced endpoint partitions `(1, 2)` and `(2, 1)`;
all intermediate tensors use the standard `(2, 2)` MPO partition.
"""
struct MPOObservable{M}
    sites::Vector{CartesianIndex{2}}
    path::Vector{CartesianIndex{2}}
    mpo::Vector{M}

    function MPOObservable(
            sites::Vector{CartesianIndex{2}}, path::Vector{CartesianIndex{2}},
            mpo::Vector{M},
        ) where {M}
        _validate_mpo_observable(sites, path, mpo)
        return new{M}(sites, path, mpo)
    end
end

"""
Construct an MPO observable from ordered operator sites and MPO tensors. The tensor `mpo[k]`
acts on `sites[k]`; consecutive sites are connected by horizontal-first shortest paths.
"""
function MPOObservable(sites, mpo)
    sites′ = CartesianIndex{2}[_mpo_observable_site(site) for site in sites]
    mpo′ = collect(mpo)
    path = _route_mpo_observable(sites′)
    return MPOObservable(sites′, path, mpo′)
end

"""
Construct an MPO observable from a dense operator. Operator sites are ordered in the same
way as `LocalOperator` terms, and consecutive sites are connected by horizontal-first
shortest paths. The lattice is a periodically indexed matrix of elementary physical spaces,
as for `LocalOperator`.
"""
function MPOObservable(
        sites, op::AbstractTensorMap, lattice::Matrix{<:ElementarySpace};
        trunc = trunctol(; atol = MPSKit.Defaults.tol),
    )
    sites′ = CartesianIndex{2}[_mpo_observable_site(site) for site in sites]
    length(sites′) >= 2 || throw(ArgumentError("an MPO observable requires at least two sites"))
    allunique(sites′) || throw(ArgumentError("operator sites should be unique"))
    length(sites′) == numin(op) == numout(op) ||
        throw(ArgumentError("number of operator legs should match the number of sites"))

    Nr, Nc = size(lattice)
    sites′, op′ = _sort_op_sites(sites′, op)
    for (i, site) in enumerate(sites′)
        V = lattice[mod1(site[1], Nr), mod1(site[2], Nc)]
        V == domain(op′)[i] == codomain(op′)[i] ||
            throw(SpaceMismatch("operator physical space does not match lattice site $site"))
    end

    mpo = gate_to_mpo(op′; trunc)
    length(mpo) == length(sites′) ||
        throw(ArgumentError("expected an MPO decomposition matching the number of sites"))
    return MPOObservable(sites′, mpo)
end

"""
Normalize a supported lattice-site representation to a two-dimensional Cartesian index.
"""
_mpo_observable_site(site::CartesianIndex{2}) = site
_mpo_observable_site(site::Tuple{Int, Int}) = CartesianIndex(site)
_mpo_observable_site(site) =
    throw(ArgumentError("MPO sites should be CartesianIndex{2} or (row, col) tuples"))

"""
Validate the topology, tensor partitions, physical placement, and adjacent string spaces of
an already routed open-boundary MPO observable.
"""
function _validate_mpo_observable(sites, path, mpo)
    length(sites) >= 2 || throw(ArgumentError("an MPO observable requires at least two sites"))
    length(sites) == length(mpo) ||
        throw(ArgumentError("the MPO should contain one tensor for every operator site"))
    allunique(sites) || throw(ArgumentError("operator sites should be unique"))
    allunique(path) || throw(ArgumentError("the MPO path should not intersect itself"))
    all(op -> op isa AbstractTensorMap, mpo) ||
        throw(ArgumentError("all MPO entries should be tensor maps"))

    for (from, to) in zip(path, Iterators.drop(path, 1))
        _step_direction(from, to)
    end
    path_positions = indexin(sites, path)
    all(!isnothing, path_positions) && issorted(path_positions) ||
        throw(ArgumentError("the MPO path should contain operator sites in MPO order"))
    first(path_positions) == 1 && last(path_positions) == length(path) ||
        throw(ArgumentError("the MPO path should start and end at operator sites"))

    numout(first(mpo)) == 1 && numin(first(mpo)) == 2 ||
        throw(ArgumentError("the first MPO tensor should have partition (1, 2)"))
    numout(last(mpo)) == 2 && numin(last(mpo)) == 1 ||
        throw(ArgumentError("the last MPO tensor should have partition (2, 1)"))
    for op in @view mpo[2:(end - 1)]
        numout(op) == 2 && numin(op) == 2 ||
            throw(ArgumentError("middle MPO tensors should have partition (2, 2)"))
    end
    for k in 1:(length(mpo) - 1)
        _mpo_right_stringspace(mpo[k])' == space(mpo[k + 1], 1) ||
            throw(SpaceMismatch("incompatible MPO string spaces between sites $k and $(k + 1)"))
    end
    return nothing
end

function VI.scalartype(observable::MPOObservable)
    return promote_type((scalartype(op) for op in observable.mpo)...)
end

"""
Connect ordered operator sites by horizontal-first shortest paths while rejecting crossings
and routes that pass through later operator sites.
"""
function _route_mpo_observable(sites)
    length(sites) >= 2 || throw(ArgumentError("an MPO observable requires at least two sites"))
    allunique(sites) || throw(ArgumentError("operator sites should be unique"))

    path = CartesianIndex{2}[first(sites)]
    measured = Set(sites)
    visited = Set(path)

    for k in 1:(length(sites) - 1)
        segment = _l_path(sites[k], sites[k + 1])

        for site in @view segment[2:(end - 1)]
            site in measured &&
                throw(ArgumentError("MPO path passes through another operator site"))
            site in visited &&
                throw(ArgumentError("MPO path intersects itself at site $site"))
            push!(path, site)
            push!(visited, site)
        end

        site = last(segment)
        site in visited && throw(ArgumentError("MPO path intersects itself at site $site"))
        push!(path, site)
        push!(visited, site)
    end
    return path
end

"""
Return the right MPO string space in the local tensor's stored leg orientation.
"""
_mpo_right_stringspace(op) = space(op, numind(op))

"""
Build a deterministic shortest nearest-neighbor path between two sites. The path traverses
the horizontal separation first and then the vertical separation.
"""
function _l_path(start::CartesianIndex{2}, stop::CartesianIndex{2})
    start == stop && throw(ArgumentError("MPO path sites should be unique"))

    path = CartesianIndex{2}[start]
    row, col = Tuple(start)
    stoprow, stopcol = Tuple(stop)

    while col != stopcol
        col += sign(stopcol - col)
        push!(path, CartesianIndex(row, col))
    end
    while row != stoprow
        row += sign(stoprow - row)
        push!(path, CartesianIndex(row, col))
    end
    return path
end

"""
Return the cardinal direction of a nearest-neighbor step from one site to another. Throw an
error when the sites are not nearest neighbors.
"""
function _step_direction(from::CartesianIndex{2}, to::CartesianIndex{2})
    delta = to - from
    delta == CartesianIndex(0, 1) && return :east
    delta == CartesianIndex(0, -1) && return :west
    delta == CartesianIndex(1, 0) && return :south
    delta == CartesianIndex(-1, 0) && return :north
    throw(ArgumentError("MPO path should use nearest-neighbor steps"))
end

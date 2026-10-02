"""
A `path => mpo` pair with one MPO tensor per site of a non-self-intersecting nearest-neighbor lattice path.
Intermediate sites carry inserted braiding tensors that propagate the MPO bond.
These properties are checked by routing validation, not enforced by this alias.
"""
const RoutedMPOTerm = Pair{<:LatticePath, <:MPOTerm}

"""
Convert a dense term to an exact MPO in column-snake order and expand its nearest-neighbor path.

Visit occupied columns from left to right, alternating between increasing and decreasing row indices within each column.
Permute the operator's physical legs into this order before decomposing it into an MPO.
For example, the six operator sites below are visited as `A → B → C → D → E → F`, with row indices increasing downward:

```text
       col 1   col 2   col 3
row 1    A       + --→-- E
         ↓       ↑       ↓
row 2    +       D       +
         ↓       ↑       ↓
row 3    B       +       F
         ↓       ↑
row 4    + --→-- C
```

The path is constructed successively between these ordered sites:

1. `A → B`: descend from `(1, 1)` through `(2, 1)` to `(3, 1)`.
2. `B → C`: connect the column bottoms at row `max(3, 4) = 4`, passing through `(4, 1)` before reaching `(4, 2)`.
3. `C → D`: ascend through `(3, 2)` to `(2, 2)`.
4. `D → E`: connect the column tops at row `min(2, 1) = 1`, passing through `(1, 2)` before reaching `(1, 3)`.
5. `E → F`: descend through `(2, 3)` to `(3, 3)`.

The `+` sites carry inserted braiding tensors that propagate the MPO bond without an additional physical operator.
Connecting successive columns at their outermost endpoint row avoids retracing the path: a horizontal-first connection from `B` to `C` would visit `(3, 2)`, which the later `C → D` segment needs.
"""
function _route_mpo_term(
        sites::LatticePath, op::AbstractTensorMap,
        lattice::Matrix{<:ElementarySpace},
    )
    isempty(sites) && throw(ArgumentError("an operator term requires at least one site"))
    allunique(sites) || throw(ArgumentError("operator sites should be unique"))
    length(sites) == numout(op) == numin(op) ||
        throw(ArgumentError("number of operator legs should match the number of sites"))
    Nr, Nc = size(lattice)
    for (k, site) in enumerate(sites)
        V = lattice[mod1(site[1], Nr), mod1(site[2], Nc)]
        V == codomain(op)[k] == domain(op)[k] ||
            throw(SpaceMismatch("operator physical space does not match lattice site $site"))
    end
    length(sites) == 1 && return copy(sites) => [op]

    columns = sort!(unique(site[2] for site in sites))
    permutation = Int[]
    for (k, col) in enumerate(columns)
        indices = findall(site -> site[2] == col, sites)
        sort!(indices; by = i -> sites[i][1], rev = iseven(k))
        append!(permutation, indices)
    end
    ordered_sites = sites[permutation]
    N = length(sites)
    ordered_op = permute(op, (Tuple(permutation), Tuple(permutation .+ N)))
    mpo = gate_to_mpo(ordered_op; trunc = notrunc())
    path = _dense_mpo_path(ordered_sites)
    return _expand_mpo_path(ordered_sites, mpo, path, lattice)
end

"""
Preserve an explicit MPO's factor order while connecting its sites with simple nearest-neighbor paths.
"""
function _route_mpo_term(
        sites::LatticePath, mpo::MPOTerm,
        lattice::Matrix{<:ElementarySpace},
    )
    _validate_mpo_term(sites, mpo, lattice)
    path = _ordered_mpo_path(sites)
    return _expand_mpo_path(sites, mpo, path, lattice)
end

"""
Connect snake-ordered columns beyond their bottom or top endpoints to avoid retracing sparse columns.
"""
function _dense_mpo_path(sites::LatticePath)
    path = CartesianIndex{2}[first(sites)]
    column_index = 1
    for (from, to) in zip(sites, Iterators.drop(sites, 1))
        if from[2] == to[2]
            append!(path, Iterators.drop(_l_path(from, to), 1))
        else
            row = isodd(column_index) ? max(from[1], to[1]) : min(from[1], to[1])
            corners = (CartesianIndex(row, from[2]), CartesianIndex(row, to[2]), to)
            for corner in corners
                last(path) == corner && continue
                append!(path, Iterators.drop(_l_path(last(path), corner), 1))
            end
            column_index += 1
        end
    end
    return path
end

"""
Route ordered sites with horizontal-first L paths, falling back to vertical-first paths without backtracking.
"""
function _ordered_mpo_path(sites::LatticePath)
    isempty(sites) && throw(ArgumentError("an MPO term requires at least one site"))
    allunique(sites) || throw(ArgumentError("operator sites should be unique"))
    path = CartesianIndex{2}[first(sites)]
    measured = Set(sites)
    visited = Set(path)
    for (from, to) in zip(sites, Iterators.drop(sites, 1))
        segment = _l_path(from, to)
        if !_mpo_segment_is_free(segment, measured, visited)
            segment = _l_path(from, to; horizontal_first = false)
            _mpo_segment_is_free(segment, measured, visited) || throw(
                ArgumentError(
                    "routing this MPO ordering is not implemented: no nonintersecting L path from $from to $to",
                ),
            )
        end
        append!(path, Iterators.drop(segment, 1))
        union!(visited, segment)
    end
    return path
end

"""
Check that a connecting segment neither revisits a site nor crosses another operator site.
"""
function _mpo_segment_is_free(segment, measured::Set, visited::Set)
    any(site -> site in visited, @view segment[2:end]) && return false
    return all(site -> !(site in measured), @view segment[2:(end - 1)])
end

"""
Insert braiding tensors at unmeasured path sites using periodic physical spaces and the neighboring MPO bond.
"""
function _expand_mpo_path(
        sites::LatticePath, mpo::MPOTerm,
        path::LatticePath, lattice::Matrix{<:ElementarySpace},
    )
    _validate_mpo_term(sites, mpo, lattice)
    allunique(path) || throw(ArgumentError("the MPO path should not intersect itself"))
    positions = indexin(sites, path)
    all(!isnothing, positions) && issorted(positions) ||
        throw(ArgumentError("the MPO path should contain operator sites in MPO order"))
    first(positions) == 1 && last(positions) == length(path) ||
        throw(ArgumentError("the MPO path should start and end at operator sites"))
    for (from, to) in zip(path, Iterators.drop(path, 1))
        _step_direction(from, to)
    end

    Nr, Nc = size(lattice)
    k = 1
    expanded_mpo = AbstractTensorMap[]
    for site in path
        if site == sites[k]
            op = mpo[k]
            k += 1
            push!(expanded_mpo, op)
            continue
        end
        previous = mpo[k - 1]
        V = lattice[mod1(site[1], Nr), mod1(site[2], Nc)]
        bond = _mpo_right_stringspace(previous)'
        braid = TensorKit.BraidingTensor{scalartype(previous)}(V, bond)
        push!(expanded_mpo, copy!(similar(previous, scalartype(previous), space(braid)), braid))
    end
    return copy(path) => expanded_mpo
end

"""
Return the right MPO string space in the local tensor's stored leg orientation.
"""
_mpo_right_stringspace(op::AbstractTensorMap) = space(op, numind(op))

"""
Build a shortest nearest-neighbor L path with the chosen horizontal or vertical first step.
"""
function _l_path(
        start::CartesianIndex{2}, stop::CartesianIndex{2}; horizontal_first::Bool = true
    )
    start == stop && throw(ArgumentError("MPO path sites should be unique"))
    path = CartesianIndex{2}[start]
    axes = horizontal_first ? (2, 1) : (1, 2)
    for axis in axes
        while last(path)[axis] != stop[axis]
            step = sign(stop[axis] - last(path)[axis])
            delta = axis == 1 ? CartesianIndex(step, 0) : CartesianIndex(0, step)
            push!(path, last(path) + delta)
        end
    end
    return path
end

"""
Return the cardinal direction of a nearest-neighbor step, rejecting longer or stationary steps.
"""
function _step_direction(from::CartesianIndex{2}, to::CartesianIndex{2})
    delta = to - from
    delta == CartesianIndex(0, 1) && return :east
    delta == CartesianIndex(0, -1) && return :west
    delta == CartesianIndex(1, 0) && return :south
    delta == CartesianIndex(-1, 0) && return :north
    throw(ArgumentError("MPO path should use nearest-neighbor steps"))
end

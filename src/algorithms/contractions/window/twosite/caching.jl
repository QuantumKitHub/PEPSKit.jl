"""
Put 2-site bonds in groups to reuse partial contractions in `correlator_approx`.

For every bond `(first_site, second_site)` in `bonds`:

- After ordering the two sites, `source`/`target` is the first/second site.
- `swapped` is `false` when `source == first_site`, and `true` otherwise.
- Bonds with the same `source` and `swapped` are grouped together.
- In each group, the inner dict records each bond's position in `bonds`.

For example, the ordered bonds

```julia
CI = CartesianIndex
bonds = [
    (CI(1, 1), CI(1, 3)),
    (CI(1, 1), CI(2, 2)),
    (CI(1, 2), CI(2, 2)), # another source site
    (CI(2, 2), CI(1, 1)), # reversed site order
]
```

are grouped as

```julia
(CI(1, 1), false) => Dict(
    CI(1, 3) => 1,
    CI(2, 2) => 2,
)
(CI(1, 1), true) => Dict(CI(2, 2) => 4)
(CI(1, 2), false) => Dict(CI(2, 2) => 3)
```
"""
function _twosite_source_groups(
        bonds::Vector{NTuple{2, CartesianIndex{2}}},
    )
    groups = Dict{
        Tuple{CartesianIndex{2}, Bool},
        Dict{CartesianIndex{2}, Int},
    }()
    for (i, (first_site, second_site)) in enumerate(bonds)
        swapped = !issorted((first_site, second_site); by = Tuple)
        source, target = swapped ? (second_site, first_site) : (first_site, second_site)
        targets = get!(Dict{CartesianIndex{2}, Int}, groups, (source, swapped))
        targets[target] = i
    end
    return groups
end

"""
Group the targets associated with one source by their row coordinate. The returned outer
dictionary maps each target row to a dictionary whose entries retain the original
`target => result_position` mapping.

This lookup lets the source contraction close all targets in the current row together while
propagating a single open MPO string between rows.
"""
function _twosite_targets_by_row(targets::Dict{CartesianIndex{2}, Int})
    targets_by_row = Dict{Int, Dict{CartesianIndex{2}, Int}}()
    for (target, position) in targets
        row_targets = get!(Dict{CartesianIndex{2}, Int}, targets_by_row, target[1])
        row_targets[target] = position
    end
    return targets_by_row
end

"""
Precompute the row MPOs (without observables) and
north/south boundary contractions reused when measuring on
many two-site bonds in one window.

The returned named tuple contains:

- `rowrange` and `colrange`: the coordinate ranges defining the window.
- `row_mpos`: the ordinary finite MPO for each row, including the west and east CTMRG edges.
- `north_prefixes`: `nrows + 1` north boundary MPSs. Entry `k` is above row `k` in the window.
- `south_suffixes`: `nrows + 1` adjointed south boundary MPSs. Entry `k + 1` is below row `k` in the window.
- `norm`: the approximate contraction of the window with no observable inserted.
"""
function _window_row_cache(
        ρ::InfinitePEPO, env::CTMRGEnv,
        rowrange::UnitRange{Int}, colrange::UnitRange{Int}, alg::WindowApprox,
    )
    row_mpos = Dict(
        row => _row_mpo(ρ, nothing, env, row, colrange)
            for row in rowrange
    )
    nrows = length(rowrange)
    north = _north_boundary_mps(env, first(rowrange), colrange)
    south = _south_boundary_mps(env, last(rowrange), colrange)

    north_prefixes = Vector{typeof(north)}(undef, nrows + 1)
    north_prefixes[1] = north
    for (k, row) in enumerate(rowrange)
        north_prefixes[k + 1] = _approximate_window_step(
            row_mpos[row], north_prefixes[k], alg
        )
    end

    south_suffixes = Vector{typeof(south)}(undef, nrows + 1)
    south_suffixes[end] = south
    for (k, row) in Iterators.reverse(enumerate(rowrange))
        W = _adjoint_mpo(row_mpos[row])
        south_suffixes[k] = _approximate_window_step(
            W, south_suffixes[k + 1], alg
        )
    end
    norm = dot(south, north_prefixes[end])
    return (; rowrange, colrange, row_mpos, north_prefixes, south_suffixes, norm)
end

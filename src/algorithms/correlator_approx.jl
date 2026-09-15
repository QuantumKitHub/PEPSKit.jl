# Approximate finite-window two-site correlators
# ----------------------------------------------

"""
$(SIGNATURES)

Approximately measure a dense two-site operator between a fixed first site `i` and second sites `js` in a single-layer PEPO.

- Operator leg 1 acts at `i`, and leg 2 acts at each second site.
- The contraction direction is chosen automatically.
    Row-by-row contraction proceeds southward and requires every `j[1] ≥ i[1]`; column-by-column contraction proceeds westward and requires every `j[2] ≤ i[2]`.
    If both are possible, consider the smallest rectangle containing `i` and all sites in `js`: use rows if it is square or wider than tall, and columns otherwise.
    If neither direction is possible, throw an `ArgumentError`.
- The collection `js` must be nonempty, contain no duplicates, and exclude `i`.
    A collection returns a vector in `vec(js)` order; a single second site returns a scalar.
- For row-by-row contraction, all windows have the same width, covering the columns of `i` and all sites in `js`, but each window ends at the row of the measured site `j`.
    For column-by-column contraction, all windows cover the same rows, but each window ends at the column of `j`.
    Each window has its own normalization, calculated without the operator using the same contraction arrangement.
    The last row or column is contracted without further truncation against the CTMRG boundary immediately beyond it.
- `trunc` controls boundary-MPS truncation; by default, it limits the rank to the largest CTMRG boundary dimension.
    After each zipup step, `maxiter` DMRG sweeps refine the result (default 1; use 0 to disable refinement).
"""
function correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        i::CartesianIndex{2}, j::CartesianIndex{2}, env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1,
    )
    return only(correlator_approx(ρ, op, i, j:j, env; trunc, maxiter))
end

function correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        i::CartesianIndex{2}, js::CoordCollection{2}, env::CTMRGEnv;
        trunc = _approx_trunc(env), maxiter::Int = 1,
    )
    isempty(js) && throw(ArgumentError("correlator_approx requires at least one second site"))
    allunique(js) || throw(ArgumentError("second sites should be unique"))
    i in js && throw(ArgumentError("second sites should be distinct from the first site"))
    rowrange, colrange = _window_ranges([i; vec(js)])
    rows, columns = first(rowrange) == i[1], last(colrange) == i[2]
    rows || columns || throw(ArgumentError("no valid sweep: second sites must all be at or south of the first row, or all at or west of the first column"))
    direction = rows && (!columns || length(colrange) >= length(rowrange)) ? :rows : :columns
    return _correlator_approx(
        ρ, op, i, collect(vec(js)), env,
        WindowApprox(Zipup(; trunc), _approx_dmrg(maxiter)), direction
    )
end

"""
Group targets by row, retaining each target's original result position.
"""
function _twosite_targets_by_row(targets::Vector{CartesianIndex{2}})
    targets_by_row = Dict{Int, Dict{CartesianIndex{2}, Int}}()
    for (position, target) in enumerate(targets)
        row_targets = get!(Dict{CartesianIndex{2}, Int}, targets_by_row, target[1])
        row_targets[target] = position
    end
    return targets_by_row
end

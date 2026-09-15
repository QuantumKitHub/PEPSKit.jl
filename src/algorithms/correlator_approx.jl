# Approximate finite-window two-site correlators
# ----------------------------------------------

"""
$(SIGNATURES)

Approximately measure a dense two-site operator between a fixed first site `i` and second sites `js` in a single-layer PEPO.

- Operator leg 1 acts at `i`, and leg 2 acts at each second site.
- The sweep direction is selected automatically: north-to-south requires every `j[1] ≥ i[1]`, and east-to-west requires every `j[2] ≤ i[2]`.
    If both sweeps are valid, use north-to-south for a square or wide window, and east-to-west for a tall window.
    If neither sweep is valid, throw an `ArgumentError`.
- `js` must be nonempty, unique, and distinct from `i`.
    A collection returns a vector in `vec(js)` order; a single second site returns a scalar.
    All targets share the smallest enclosing rectangular window and its normalization, calculated once by a full sweep without observables.
- Boundary-MPS truncation uses `trunc` (defaulting to the largest CTMRG boundary dimension), with `maxiter` DMRG refinement sweeps after each zipup step (default 1; 0 disables refinement).
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

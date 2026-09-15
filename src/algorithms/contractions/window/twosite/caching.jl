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

"""
Cache observable-free row MPOs, the initial north boundary, south boundaries, and the window normalization.

The fields contain:

- `rowrange` and `colrange`: the coordinate ranges defining the window.
- `row_mpos`: observable-free row MPOs, including the west and east CTMRG edges, indexed by window-relative row position.
- `north_boundary`: the initial north boundary MPS above the first window row.
- `south_boundaries`: adjointed south boundary MPSs, with entry `k` immediately below window row `k`.
- `norm`: the approximate contraction without observables, calculated by one full north-to-south sweep without retaining intermediate states.
"""
struct WindowRowCache{M <: FiniteMPO, N <: FiniteMPS, S <: FiniteMPS, T <: Number}
    rowrange::UnitRange{Int}
    colrange::UnitRange{Int}
    row_mpos::Vector{M}
    north_boundary::N
    south_boundaries::Vector{S}
    norm::T
end

"""
Precompute shared row MPOs and south boundaries, and normalize with one rolling north state.
"""
function _window_row_cache(
        ρ::InfinitePEPO, env::CTMRGEnv,
        rowrange::UnitRange{Int}, colrange::UnitRange{Int}, alg::WindowApprox,
    )::WindowRowCache
    row_mpos = [_row_mpo(ρ, nothing, env, row, colrange) for row in rowrange]
    north = _north_boundary_mps(env, first(rowrange), colrange)
    south = _south_boundary_mps(env, last(rowrange), colrange)
    state = north
    for W in row_mpos
        state = _approximate(W, state, alg)
    end
    norm = dot(south, state)

    south_boundaries = Vector{typeof(south)}(undef, length(rowrange))
    south_boundaries[end] = south
    for k in (length(rowrange) - 1):-1:1
        W = _adjoint_mpo(row_mpos[k + 1])
        ψ = south_boundaries[k + 1]
        south_boundaries[k] = _approximate(W, ψ, alg)
    end
    return WindowRowCache(rowrange, colrange, row_mpos, north, south_boundaries, norm)
end

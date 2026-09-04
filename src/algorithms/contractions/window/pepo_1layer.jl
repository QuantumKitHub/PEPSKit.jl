# Approximate finite-window contractions for single-layer PEPO networks.

"""
Validate that the network is a single-layer PEPO and that the sweep direction is supported.
"""
function _check_window_inputs(ρ::InfinitePEPO, direction::Symbol)
    size(ρ, 3) == 1 || throw(DimensionMismatch("only single-layer PEPO contractions are supported"))
    direction in (:auto, :rows, :columns) ||
        throw(ArgumentError("invalid sweep direction: $direction"))
    return nothing
end

"""
Return a PEPO and CTMRG environment with standard virtual-space dualness without mutating the inputs.
"""
function standardize_dualness(ρ::InfinitePEPO, env::CTMRGEnv)
    isdual_easts, isdual_norths = _check_virtual_dualness(ρ)
    all(isdual_easts) && all(isdual_norths) && return ρ, env

    nrows, ncols = size(ρ, 1), size(ρ, 2)
    tensors = map(CartesianIndices(unitcell(ρ))) do site
        row, col, layer = Tuple(site)
        directions = Int[]
        !isdual_norths[row, col, layer] && push!(directions, NORTH)
        !isdual_easts[row, col, layer] && push!(directions, EAST)
        !isdual_norths[_next(row, nrows), col, layer] && push!(directions, SOUTH)
        !isdual_easts[row, _prev(col, ncols), layer] && push!(directions, WEST)
        A = unitcell(ρ)[site]
        return isempty(directions) ? A : flip_virtualspace(A, directions)
    end
    ρ′ = InfinitePEPO(tensors)

    edges = map(CartesianIndices(env.edges)) do index
        direction, row, col = Tuple(index)
        should_flip = if direction == NORTH
            !isdual_norths[_next(row, nrows), col, 1]
        elseif direction == EAST
            !isdual_easts[row, _prev(col, ncols), 1]
        elseif direction == SOUTH
            !isdual_norths[row, col, 1]
        else
            !isdual_easts[row, col, 1]
        end
        E = env.edges[index]
        return should_flip ? flip(E, 2) : E
    end
    env′ = CTMRGEnv(copy(env.corners), edges)
    return ρ′, env′
end

"""
Contract an MPO observable in its enclosing window, rotating column sweeps into row sweeps.
"""
function _expectation_value_approx(
        ρ::InfinitePEPO, observable::MPOObservable, env::CTMRGEnv,
        alg::WindowApprox, direction::Symbol,
    )
    _check_window_inputs(ρ, direction)
    rowrange, colrange = _window_ranges(observable)
    sweep = direction === :auto ? (length(colrange) > length(rowrange) ? :rows : :columns) : direction
    if sweep === :rows
        return _expectation_value_approx_rows(
            ρ, observable, env, rowrange, colrange, alg
        )
    else
        unitcell = size(ρ)[1:2]
        sites = siterotl90.(observable.sites, Ref(unitcell))
        path = siterotl90.(observable.path, Ref(unitcell))
        rotated_observable = MPOObservable(sites, path, observable.mpo)
        rotated_rowrange, rotated_colrange = _window_ranges(rotated_observable)
        return _expectation_value_approx_rows(
            rotl90(ρ), rotated_observable, rotl90(env),
            rotated_rowrange, rotated_colrange, alg
        )
    end
end

"""
Build a local tensor for row MPOs without observables by tracing the PEPO physical legs.
"""
function _window_site_tensor(
        ρ::InfinitePEPO, ::Nothing, row::Int, col::Int,
    )
    return trace_physicalspaces(ρ[row, col, 1])
end

"""
Build the local row-MPO tensor at one window site, inserting the observable tensor or routed
string when the site lies on the MPO path and tracing the PEPO physical legs otherwise.
"""
function _window_site_tensor(
        ρ::InfinitePEPO, observable::MPOObservable, row::Int, col::Int,
    )
    A = ρ[row, col, 1]
    site = CartesianIndex(row, col)
    path_index = findfirst(==(site), observable.path)
    isnothing(path_index) && return trace_physicalspaces(A)

    mpo_index = findfirst(==(site), observable.sites)
    # sites with string passing by (cannot be first/last site)
    if isnothing(mpo_index)
        incoming = _step_direction(observable.path[path_index], observable.path[path_index - 1])
        outgoing = _step_direction(observable.path[path_index], observable.path[path_index + 1])
        next_mpo_index = count(in(observable.sites), @view observable.path[1:path_index]) + 1
        stringspace = space(observable.mpo[next_mpo_index], 1)
        return mpo_path_string(A, stringspace, Val((incoming, outgoing)))
    end

    # sites acted on by the MPO
    op = observable.mpo[mpo_index]
    if mpo_index == 1
        direction = _step_direction(observable.path[1], observable.path[2])
        return mpo_path_first(A, op, Val(direction))
    elseif mpo_index == length(observable.mpo)
        direction = _step_direction(observable.path[end], observable.path[end - 1])
        return mpo_path_last(A, op, Val(direction))
    else
        incoming = _step_direction(observable.path[path_index], observable.path[path_index - 1])
        outgoing = _step_direction(observable.path[path_index], observable.path[path_index + 1])
        return mpo_path_middle(A, op, Val((incoming, outgoing)))
    end
end

"""
Contract and normalize an MPO observable using row-oriented window boundary contractions.
"""
function _expectation_value_approx_rows(
        ρ::InfinitePEPO, observable::MPOObservable, env::CTMRGEnv,
        rowrange::UnitRange{Int}, colrange::UnitRange{Int}, alg::WindowApprox,
    )
    ρ, env = standardize_dualness(ρ, env)
    numerator = _contract_window_rows(ρ, observable, env, rowrange, colrange, alg)
    norm = _contract_window_rows(ρ, nothing, env, rowrange, colrange, alg)
    return numerator / norm
end

"""
Contract a complete PEPO window row by row from north to south,
optionally inserting an MPO observable.
"""
function _contract_window_rows(
        ρ::InfinitePEPO, observable::Union{Nothing, MPOObservable},
        env::CTMRGEnv, rowrange::UnitRange{Int}, colrange::UnitRange{Int},
        alg::WindowApprox,
    )
    ψ = _north_boundary_mps(env, first(rowrange), colrange)
    for row in rowrange
        W = _row_mpo(ρ, observable, env, row, colrange)
        ψ = _approximate_window_step(W, ψ, alg)
    end
    south = _south_boundary_mps(env, last(rowrange), colrange)
    return dot(south, ψ)
end

"""
Build one finite row MPO from west/east CTMRG edges and the PEPO tensors inside the window.

Convention of west, east CTM edges and the PF tensors:
```
    [1 2; 3]    [1 2; 3 4]     [1 2; 3]
    3               3               1
    ↓               ↓               ↑
    E₄-←-2      1-←-O-←-4       2-←-C₂
    ↓               ↓               ↑
    1               2               3
```
Legs 1, 3 need to be flipped to match standard MPS convention
"""
function _row_mpo(
        ρ::InfinitePEPO, observable::Union{Nothing, MPOObservable},
        env::CTMRGEnv, row::Int, colrange::UnitRange{Int},
    )
    cmin, cmax = first(colrange), last(colrange)
    W = repartition(edge(env, WEST, row, cmin - 1), 1, 2)
    tensors = [insertleftunit(W, 1)]
    append!(
        tensors,
        (
            _window_site_tensor(ρ, observable, row, col)
                for col in colrange
        ),
    )
    E = permute(
        flip(edge(env, EAST, row, cmax + 1), (1, 3)),
        ((2, 3), (1,))
    )
    push!(tensors, insertrightunit(E, 3))
    return FiniteMPO(tensors)
end

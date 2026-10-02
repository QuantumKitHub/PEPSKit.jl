# Approximate finite-patch contractions for single-layer PEPO networks.

"""
Validate that the network is a single-layer PEPO and that the sweep direction is supported.
"""
function _check_patch_inputs(ρ::InfinitePEPO, direction::Symbol)
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
Contract a routed MPO term in its enclosing patch, rotating column sweeps into row sweeps.
"""
function _expectation_value_approx(
        ρ::InfinitePEPO, routed::RoutedMPOTerm, env::CTMRGEnv,
        alg::PatchApprox, direction::Symbol,
    )
    _check_patch_inputs(ρ, direction)
    rowrange, colrange = _patch_ranges(first(routed))
    sweep = if direction === :auto
        length(colrange) >= length(rowrange) ? :rows : :columns
    else
        direction
    end
    if sweep === :rows
        return _expectation_value_approx_rows(
            ρ, routed, env, rowrange, colrange, alg
        )
    else
        unitcell = size(ρ)[1:2]
        path, mpo = routed
        rotated = siterotl90.(path, Ref(unitcell)) => mpo
        rotated_rowrange, rotated_colrange = _patch_ranges(first(rotated))
        return _expectation_value_approx_rows(
            rotl90(ρ), rotated, rotl90(env),
            rotated_rowrange, rotated_colrange, alg
        )
    end
end

"""
Build a local tensor for row MPOs without observables by tracing the PEPO physical legs.
"""
function _patch_site_tensor(
        ρ::InfinitePEPO, ::Nothing, row::Int, col::Int,
    )
    return trace_physicalspaces(ρ[row, col, 1])
end

"""
Insert the MPO factor at a path site, or trace the PEPO physical legs at an off-path site.
"""
function _patch_site_tensor(
        ρ::InfinitePEPO, routed::RoutedMPOTerm,
        row::Int, col::Int,
    )
    A = ρ[row, col, 1]
    path, mpo = routed
    k = findfirst(==(CartesianIndex(row, col)), path)
    isnothing(k) && return trace_physicalspaces(A)

    if k == 1
        direction = _step_direction(path[1], path[2])
        return mpo_path_first(A, mpo[k], Val(direction))
    elseif k == length(path)
        direction = _step_direction(path[end], path[end - 1])
        return mpo_path_last(A, mpo[k], Val(direction))
    else
        incoming = _step_direction(path[k], path[k - 1])
        outgoing = _step_direction(path[k], path[k + 1])
        return mpo_path_middle(A, mpo[k], Val((incoming, outgoing)))
    end
end

"""
Contract and normalize a routed MPO term using row-oriented patch boundary contractions.
"""
function _expectation_value_approx_rows(
        ρ::InfinitePEPO, routed::RoutedMPOTerm, env::CTMRGEnv,
        rowrange::UnitRange{Int}, colrange::UnitRange{Int}, alg::PatchApprox,
    )
    ρ, env = standardize_dualness(ρ, env)
    numerator = _contract_patch_rows(ρ, routed, env, rowrange, colrange, alg)
    norm = _contract_patch_rows(ρ, nothing, env, rowrange, colrange, alg)
    return numerator / norm
end

"""
Contract a complete PEPO patch row by row from north to south, optionally inserting a routed MPO term.
"""
function _contract_patch_rows(
        ρ::InfinitePEPO, routed::Union{Nothing, RoutedMPOTerm},
        env::CTMRGEnv, rowrange::UnitRange{Int}, colrange::UnitRange{Int},
        alg::PatchApprox,
    )
    ψ = _north_boundary_mps(env, first(rowrange), colrange)
    for row in rowrange
        W = _row_mpo(ρ, routed, env, row, colrange)
        ψ = _approximate(W, ψ, alg)
    end
    south = _south_boundary_mps(env, last(rowrange), colrange)
    return dot_noconj(south, ψ)
end

"""
Build one finite row MPO from west/east CTMRG edges and the PEPO tensors inside the patch.

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
        ρ::InfinitePEPO, routed::Union{Nothing, RoutedMPOTerm},
        env::CTMRGEnv, row::Int, colrange::UnitRange{Int},
    )
    cmin, cmax = first(colrange), last(colrange)
    W = repartition(edge(env, WEST, row, cmin - 1), 1, 2)
    tensors = [insertleftunit(W, 1)]
    append!(
        tensors,
        (
            _patch_site_tensor(ρ, routed, row, col)
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

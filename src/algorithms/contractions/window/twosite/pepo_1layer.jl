"""
Validate operator spaces and rotate column sweeps into the row-oriented contraction.
"""
function _correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        source::CartesianIndex{2}, targets::Vector{CartesianIndex{2}},
        env::CTMRGEnv, alg::WindowApprox, direction::Symbol,
    )
    _check_window_inputs(ρ, direction)
    numout(op) == numin(op) == 2 ||
        throw(ArgumentError("correlator_approx requires a two-site operator"))
    for (leg, sites) in enumerate(((source,), targets)), site in sites
        V = physicalspace(ρ, Tuple(site)...)
        V == codomain(op)[leg] == domain(op)[leg] ||
            throw(SpaceMismatch("operator physical space does not match PEPO site $site"))
    end
    if direction === :columns
        unitcell = size(ρ)[1:2]
        source = siterotl90(source, unitcell)
        targets = siterotl90.(targets, Ref(unitcell))
        ρ, env = rotl90(ρ), rotl90(env)
    end
    return _correlator_approx_rows(ρ, op, source, targets, env, alg)
end

"""
Measure all targets with one MPO decomposition and a single open string propagated south from the fixed source.
"""
function _correlator_approx_rows(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        source::CartesianIndex{2}, targets::Vector{CartesianIndex{2}},
        env::CTMRGEnv, alg::WindowApprox,
    )
    ρ, env = standardize_dualness(ρ, env)
    rowrange, colrange = _window_ranges([source; targets])
    cache = _window_row_cache(ρ, env, rowrange, colrange, alg)
    targets_by_row = _twosite_targets_by_row(targets)
    mpo = gate_to_mpo(op; trunc = notrunc())
    stringspace = space(mpo[2], 1)
    numerators = zeros(promote_type(scalartype(op), typeof(cache.norm)), length(targets))
    north = cache.north_boundary
    for row in rowrange
        if haskey(targets_by_row, row)
            _contract_twosite_target_row!(
                numerators, ρ, mpo, source, targets_by_row[row], north, cache
            )
        end
        row == last(rowrange) && break
        A = ρ[row, source[2], 1]
        tensor = row == source[1] ? mpo_path_first(A, mpo[1], Val(:south)) :
            mpo_path_string(A, stringspace, Val((:north, :south)))
        W = _row_mpo_with_site(cache, tensor, row, source[2])
        north = _approximate(W, north, alg)
    end
    return numerators ./ cache.norm
end

"""
Contract all targets in one row with a shared north state, writing results into `numerators`.
"""
function _contract_twosite_target_row!(
        numerators::Vector{<:Number}, ρ::InfinitePEPO,
        mpo::AbstractVector{<:AbstractTensorMap},
        source::CartesianIndex{2}, targets::Dict{CartesianIndex{2}, Int},
        north::FiniteMPS, cache::WindowRowCache,
    )
    row = first(keys(targets))[1]
    row_idx = row - first(cache.rowrange) + 1
    south = cache.south_boundaries[row_idx]
    envs = environments(south, cache.row_mpos[row_idx], north)
    source_site = _window_mps_site(source[2], cache.colrange)
    stringspace = space(mpo[2], 1)

    # close the target right at the column of the incoming string
    same_col = get(targets, CartesianIndex(row, source[2]), nothing)
    if !isnothing(same_col)
        target_tensor = mpo_path_last(ρ[row, source[2], 1], mpo[2], Val(:north))
        value = _contract_window_site(envs, north, south, source_site, target_tensor)
        numerators[same_col] = value
    end

    # Close targets on the right of the incoming string from left to right
    right_targets = [target for target in keys(targets) if target[2] > source[2]]
    if !isempty(right_targets)
        sort!(right_targets; by = x -> x[2])
        A = ρ[row, source[2], 1]
        source_tensor = if row == source[1]
            mpo_path_first(A, mpo[1], Val(:east))
        else
            mpo_path_string(A, stringspace, Val((:north, :east)))
        end
        left = leftenv(envs, source_site, south) *
            TransferMatrix(north.AC[source_site], source_tensor, south.AC[source_site])
        previous_col = source[2]
        for target in right_targets
            target_col = target[2]
            for col in (previous_col + 1):(target_col - 1)
                site = _window_mps_site(col, cache.colrange)
                string_tensor = mpo_path_string(ρ[row, col, 1], stringspace, Val((:west, :east)))
                left = left * TransferMatrix(north.AR[site], string_tensor, south.AR[site])
            end

            target_site = _window_mps_site(target_col, cache.colrange)
            target_tensor = mpo_path_last(ρ[row, target_col, 1], mpo[2], Val(:west))
            target_left = left * TransferMatrix(north.AR[target_site], target_tensor, south.AR[target_site])
            value = _contract_transfer_boundaries(target_left, rightenv(envs, target_site, south))
            numerators[targets[target]] = value

            string_tensor = mpo_path_string(ρ[row, target_col, 1], stringspace, Val((:west, :east)))
            left = left * TransferMatrix(north.AR[target_site], string_tensor, south.AR[target_site])
            previous_col = target_col
        end
    end

    # Close targets on the left of the incoming string from right to left
    left_targets = [target for target in keys(targets) if target[2] < source[2]]
    if !isempty(left_targets)
        sort!(left_targets; by = x -> x[2], rev = true)
        A = ρ[row, source[2], 1]
        source_tensor = if row == source[1]
            mpo_path_first(A, mpo[1], Val(:west))
        else
            mpo_path_string(A, stringspace, Val((:north, :west)))
        end
        right = TransferMatrix(
            north.AC[source_site], source_tensor, south.AC[source_site]
        ) * rightenv(envs, source_site, south)
        previous_col = source[2]
        for target in left_targets
            target_col = target[2]
            for col in (previous_col - 1):-1:(target_col + 1)
                site = _window_mps_site(col, cache.colrange)
                string_tensor = mpo_path_string(ρ[row, col, 1], stringspace, Val((:east, :west)))
                right = TransferMatrix(north.AL[site], string_tensor, south.AL[site]) * right
            end

            target_site = _window_mps_site(target_col, cache.colrange)
            target_tensor = mpo_path_last(ρ[row, target_col, 1], mpo[2], Val(:east))
            target_right = TransferMatrix(north.AL[target_site], target_tensor, south.AL[target_site]) * right
            value = _contract_transfer_boundaries(leftenv(envs, target_site, south), target_right)
            numerators[targets[target]] = value

            string_tensor = mpo_path_string(ρ[row, target_col, 1], stringspace, Val((:east, :west)))
            right = TransferMatrix(north.AL[target_site], string_tensor, south.AL[target_site]) * right
            previous_col = target_col
        end
    end
    return numerators
end

"""
Map a PEPO column to its finite-MPS site, accounting for the additional west CTM edge.
"""
_window_mps_site(col::Int, colrange::UnitRange{Int}) = col - first(colrange) + 2

"""
Replace one site in a copied row-MPO tensor container, leaving the cached row unchanged.
"""
function _row_mpo_with_site(
        cache::WindowRowCache, tensor::MPOTensor, row::Int, col::Int,
    )
    tensors = copy(parent(cache.row_mpos[row - first(cache.rowrange) + 1]))
    tensors[_window_mps_site(col, cache.colrange)] = tensor
    return FiniteMPO(tensors)
end

"""
Contract one modified row site between precomputed left and right MPS environments.
"""
function _contract_window_site(
        envs::MPSKit.FiniteEnvironments, north::FiniteMPS, south::FiniteMPS,
        site::Int, tensor::MPOTensor,
    )
    left = leftenv(envs, site, south) *
        TransferMatrix(north.AC[site], tensor, south.AC[site])
    return _contract_transfer_boundaries(left, rightenv(envs, site, south))
end

"""
Contract the left and right transfer-matrix environments to a scalar.
```
    (north)
    ┌-←-- 3 --←-┐
    |           |
    L-←-- 2 --←-R
    |           |
    └-→-- 1 --→-┘
    (south)
```
"""
function _contract_transfer_boundaries(left::MPSTensor, right::MPSTensor)
    # The three bonds close around the window without crossing
    return @plansor left[1 2; 3] * right[3 2; 1]
end

# Source-cached dense two-site contractions for single-layer PEPO networks.

"""
Validate a dense two-site measurement, choose its sweep orientation, and dispatch to the
row-oriented source-cached contraction.
"""
function _correlator_approx(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        bonds::Vector{NTuple{2, CartesianIndex{2}}},
        env::CTMRGEnv, alg::WindowApprox, direction::Symbol,
    )
    _check_window_inputs(ρ, direction)
    numout(op) == numin(op) == 2 ||
        throw(ArgumentError("correlator_approx requires a two-site operator"))
    for bond in bonds, (i, site) in enumerate(bond)
        V = physicalspace(ρ, Tuple(site)...)
        V == codomain(op)[i] == domain(op)[i] ||
            throw(SpaceMismatch("operator physical space does not match PEPO site $site"))
    end

    rowrange, colrange = _window_ranges(bonds)
    sweep = direction === :auto ? (length(colrange) > length(rowrange) ? :rows : :columns) : direction
    if sweep === :rows
        return _correlator_approx_rows(
            ρ, op, bonds, env, rowrange, colrange, alg
        )
    end
    # rotate column-wise contraction to reuse row-wise code
    unitcell = size(ρ)[1:2]
    rotated_bonds = map(bonds) do bond
        return (siterotl90(bond[1], unitcell), siterotl90(bond[2], unitcell))
    end
    rotated_rowrange, rotated_colrange = _window_ranges(rotated_bonds)
    return _correlator_approx_rows(
        rotl90(ρ), op, rotated_bonds, rotl90(env),
        rotated_rowrange, rotated_colrange, alg
    )
end

"""
Measure all ordered bonds in one row-oriented window using shared row MPOs without observables, shared boundaries, and one exactly decomposed MPO for each operator-leg ordering.
"""
function _correlator_approx_rows(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        bonds::Vector{NTuple{2, CartesianIndex{2}}}, env::CTMRGEnv,
        rowrange::UnitRange{Int}, colrange::UnitRange{Int}, alg::WindowApprox,
    )
    ρ, env = standardize_dualness(ρ, env)
    cache = _window_row_cache(ρ, env, rowrange, colrange, alg)
    groups = _twosite_source_groups(bonds)
    mpo = gate_to_mpo(op; trunc = notrunc())
    swapped_mpo = if any(key[2] for key in keys(groups))
        swapped_op = permute(op, ((2, 1), (4, 3)))
        gate_to_mpo(swapped_op; trunc = notrunc())
    end

    T = promote_type(scalartype(op), typeof(cache.norm))
    numerators = zeros(T, length(bonds))
    for ((source, swapped), targets) in groups
        source_mpo = swapped ? something(swapped_mpo) : mpo
        _contract_twosite_source!(
            numerators, ρ, source_mpo, source, targets, env, cache, alg,
        )
    end
    return numerators ./ cache.norm
end

"""
Contract the correlator numerator for all targets associated with the same source
and one ordering of the dense operator, writing each result into `numerators`.
"""
function _contract_twosite_source!(
        numerators::Vector{<:Number}, ρ::InfinitePEPO,
        mpo::AbstractVector{<:AbstractTensorMap},
        source::CartesianIndex{2}, targets::Dict{CartesianIndex{2}, Int},
        env::CTMRGEnv, cache::WindowRowCache, alg::WindowApprox,
    )
    # grouping targets by which row they are in
    targets_by_row = _twosite_targets_by_row(targets)
    source_idx = source[1] - first(cache.rowrange) + 1
    north = cache.north_prefixes[source_idx]

    # Close targets in the same row as the source
    if haskey(targets_by_row, source[1])
        _contract_twosite_target_row!(
            numerators, ρ, mpo, source, targets_by_row[source[1]], north, cache
        )
    end
    last_target_row = maximum(keys(targets_by_row))
    last_target_row == source[1] && return numerators

    # Open the MPO string toward the south for targets in later rows.
    A = ρ[source[1], source[2], 1]
    source_tensor = mpo_path_first(A, mpo[1], Val(:south))
    W = _row_mpo_with_site(ρ, source_tensor, env, source[1], source[2], cache.colrange)
    north = _approximate_window_step(W, north, alg)

    stringspace = space(mpo[2], 1)
    for row in (source[1] + 1):last_target_row
        # Close every target in this row
        if haskey(targets_by_row, row)
            _contract_twosite_target_row!(
                numerators, ρ, mpo, source, targets_by_row[row], north, cache
            )
        end
        row == last_target_row && break
        # Carry the open string down to the next row
        A = ρ[row, source[2], 1]
        string_tensor = mpo_path_string(A, stringspace, Val((:north, :south)))
        W = _row_mpo_with_site(ρ, string_tensor, env, row, source[2], cache.colrange)
        north = _approximate_window_step(W, north, alg)
    end
    return numerators
end

"""
Contract the correlator numerator for all targets in one row with a
shared open-string north state, writing the results into `numerators`.
"""
function _contract_twosite_target_row!(
        numerators::Vector{<:Number}, ρ::InfinitePEPO,
        mpo::AbstractVector{<:AbstractTensorMap},
        source::CartesianIndex{2}, targets::Dict{CartesianIndex{2}, Int},
        north::FiniteMPS, cache::WindowRowCache,
    )
    row = first(keys(targets))[1]
    row_idx = row - first(cache.rowrange) + 1
    south = cache.south_suffixes[row_idx + 1]
    envs = environments(south, cache.row_mpos[row], north)
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
        source_tensor = mpo_path_string(A, stringspace, Val((:north, :west)))
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
Map a PEPO column coordinate to its finite-MPS site number,
which includes an additional west edge CTM tensor.
"""
_window_mps_site(col::Int, colrange::UnitRange{Int}) = col - first(colrange) + 2

"""
Build a finite row MPO by replacing one site tensor in one of the row MPOs without observables.
"""
function _row_mpo_with_site(
        ρ::InfinitePEPO, tensor::MPOTensor, env::CTMRGEnv,
        row::Int, col::Int, colrange::UnitRange{Int},
    )
    W = _row_mpo(ρ, nothing, env, row, colrange)
    parent(W)[_window_mps_site(col, colrange)] = tensor
    return W
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

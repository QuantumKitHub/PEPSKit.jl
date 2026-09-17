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
Measure targets in windows of fixed width ending at each target row, propagating observable-free and open-string north states together.
"""
function _correlator_approx_rows(
        ρ::InfinitePEPO, op::AbstractTensorMap,
        source::CartesianIndex{2}, targets::Vector{CartesianIndex{2}},
        env::CTMRGEnv, alg::WindowApprox,
    )
    ρ, env = standardize_dualness(ρ, env)
    rowrange, colrange = _window_ranges([source; targets])
    targets_by_row = _twosite_targets_by_row(targets)
    mpo = gate_to_mpo(op; trunc = notrunc())
    stringspace = space(mpo[2], 1)
    values = zeros(promote_type(scalartype(op), scalartype(ρ), scalartype(env)), length(targets))
    north = _north_boundary_mps(env, source[1], colrange)
    plain_north = north
    for row in rowrange
        W = _row_mpo(ρ, nothing, env, row, colrange)
        if haskey(targets_by_row, row)
            south = _south_boundary_mps(env, row, colrange)
            norm = dot_noconj(south, W, plain_north)
            _contract_twosite_target_row!(
                values, ρ, mpo, source, targets_by_row[row], north, south, W, colrange
            )
            for k in Base.values(targets_by_row[row])
                values[k] /= norm
            end
        end
        row == last(rowrange) && break
        A = ρ[row, source[2], 1]
        tensor = row == source[1] ? mpo_path_first(A, mpo[1], Val(:south)) :
            mpo_path_string(A, stringspace, Val((:north, :south)))
        plain_north = _approximate(W, plain_north, alg)
        parent(W)[_window_mps_site(source[2], colrange)] = tensor
        north = _approximate(W, north, alg)
    end
    return values
end

"""
Contract all targets in one row `W` with a shared `north` and `south` boundary MPS, writing results into `numerators`.
"""
function _contract_twosite_target_row!(
        numerators::Vector{<:Number}, ρ::InfinitePEPO,
        mpo::AbstractVector{<:AbstractTensorMap},
        source::CartesianIndex{2}, targets::Dict{CartesianIndex{2}, Int},
        north::FiniteMPS, south::FiniteMPS, W::FiniteMPO, colrange::UnitRange{Int},
    )
    N = length(south)
    row = first(keys(targets))[1]
    source_site = _window_mps_site(source[2], colrange)
    envs = _window_edge_environments(south, W, north, source_site)
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
        left = envs.lefts[source_site] *
            edge_transfermatrix(north.AC[source_site], source_tensor, south.AC[south_site(source_site, N)])
        previous_col = source[2]
        for target in right_targets
            target_col = target[2]
            for col in (previous_col + 1):(target_col - 1)
                site = _window_mps_site(col, colrange)
                string_tensor = mpo_path_string(ρ[row, col, 1], stringspace, Val((:west, :east)))
                left = left * edge_transfermatrix(north.AR[site], string_tensor, south.AL[south_site(site, N)])
            end

            target_site = _window_mps_site(target_col, colrange)
            target_tensor = mpo_path_last(ρ[row, target_col, 1], mpo[2], Val(:west))
            target_left = left * edge_transfermatrix(north.AR[target_site], target_tensor, south.AL[south_site(target_site, N)])
            value = _contract_transfer_boundaries(target_left, envs.rights[target_site - source_site + 1])
            numerators[targets[target]] = value

            string_tensor = mpo_path_string(ρ[row, target_col, 1], stringspace, Val((:west, :east)))
            left = left * edge_transfermatrix(north.AR[target_site], string_tensor, south.AL[south_site(target_site, N)])
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
        right = edge_transfermatrix(
            north.AC[source_site], source_tensor, south.AC[south_site(source_site, N)]
        ) * first(envs.rights)
        previous_col = source[2]
        for target in left_targets
            target_col = target[2]
            for col in (previous_col - 1):-1:(target_col + 1)
                site = _window_mps_site(col, colrange)
                string_tensor = mpo_path_string(ρ[row, col, 1], stringspace, Val((:east, :west)))
                right = edge_transfermatrix(north.AL[site], string_tensor, south.AR[south_site(site, N)]) * right
            end

            target_site = _window_mps_site(target_col, colrange)
            target_tensor = mpo_path_last(ρ[row, target_col, 1], mpo[2], Val(:east))
            target_right = edge_transfermatrix(north.AL[target_site], target_tensor, south.AR[south_site(target_site, N)]) * right
            value = _contract_transfer_boundaries(envs.lefts[target_site], target_right)
            numerators[targets[target]] = value

            string_tensor = mpo_path_string(ρ[row, target_col, 1], stringspace, Val((:east, :west)))
            right = edge_transfermatrix(north.AL[target_site], string_tensor, south.AR[south_site(target_site, N)]) * right
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
Contract one modified row site between precomputed left and right MPS environments.
"""
function _contract_window_site(
        envs::NamedTuple, north::FiniteMPS, south::FiniteMPS,
        site::Int, tensor::MPOTensor,
    )
    N = length(south)
    left = envs.lefts[site] *
        edge_transfermatrix(north.AC[site], tensor, south.AC[south_site(site, N)])
    return _contract_transfer_boundaries(left, envs.rights[site - length(envs.lefts) + 1])
end

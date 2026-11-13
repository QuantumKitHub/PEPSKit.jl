"""
    network_value(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular)

Return the value per unit cell using the principal cube root of the contraction-ratio
product. Boundary phases cancel before taking the cube root. For general complex
networks, this estimator determines the phase only modulo `2π/3`.
"""
function network_value(network::InfiniteTriangularNetwork, env::CTMRGEnvTriangular)
    value = prod(Iterators.product(axes(network)...)) do (r, c)
        nw_corners = complex(_contract_corners((r, c), network, env))
        nw_full = complex(_contract_site_large((r, c), network, env))
        nw_0 = complex(_contract_edges_0((r, c), network, env))
        nw_60 = complex(_contract_edges_60((r, c), network, env))
        nw_120 = complex(_contract_edges_120((r, c), network, env))
        ratio = nw_full * nw_corners^2 / (nw_0 * nw_60 * nw_120)
        # Keep the full phase until the unit-cell product, without cubing its magnitude.
        return cbrt(abs(ratio)) * sign(ratio)
    end
    return abs(value) * cis(angle(value) / 3)
end

function local_expectation_value(
        inds, bra::InfinitePEPSTriangular, operator::AbstractTensorMap,
        ket::InfinitePEPSTriangular, env::CTMRGEnvTriangular,
    )
    site = first(inds)
    if length(inds) == 1
        contract = _contract_corners
    elseif length(inds) == 2
        offsets = (CartesianIndex(0, 1), CartesianIndex(1, -1), CartesianIndex(1, 0))
        offset = inds[2] - inds[1]
        dir = findfirst(==(offset), offsets)
        if isnothing(dir)
            dir = findfirst(==(-offset), offsets)
            isnothing(dir) && throw(ArgumentError("only nearest-neighbor triangular terms are supported"))
            site = inds[2]
            operator = permute(operator, ((2, 1), (4, 3)))
        end
        contract = (_contract_edges_0, _contract_edges_60, _contract_edges_120)[dir]
    else
        throw(ArgumentError("only one-site and two-site triangular terms are supported"))
    end
    network = InfiniteTriangularNetwork(ket, bra)
    site = mod1.(Tuple(site), size(network))
    return contract(site, network, env, operator) / contract(site, network, env)
end

function energy(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, onesite_op, twosite_op) where {P <: PEPSSandwichTriangular}
    E = sum(Iterators.product(axes(network)...)) do (r, c)
        expval_onesite = _contract_corners((r, c), network, env, onesite_op) / _contract_corners((r, c), network, env)
        expval_twosite = _contract_edges_0((r, c), network, env, twosite_op) / _contract_edges_0((r, c), network, env)
        return expval_onesite + expval_twosite
    end
    ignore_derivatives() do
        isapprox(imag(E), 0; atol = sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end
function energy(state::InfinitePEPSTriangular, env::CTMRGEnvTriangular, args...)
    return energy(InfiniteTriangularNetwork(state), env, args...)
end

"""
    energy(network::InfiniteTriangularNetwork, env, onesite_op, twosite_ops::NTuple{3})

Compute the energy per unit cell with a one-site term and nearest-neighbor terms
in the 0°, 60°, and 120° bond families. The oriented bonds connect `(r, c)` to
`(r, c + 1)`, `(r + 1, c - 1)`, and `(r + 1, c)`, respectively.
Divide by `length(network)` for the energy per site.
"""
function energy(network::InfiniteTriangularNetwork{P}, env::CTMRGEnvTriangular, onesite_op, twosite_ops::NTuple{3, AbstractTensorMap}) where {P <: PEPSSandwichTriangular}
    contractions = (_contract_edges_0, _contract_edges_60, _contract_edges_120)
    E = sum(Iterators.product(axes(network)...)) do site
        expval_onesite = _contract_corners(site, network, env, onesite_op) / _contract_corners(site, network, env)
        expval_twosite = sum(zip(contractions, twosite_ops)) do (contract, op)
            return contract(site, network, env, op) / contract(site, network, env)
        end
        return expval_onesite + expval_twosite
    end
    ignore_derivatives() do
        isapprox(imag(E), 0; atol = sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end

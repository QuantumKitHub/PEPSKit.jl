function LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)
    return network_value(InfiniteSquareNetwork(peps), env)
end

"""
    edge_transfer_spectrum(top::Vector{E}, bot::Vector{E}; tol=Defaults.tol, num_vals=20,
                           sector=one(sectortype(E))) where {E<:CTMRGEdgeTensor}

Calculate the partial spectrum of the left edge transfer matrix corresponding to the given
`top` vector of edges and a `bot` vector of edge. The `sector` keyword argument can be used
to specify a non-trivial total charge for the transfer matrix eigenvectors. Specifically, an
auxiliary space `ℂ[typeof(sector)](sector => 1)'` will be added to the domain of each
eigenvector. The `tol` and `num_vals` keyword arguments are passed to `KrylovKit.eigolve`.
"""
function edge_transfer_spectrum(
        top::Vector{E}, bot::Vector{E}; tol = MPSKit.Defaults.tol, num_vals = 20,
        sector = one(sectortype(E))
    ) where {E <: CTMRGEdgeTensor}
    init = randn(
        scalartype(E),
        space(first(bot), numind(first(bot)))' ← ℂ[typeof(sector)](sector => 1)' ⊗ space(first(top), 1),
    )

    transferspace = fuse(space(first(top), 1) * space(first(bot), numind(first(bot)))')
    num_vals = min(dim(transferspace, sector), num_vals) # we can ask at most this many values
    eigenvals, eigenvecs, convhist = eigsolve(
        flip(edge_transfermatrix(top, bot)), init, num_vals, :LM; tol = tol
    )
    convhist.converged < num_vals &&
        @warn "correlation length failed to converge: normres = $(convhist.normres)"

    return eigenvals
end


# TODO: decide on appropriate signature and returns for the more generic case
"""
    correlation_length(state, env::CTMRGEnv; num_vals=2, kwargs...)

Compute the correlation length associated to `state` as contracted using the environment
`env`, based on the spectrum of the horizontal and vertical transfer matrices associated to
`env`. Additionally the (normalized) eigenvalue spectrum is returned. The number of computed
eigenvalues can be specified using `num_vals`, and any remaining keyword arguments are
passed through to `MPSKit.transfer_spectrum` (e.g. allowing to target the correlation length
in a specific symmetry sector).

"""
function MPSKit.correlation_length(state, env::CTMRGEnv; num_vals = 2, kwargs...)
    return _correlation_length(env; num_vals, kwargs...)
end

function _correlation_length(
        env::CTMRGEnv; num_vals = 2, sector = one(sectortype(env)), kwargs...
    )
    _, n_rows, n_cols = size(env)

    # Horizontal
    λ_h = map(1:n_rows) do r
        top = env.edges[NORTH, r, :]
        bot = env.edges[SOUTH, _next(r, n_rows), :]
        vals = edge_transfer_spectrum(top, bot; num_vals, sector, kwargs...)

        # normalize using largest eigenvalue in trivial sector
        if isone(sector)
            N = first(vals)
        else
            vals_triv = edge_transfer_spectrum(top, bot; num_vals = 1, kwargs...)
            N = first(vals_triv)
        end
        return vals ./ N # normalize largest eigenvalue
    end

    # Vertical
    λ_v = map(1:n_cols) do c
        top = env.edges[EAST, :, c]
        bot = env.edges[WEST, :, _next(c, n_cols)]
        vals = edge_transfer_spectrum(top, bot; num_vals, sector, kwargs...)

        # normalize using largest eigenvalue in trivial sector
        if isone(sector)
            N = first(vals)
        else
            vals_triv = edge_transfer_spectrum(top, bot; num_vals = 1, kwargs...)
            N = first(vals_triv)
        end
        return vals ./ N # normalize largest eigenvalue
    end

    if isone(sector)
        ξ_h = map(λ -> -1 / log(abs(λ[2])), λ_h)
        ξ_v = map(λ -> -1 / log(abs(λ[2])), λ_v)
    else
        ξ_h = map(λ -> -1 / log(abs(λ[1])), λ_h)
        ξ_v = map(λ -> -1 / log(abs(λ[1])), λ_v)
    end

    return ξ_h, ξ_v, λ_h, λ_v
end

"""
    product_peps(peps_args...; unitcell=(1, 1), noise_amp=1e-2, state_vector=nothing)

Initialize a normalized random product PEPS with noise. The given arguments are passed on to
the `InfinitePEPS` constructor.

The noise intensity can be tuned with `noise_amp`. The product state coefficients can be
specified using the `state_vector` kwarg in the form of a matrix of size `unitcell`
containing vectors that match the PEPS physical dimensions. If `nothing` is provided,
random Gaussian coefficients are used.
"""
function product_peps(peps_args...; unitcell = (1, 1), noise_amp = 1.0e-2, state_vector = nothing)
    noise_peps = InfinitePEPS(peps_args...; unitcell)
    typeof(spacetype(noise_peps[1])) <: GradedSpace &&
        error("symmetric tensors not generically supported")
    if isnothing(state_vector)
        state_vector = map(noise_peps.A) do t
            randn(scalartype(t), dim(space(t, 1)))
        end
    else
        all(dim.(space.(noise_peps.A, 1)) .== length.(state_vector)) ||
            throw(ArgumentError("state vectors must match the physical dimension"))
    end
    prod_tensors = map(noise_peps.A, state_vector) do t, v
        pt = zero(t)
        pt[][:, 1, 1, 1, 1] .= v
        return pt
    end
    prod_peps = InfinitePEPS(prod_tensors)
    ψ = prod_peps + noise_amp * noise_peps
    return ψ / norm(ψ)
end

# Contract local tensors

"""
    contract_local_tensor(inds, O::PFTensor, env)

Contract a local tensor `O` inserted into a partition function `pf` at position `inds`,
using the environment `env`.
"""
function contract_local_tensor(
        inds::Tuple{Int, Int}, O::PFTensor, env::CTMRGEnv{C, <:CTMRG_PF_EdgeTensor}
    ) where {C}
    r, c = inds
    return _contract_site(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r + 1, c + 1),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, NORTH, r - 1, c), edge(env, EAST, r, c + 1),
        edge(env, SOUTH, r + 1, c), edge(env, WEST, r, c - 1),
        O,
    )
end

"""
    contract_local_tensor(inds, O::PEPOTensor, network, env)

Contract a local tensor `O` inserted into the PEPO of a given `network` at position `inds`,
using the environment `env`.
"""
function contract_local_tensor(
        ind::Tuple{Int, Int, Int},
        O::PEPOTensor,
        network::InfiniteSquareNetwork{<:PEPOSandwich},
        env::CTMRGEnv,
    )
    r, c, h = ind
    sandwich´ = Base.setindex(network[r, c], O, h + 2)
    return _contract_site(
        corner(env, NORTHWEST, r - 1, c - 1),
        corner(env, NORTHEAST, r - 1, c + 1),
        corner(env, SOUTHEAST, r + 1, c + 1),
        corner(env, SOUTHWEST, r + 1, c - 1),
        edge(env, NORTH, r - 1, c), edge(env, EAST, r, c + 1),
        edge(env, SOUTH, r + 1, c), edge(env, WEST, r, c - 1),
        sandwich´,
    )
end

function contract_local_tensor(inds::CartesianIndex, O::AbstractTensorMap, env::CTMRGEnv)
    return contract_local_tensor(Tuple(inds), O, env)
end

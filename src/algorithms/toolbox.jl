"""
    expectation_value(state, O::LocalOperator, env::CTMRGEnv)
    expectation_value(bra, O::LocalOperator, ket, env::CTMRGEnv)

Compute the expectation value ⟨bra|O|ket⟩ / ⟨bra|ket⟩ of a [`LocalOperator`](@ref) `O`.
This can be done either for a PEPS, or alternatively for a density matrix PEPO.
In the latter case the first signature corresponds to a single layer PEPO contraction, while
the second signature yields a bilayer contraction instead.
"""
function MPSKit.expectation_value(
        bra::Union{InfinitePEPS, InfinitePEPO}, O::LocalOperator,
        ket::Union{InfinitePEPS, InfinitePEPO}, env::CTMRGEnv
    )
    checklattice(bra, O, ket)
    ev = mapreduce(+, [O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        ρ = reduced_densitymatrix(inds, ket, bra, env)
        return trmul(operator, ρ)
    end
    return ev
end
MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env::CTMRGEnv) = expectation_value(peps, O, peps, env)
function MPSKit.expectation_value(
        state::InfinitePEPO, O::LocalOperator, env::CTMRGEnv
    )
    checklattice(state, O)
    term_vals = dtmap([O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        ρ = reduced_densitymatrix(inds, state, env)
        return trmul(operator, ρ)
    end
    return sum(term_vals)
end

"""
    expectation_value(pf::InfinitePartitionFunction, inds => O, env::CTMRGEnv)

Compute the expectation value corresponding to inserting a local tensor(s) `O` at
position `inds` in the partition function `pf` and contracting the chole using a given CTMRG
environment `env`.

Here `inds` can be specified as either a `Tuple{Int,Int}` or a `CartesianIndex{2}`, and `O`
should be a rank-4 tensor conforming to the [`PartitionFunctionTensor`](@ref) indexing
convention.
"""
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction,
        op::Pair{CartesianIndex{2}, <:AbstractTensorMap{T, S, 2, 2}},
        env::CTMRGEnv,
    ) where {T, S}
    return contract_local_tensor(op[1], op[2], env) /
        contract_local_tensor(op[1], pf[op[1]], env)
end
function MPSKit.expectation_value(
        pf::InfinitePartitionFunction, op::Pair{Tuple{Int, Int}}, env::CTMRGEnv
    )
    return expectation_value(pf, CartesianIndex(op[1]) => op[2], env)
end

"""
    cost_function(peps::InfinitePEPS, env::CTMRGEnv, O::LocalOperator)

Real part of expectation value of `O`. Prints a warning if the expectation value
yields a finite imaginary part (up to a tolerance).
"""
function cost_function(peps::InfinitePEPS, env::CTMRGEnv, O::LocalOperator)
    E = MPSKit.expectation_value(peps, O, env)
    ignore_derivatives() do
        isapprox(imag(E), 0; atol = sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end

function LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)
    return network_value(InfiniteSquareNetwork(peps), env)
end

"""
    network_value(network::InfiniteSquareNetwork, env::CTMRGEnv)

Return the value (per unit cell) of a given contractible network contracted using a given
CTMRG environment.
"""
function network_value(network::InfiniteSquareNetwork, env::CTMRGEnv)
    return prod(Iterators.product(axes(network)...)) do (r, c)
        return _contract_site((r, c), network, env) * _contract_corners((r, c), env) /
            _contract_vertical_edges((r, c), env) / _contract_horizontal_edges((r, c), env)
    end
end
network_value(state, env::CTMRGEnv) = network_value(InfiniteSquareNetwork(state), env)

"""
    _contract_site(ind::Tuple{Int,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv)

Contract around a single site `ind` of a square network using a given CTMRG environment.
"""
function _contract_site(ind::Tuple{Int, Int}, network, env::CTMRGEnv)
    r, c = ind
    return _contract_site(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
        network[r, c],
    )
end
function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::CTMRG_PEPS_EdgeTensor, E_east::CTMRG_PEPS_EdgeTensor,
        E_south::CTMRG_PEPS_EdgeTensor, E_west::CTMRG_PEPS_EdgeTensor,
        O::PEPSSandwich,
    )
    return @autoopt @tensor E_west[χ_WSW D_W_above D_W_below; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N_above D_N_below; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E_above D_E_below; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S_above D_S_below; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        ket(O)[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(O)[d; D_N_below D_E_below D_S_below D_W_below])
end
function _contract_site(
        C_northwest, C_northeast, C_southeast, C_southwest,
        E_north::CTMRG_PF_EdgeTensor, E_east::CTMRG_PF_EdgeTensor,
        E_south::CTMRG_PF_EdgeTensor, E_west::CTMRG_PF_EdgeTensor,
        O::PFTensor,
    )
    return @autoopt @tensor E_west[χ_WSW D_W; χ_WNW] *
        C_northwest[χ_WNW; χ_NNW] *
        E_north[χ_NNW D_N; χ_NNE] *
        C_northeast[χ_NNE; χ_ENE] *
        E_east[χ_ENE D_E; χ_ESE] *
        C_southeast[χ_ESE; χ_SSE] *
        E_south[χ_SSE D_S; χ_SSW] *
        C_southwest[χ_SSW; χ_WSW] *
        O[D_W D_S; D_N D_E]
end

"""
    _contract_corners(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract all corners around the south-east at position `ind` of the CTMRG
environment `env`.
"""
function _contract_corners(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    C_NW = env.corners[NORTHWEST, _prev(r, end), _prev(c, end)]
    C_NE = env.corners[NORTHEAST, _prev(r, end), c]
    C_SE = env.corners[SOUTHEAST, r, c]
    C_SW = env.corners[SOUTHWEST, r, _prev(c, end)]
    return @tensor C_NW[1; 2] * C_NE[2; 3] * C_SE[3; 4] * C_SW[4; 1]
end

"""
    _contract_vertical_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the vertical edges and corners around the east edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_vertical_edges(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    return _contract_vertical_edges(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), c],
        env.corners[SOUTHEAST, _next(r, end), c],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[EAST, r, c],
        env.edges[WEST, r, _prev(c, end)],
    )
end
@generated function _contract_vertical_edges(
        C_northwest::CTMRGCornerTensor, C_northeast::CTMRGCornerTensor,
        C_southeast::CTMRGCornerTensor, C_southwest::CTMRGCornerTensor,
        E_east::CTMRGEdgeTensor{T, S, N},
        E_west::CTMRGEdgeTensor{T, S, N},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:NW),), (envlabel(:N),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:N),), (envlabel(:NE),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:SE),), (envlabel(:S),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:S),), (envlabel(:SW),))

    E_east_e = tensorexpr(
        :E_east, (envlabel(:NE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SE),)
    )
    E_west_e = tensorexpr(
        :E_west, (envlabel(:SW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NW),)
    )

    rhs = Expr(
        :call, :*,
        E_west_e, C_northwest_e, C_northeast_e, E_east_e, C_southeast_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
end

"""
    _contract_horizontal_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the horizontal edges and corners around the south edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_horizontal_edges(ind::Tuple{Int, Int}, env::CTMRGEnv)
    r, c = ind
    return _contract_horizontal_edges(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, r, _next(c, end)],
        env.corners[SOUTHWEST, r, _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c],
        env.edges[SOUTH, r, c],
    )
end
@generated function _contract_horizontal_edges(
        C_northwest::CTMRGCornerTensor, C_northeast::CTMRGCornerTensor,
        C_southeast::CTMRGCornerTensor, C_southwest::CTMRGCornerTensor,
        E_north::CTMRGEdgeTensor{T, S, N}, E_south::CTMRGEdgeTensor{T, S, N},
    ) where {T, S, N}
    C_northwest_e = tensorexpr(:C_northwest, (envlabel(:W),), (envlabel(:NW),))
    C_northeast_e = tensorexpr(:C_northeast, (envlabel(:NE),), (envlabel(:E),))
    C_southeast_e = tensorexpr(:C_southeast, (envlabel(:E),), (envlabel(:SE),))
    C_southwest_e = tensorexpr(:C_southwest, (envlabel(:SW),), (envlabel(:W),))

    E_north_e = tensorexpr(
        :E_north, (envlabel(:NW), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:NE),)
    )
    E_south_e = tensorexpr(
        :E_south, (envlabel(:SE), ntuple(i -> virtuallabel(i), N - 1)...), (envlabel(:SW),)
    )

    rhs = Expr(
        :call, :*,
        C_northwest_e, E_north_e, C_northeast_e, C_southeast_e, E_south_e, C_southwest_e,
    )

    return macroexpand(@__MODULE__, :(return @autoopt @tensor $rhs))
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
        storagetype(E),
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
            randn(storagetype(t), dim(space(t, 1)))
        end
    else
        all(dim.(space.(noise_peps.A, 1)) .== length.(state_vector)) ||
            throw(ArgumentError("state vectors must match the physical dimension"))
    end
    prod_tensors = map(zip(noise_peps.A, state_vector)) do (t, v)
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
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
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
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)],
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)],
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)],
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)],
        env.edges[NORTH, _prev(r, end), c], env.edges[EAST, r, _next(c, end)],
        env.edges[SOUTH, _next(r, end), c], env.edges[WEST, r, _prev(c, end)],
        sandwich´,
    )
end

function contract_local_tensor(inds::CartesianIndex, O::AbstractTensorMap, env::CTMRGEnv)
    return contract_local_tensor(Tuple(inds), O, env)
end

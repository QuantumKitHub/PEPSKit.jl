"""
    expectation_value(peps::InfinitePEPS, O::LocalOperator, env::CTMRGEnv)

Compute the expectation value ⟨peps|O|peps⟩ / ⟨peps|peps⟩ of a [`LocalOperator`](@ref) `O`
for a PEPS `peps` using a given CTMRG environment `env`.
"""
function MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, env::CTMRGEnv)
    checklattice(peps, O)
    term_vals = dtmap([O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        contract_local_operator(inds, operator, peps, peps, env) /
        contract_local_norm(inds, peps, peps, env)
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
    op::Pair{CartesianIndex{2},<:AbstractTensorMap{T,S,2,2}},
    env::CTMRGEnv,
) where {T,S}
    return contract_local_tensor(op[1], op[2], env) /
           contract_local_tensor(op[1], pf[op[1]], env)
end
function MPSKit.expectation_value(
    pf::InfinitePartitionFunction, op::Pair{Tuple{Int,Int}}, env::CTMRGEnv
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
        isapprox(imag(E), 0; atol=sqrt(eps(real(E)))) ||
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
               _contract_vertical_edges((r, c), env) /
               _contract_horizontal_edges((r, c), env)
    end
end
network_value(state, env::CTMRGEnv) = network_value(InfiniteSquareNetwork(state), env)

"""
    _contract_site(ind::Tuple{Int,Int}, network::InfiniteSquareNetwork, env::CTMRGEnv)

Contract around a single site `ind` of a square network using a given CTMRG environment.
"""
function _contract_site(
    ind::Tuple{Int,Int}, network::InfiniteSquareNetwork{<:PEPSSandwich}, env::CTMRGEnv
)
    r, c = ind
    return @autoopt @tensor env.edges[WEST, r, _prev(c, end)][
            χ_WSW D_W_above D_W_below
            χ_WNW
        ] *
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][χ_WNW; χ_NNW] *
        env.edges[NORTH, _prev(r, end), c][χ_NNW D_N_above D_N_below; χ_NNE] *
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)][χ_NNE; χ_ENE] *
        env.edges[EAST, r, _next(c, end)][χ_ENE D_E_above D_E_below; χ_ESE] *
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)][χ_ESE; χ_SSE] *
        env.edges[SOUTH, _next(r, end), c][χ_SSE D_S_above D_S_below; χ_SSW] *
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)][χ_SSW; χ_WSW] *
        ket(network[r, c])[d; D_N_above D_E_above D_S_above D_W_above] *
        conj(bra(network[r, c])[d; D_N_below D_E_below D_S_below D_W_below])
end
function _contract_site(
    ind::Tuple{Int,Int}, network::InfiniteSquareNetwork{<:PFSandwich}, env::CTMRGEnv
)
    r, c = ind
    return @autoopt @tensor env.edges[WEST, r, _prev(c, end)][χ_WSW D_W; χ_WNW] *
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][χ_WNW; χ_NNW] *
        env.edges[NORTH, _prev(r, end), c][χ_NNW D_N; χ_NNE] *
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)][χ_NNE; χ_ENE] *
        env.edges[EAST, r, _next(c, end)][χ_ENE D_E; χ_ESE] *
        env.corners[SOUTHEAST, _next(r, end), _next(c, end)][χ_ESE; χ_SSE] *
        env.edges[SOUTH, _next(r, end), c][χ_SSE D_S; χ_SSW] *
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)][χ_SSW; χ_WSW] *
        tensor(network[r, c])[D_W D_S; D_N D_E]
end

"""
    _contract_corners(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract all corners around the south-east at position `ind` of the CTMRG
environment `env`.
"""
function _contract_corners(ind::Tuple{Int,Int}, env::CTMRGEnv)
    r, c = ind
    return tr(
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)] *
        env.corners[NORTHEAST, _prev(r, end), c] *
        env.corners[SOUTHEAST, r, c] *
        env.corners[SOUTHWEST, r, _prev(c, end)],
    )
end

"""
    _contract_vertical_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the vertical edges and corners around the east edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_vertical_edges(
    ind::Tuple{Int,Int}, env::CTMRGEnv{<:Any,<:CTMRG_PEPS_EdgeTensor}
)
    r, c = ind
    return @autoopt @tensor env.edges[WEST, r, _prev(c, end)][χ_SW D_above D_below; χ_NW] *
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][χ_NW; χ_N] *
        env.corners[NORTHEAST, _prev(r, end), c][χ_N; χ_NE] *
        env.edges[EAST, r, c][χ_NE D_above D_below; χ_SE] *
        env.corners[SOUTHEAST, _next(r, end), c][χ_SE; χ_S] *
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)][χ_S; χ_SW]
end
function _contract_vertical_edges(
    ind::Tuple{Int,Int}, env::CTMRGEnv{<:Any,<:CTMRG_PF_EdgeTensor}
)
    r, c = ind
    return @autoopt @tensor env.edges[WEST, r, _prev(c, end)][χ_SW D; χ_NW] *
        env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][χ_NW; χ_N] *
        env.corners[NORTHEAST, _prev(r, end), c][χ_N; χ_NE] *
        env.edges[EAST, r, c][χ_NE D; χ_SE] *
        env.corners[SOUTHEAST, _next(r, end), c][χ_SE; χ_S] *
        env.corners[SOUTHWEST, _next(r, end), _prev(c, end)][χ_S; χ_SW]
end

"""
    _contract_horizontal_edges(ind::Tuple{Int,Int}, env::CTMRGEnv)

Contract the horizontal edges and corners around the south edge at position `ind` of the
CTMRG environment `env`.
"""
function _contract_horizontal_edges(
    ind::Tuple{Int,Int}, env::CTMRGEnv{<:Any,<:CTMRG_PEPS_EdgeTensor}
)
    r, c = ind
    return @autoopt @tensor env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][
            χ_W
            χ_NW
        ] *
        env.edges[NORTH, _prev(r, end), c][χ_NW D_above D_below; χ_NE] *
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)][χ_NE; χ_E] *
        env.corners[SOUTHEAST, r, _next(c, end)][χ_E; χ_SE] *
        env.edges[SOUTH, r, c][χ_SE D_above D_below; χ_SW] *
        env.corners[SOUTHWEST, r, _prev(c, end)][χ_SW; χ_W]
end
function _contract_horizontal_edges(
    ind::Tuple{Int,Int}, env::CTMRGEnv{<:Any,<:CTMRG_PF_EdgeTensor}
)
    r, c = ind
    return @autoopt @tensor env.corners[NORTHWEST, _prev(r, end), _prev(c, end)][
            χ_W
            χ_NW
        ] *
        env.edges[NORTH, _prev(r, end), c][χ_NW D; χ_NE] *
        env.corners[NORTHEAST, _prev(r, end), _next(c, end)][χ_NE; χ_E] *
        env.corners[SOUTHEAST, r, _next(c, end)][χ_E; χ_SE] *
        env.edges[SOUTH, r, c][χ_SE D; χ_SW] *
        env.corners[SOUTHWEST, r, _prev(c, end)][χ_SW; χ_W]
end

function MPSKit.transfer_spectrum(
    above::MultilineMPS,
    O::MultilineTransferMatrix,
    below::MultilineMPS;
    num_vals=2,
    solver=MPSKit.Defaults.eigsolver,
)
    @assert size(above) == size(O)
    @assert size(below) == size(O)

    numrows = size(above, 1)
    eigenvals = Vector{Vector{scalartype(above)}}(undef, numrows)

    @threads for cr in 1:numrows
        L0 = MPSKit.randomize!(MPSKit.allocate_GL(above[cr - 1], O[cr], below[cr + 1], 1))

        E_LL = MPSKit.TransferMatrix(above[cr - 1].AL, O[cr], below[cr + 1].AL)  # Note that this index convention is different from above!
        λ, _, convhist = eigsolve(flip(E_LL), L0, num_vals, :LM, solver)
        convhist.converged < num_vals &&
            @warn "correlation length failed to converge: normres = $(convhist.normres)"
        eigenvals[cr] = λ
    end

    return eigenvals
end

"""
Adjoint of an MPS tensor, but permutes the physical spaces back into the codomain.
Intuitively, this conjugates a tensor and then reinterprets its 'direction' as an MPS
tensor.
"""
function _dag(A::MPSKit.GenericMPSTensor{S,N}) where {S,N}
    return permute(A', ((1, (3:(N + 1))...), (2,)))
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
MPSKit.correlation_length(state, env::CTMRGEnv; num_vals=2, kwargs...) =
    _correlation_length(env; num_vals, kwargs...)

function _correlation_length(env::CTMRGEnv; num_vals=2, kwargs...)
    T = scalartype(env)
    ξ_h = Vector{real(T)}(undef, size(env, 2))
    ξ_v = Vector{real(T)}(undef, size(env, 3))
    λ_h = Vector{Vector{T}}(undef, size(env, 2))
    λ_v = Vector{Vector{T}}(undef, size(env, 3))

    # Horizontal
    λ_h = map(1:size(env, 2)) do r
        above = InfiniteMPS(env.edges[NORTH, r, :])
        below = InfiniteMPS(_dag.(env.edges[SOUTH, r, :]))
        vals = MPSKit.transfer_spectrum(above; below, num_vals, kwargs...)
        return vals ./ abs(vals[1]) # normalize largest eigenvalue
    end
    ξ_h = map(λ_row -> -1 / log(abs(λ_row[2])), λ_h)

    # Vertical
    λ_v = map(1:size(env, 3)) do c
        above = InfiniteMPS(env.edges[EAST, :, c])
        below = InfiniteMPS(_dag.(env.edges[WEST, :, c]))
        vals = MPSKit.transfer_spectrum(above; below, num_vals, kwargs...)
        return vals ./ abs(vals[1]) # normalize largest eigenvalue
    end
    ξ_v = map(λ_row -> -1 / log(abs(λ_row[2])), λ_v)

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
function product_peps(peps_args...; unitcell=(1, 1), noise_amp=1e-2, state_vector=nothing)
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
    prod_tensors = map(zip(noise_peps.A, state_vector)) do (t, v)
        pt = zero(t)
        pt[][:, 1, 1, 1, 1] .= v
        return pt
    end
    prod_peps = InfinitePEPS(prod_tensors)
    ψ = prod_peps + noise_amp * noise_peps
    return ψ / norm(ψ)
end

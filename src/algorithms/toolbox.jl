"""
    expectation_value(peps::InfinitePEPS, O::LocalOperator, envs::CTMRGEnv)

Compute the expectation value ⟨peps|O|peps⟩ / ⟨peps|peps⟩ of a [`LocalOperator`](@ref) `O`
for a PEPS `peps` using a given CTMRG environment `envs`.
"""
function MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, envs::CTMRGEnv)
    checklattice(peps, O)
    term_vals = dtmap([O.terms...]) do (inds, operator)  # OhMyThreads can't iterate over O.terms directly
        contract_local_operator(inds, operator, peps, peps, envs) /
        contract_local_norm(inds, peps, peps, envs)
    end
    return sum(term_vals)
end
"""
    expectation_value(inds, O, pf::InfinitePartitionFunction, envs::CTMRGEnv)
    expectation_value(inds => O, pf::InfinitePartitionFunction, envs::CTMRGEnv)

Compute the expectation value corresponding to inserting a (set of) local tensor(s) `O` at
position `inds` in the partition function `pf` and contracting the chole using a given CTMRG
environment `envs`.

Here `inds` can be a single index or a tuple of indices, each specified as a
`Tuple{Int,Int}` or a `CartesianIndex{2}`. `O` can be either a single tensor or a matrix of
tensors respectively, each specified as an `AbstractTensorMap{S,2,2}` conforming to the
[`PartitionFunctionTensor`](@ref) indexing convention.

Alternatively, `O` can be a single higher-rank tensor map, in which case it is inserted
inside a rectangular region defined by the indices in `inds`, where in addition to the
[`PartitionFunctionTensor`](@ref) indexing convention its spaces within each direction are
ordered according to the axis directions of the usual unit cell convention.
"""
function MPSKit.expectation_value(
    inds::NTuple{N,CartesianIndex{2}},
    O::Union{AbstractTensorMap{S,M,M},Matrix{<:AbstractTensorMap{S,2,2}}},
    pf::InfinitePartitionFunction,
    envs::CTMRGEnv,
) where {N,S,M}
    if O isa Matrix
        (length(inds) != length(O)) &&
            throw(ArgumentError("Indices and tensor matrix must match"))
    else
        rmin, rmax = minimum(x -> x.I[1], inds), maximum(x -> x.I[1], inds)
        cmin, cmax = minimum(x -> x.I[2], inds), maximum(x -> x.I[2], inds)
        (M != rmax - rmin + cmax - cmin + 2) &&
            throw(ArgumentError("Indices don't match rectangular patch size"))
    end
    return contract_local_tensor(inds, O, envs) / contract_local_tensor(inds, pf.A, envs)
end
function MPSKit.expectation_value(
    inds::NTuple{N,Tuple{Int,Int}},
    O::Union{AbstractTensorMap{S,M,M},Matrix{<:AbstractTensorMap{S,2,2}}},
    pf::InfinitePartitionFunction,
    envs::CTMRGEnv,
) where {N,S,M}
    return expectation_value(CartesianIndex.(inds), O, pf, envs)
end
function MPSKit.expectation_value(
    inds::Union{Tuple{Int,Int},CartesianIndex{2}},
    O::AbstractTensorMap{S,2,2},
    pf::InfinitePartitionFunction,
    envs::CTMRGEnv,
) where {S}
    return expectation_value((inds,), [O;;], pf, envs)
end
function MPSKit.expectation_value(op::Pair, pf::InfinitePartitionFunction, envs::CTMRGEnv)
    return expectation_value(first.(op), last.(op), pf, envs)
end

function costfun(peps::InfinitePEPS, envs::CTMRGEnv, O::LocalOperator)
    E = MPSKit.expectation_value(peps, O, envs)
    ignore_derivatives() do
        isapprox(imag(E), 0; atol=sqrt(eps(real(E)))) ||
            @warn "Expectation value is not real: $E."
    end
    return real(E)
end

function LinearAlgebra.norm(peps::InfinitePEPS, env::CTMRGEnv)
    total = one(scalartype(peps))

    for r in 1:size(peps, 1), c in 1:size(peps, 2)
        rprev = _prev(r, size(peps, 1))
        rnext = _next(r, size(peps, 1))
        cprev = _prev(c, size(peps, 2))
        cnext = _next(c, size(peps, 2))
        total *= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.edges[NORTH, rprev, c][χ3 D3 D4; χ4] *
            env.corners[NORTHEAST, rprev, cnext][χ4; χ5] *
            env.edges[EAST, r, cnext][χ5 D5 D6; χ6] *
            env.corners[SOUTHEAST, rnext, cnext][χ6; χ7] *
            env.edges[SOUTH, rnext, c][χ7 D7 D8; χ8] *
            env.corners[SOUTHWEST, rnext, cprev][χ8; χ1] *
            peps[r, c][d; D3 D5 D7 D1] *
            conj(peps[r, c][d; D4 D6 D8 D2])
        total *= tr(
            env.corners[NORTHWEST, rprev, cprev] *
            env.corners[NORTHEAST, rprev, c] *
            env.corners[SOUTHEAST, r, c] *
            env.corners[SOUTHWEST, r, cprev],
        )
        total /= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1 D2; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.corners[NORTHEAST, rprev, c][χ3; χ4] *
            env.edges[EAST, r, c][χ4 D1 D2; χ5] *
            env.corners[SOUTHEAST, rnext, c][χ5; χ6] *
            env.corners[SOUTHWEST, rnext, cprev][χ6; χ1]
        total /= @autoopt @tensor env.corners[NORTHWEST, rprev, cprev][χ1; χ2] *
            env.edges[NORTH, rprev, c][χ2 D1 D2; χ3] *
            env.corners[NORTHEAST, rprev, cnext][χ3; χ4] *
            env.corners[SOUTHEAST, r, cnext][χ4; χ5] *
            env.edges[SOUTH, r, c][χ5 D1 D2; χ6] *
            env.corners[SOUTHWEST, r, cprev][χ6; χ1]
    end

    return total
end

"""
    value(partfunc::InfinitePartitionFunction, env::CTMRGEnv)

Return the value (per site) of a given partition function contracted using a given CTMRG
environment.
"""
function value(partfunc::InfinitePartitionFunction, env::CTMRGEnv)
    total = one(scalartype(partfunc))

    for r in 1:size(partfunc, 1), c in 1:size(partfunc, 2)
        rprev = _prev(r, size(partfunc, 1))
        rnext = _next(r, size(partfunc, 1))
        cprev = _prev(c, size(partfunc, 2))
        cnext = _next(c, size(partfunc, 2))
        total *= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.edges[NORTH, rprev, c][χ3 D3; χ4] *
            env.corners[NORTHEAST, rprev, cnext][χ4; χ5] *
            env.edges[EAST, r, cnext][χ5 D5; χ6] *
            env.corners[SOUTHEAST, rnext, cnext][χ6; χ7] *
            env.edges[SOUTH, rnext, c][χ7 D7; χ8] *
            env.corners[SOUTHWEST, rnext, cprev][χ8; χ1] *
            partfunc[r, c][D1 D7; D3 D5]
        total *= tr(
            env.corners[NORTHWEST, rprev, cprev] *
            env.corners[NORTHEAST, rprev, c] *
            env.corners[SOUTHEAST, r, c] *
            env.corners[SOUTHWEST, r, cprev],
        )
        total /= @autoopt @tensor env.edges[WEST, r, cprev][χ1 D1; χ2] *
            env.corners[NORTHWEST, rprev, cprev][χ2; χ3] *
            env.corners[NORTHEAST, rprev, c][χ3; χ4] *
            env.edges[EAST, r, c][χ4 D1; χ5] *
            env.corners[SOUTHEAST, rnext, c][χ5; χ6] *
            env.corners[SOUTHWEST, rnext, cprev][χ6; χ1]
        total /= @autoopt @tensor env.corners[NORTHWEST, rprev, cprev][χ1; χ2] *
            env.edges[NORTH, rprev, c][χ2 D1; χ3] *
            env.corners[NORTHEAST, rprev, cnext][χ3; χ4] *
            env.corners[SOUTHEAST, r, cnext][χ4; χ5] *
            env.edges[SOUTH, r, c][χ5 D1; χ6] *
            env.corners[SOUTHWEST, r, cprev][χ6; χ1]
    end

    return total
end

"""
    correlation_length(peps::InfinitePEPS, env::CTMRGEnv; num_vals=2)

Compute the PEPS correlation length based on the horizontal and vertical
transfer matrices. Additionally the (normalized) eigenvalue spectrum is
returned. Specify the number of computed eigenvalues with `num_vals`.
"""
function MPSKit.correlation_length(peps::InfinitePEPS, env::CTMRGEnv; num_vals=2)
    T = scalartype(peps)
    ξ_h = Vector{real(T)}(undef, size(peps, 1))
    ξ_v = Vector{real(T)}(undef, size(peps, 2))
    λ_h = Vector{Vector{T}}(undef, size(peps, 1))
    λ_v = Vector{Vector{T}}(undef, size(peps, 2))

    # Horizontal
    above_h = MPSMultiline(map(r -> InfiniteMPS(env.edges[1, r, :]), 1:size(peps, 1)))
    respaced_edges_h = map(zip(space.(env.edges)[1, :, :], env.edges[3, :, :])) do (V1, T3)
        return TensorMap(T3.data, V1)
    end
    below_h = MPSMultiline(map(r -> InfiniteMPS(respaced_edges_h[r, :]), 1:size(peps, 1)))
    transfer_peps_h = TransferPEPSMultiline(peps, NORTH)
    vals_h = MPSKit.transfer_spectrum(above_h, transfer_peps_h, below_h; num_vals)
    λ_h = map(λ_row -> λ_row / abs(λ_row[1]), vals_h)  # Normalize largest eigenvalue
    ξ_h = map(λ_row -> -1 / log(abs(λ_row[2])), λ_h)

    # Vertical
    above_v = MPSMultiline(map(c -> InfiniteMPS(env.edges[2, :, c]), 1:size(peps, 2)))
    respaced_edges_v = map(zip(space.(env.edges)[2, :, :], env.edges[4, :, :])) do (V2, T4)
        return TensorMap(T4.data, V2)
    end
    below_v = MPSMultiline(map(c -> InfiniteMPS(respaced_edges_v[:, c]), 1:size(peps, 2)))
    transfer_peps_v = TransferPEPSMultiline(peps, EAST)
    vals_v = MPSKit.transfer_spectrum(above_v, transfer_peps_v, below_v; num_vals)
    λ_v = map(λ_row -> λ_row / abs(λ_row[1]), vals_v)  # Normalize largest eigenvalue
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

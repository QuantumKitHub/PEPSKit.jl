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
    expectation_value(pf::InfinitePartitionFunction, inds => O, envs::CTMRGEnv)

Compute the expectation value corresponding to inserting a local tensor(s) `O` at
position `inds` in the partition function `pf` and contracting the chole using a given CTMRG
environment `envs`.

Here `inds` can be specified as either a `Tuple{Int,Int}` or a `CartesianIndex{2}`, and `O`
should be a rank-4 tensor conforming to the [`PartitionFunctionTensor`](@ref) indexing
convention.
"""
function MPSKit.expectation_value(
    pf::InfinitePartitionFunction,
    op::Pair{CartesianIndex{2},<:AbstractTensorMap{T,S,2,2}},
    envs::CTMRGEnv,
) where {T,S}
    return contract_local_tensor(op[1], op[2], envs) /
           contract_local_tensor(op[1], pf[op[1]], envs)
end
function MPSKit.expectation_value(
    pf::InfinitePartitionFunction, op::Pair{Tuple{Int,Int}}, envs::CTMRGEnv
)
    return expectation_value(pf, CartesianIndex(op[1]) => op[2], envs)
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

"""
    correlation_length(env::CTMRGEnv; num_vals=2, kwargs...)

Compute the correlation length associated to the environment of a given state based on the
horizontal and vertical transfer matrices. Additionally the (normalized) eigenvalue spectrum
is returned. The number of computed eigenvalues can be specified using `num_vals`, and any
remaining keyword arguments are passed through to `MPSKit.correlation_length` (e.g. allowing
to target the correlation length in a specific symmetry sector).
"""
function MPSKit.correlation_length(env::CTMRGEnv; num_vals=2, kwargs...)
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

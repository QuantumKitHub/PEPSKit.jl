function MPSKit.expectation_value(peps::InfinitePEPS, O::LocalOperator, envs::CTMRGEnv)
    checklattice(peps, O)
    return sum(O.terms) do (inds, operator)  # TODO: parallelize this map, especially for the backwards pass
        contract_localoperator(inds, operator, peps, peps, envs) /
        contract_localnorm(inds, peps, peps, envs)
    end
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

function LinearAlgebra.norm(ipeps::InfinitePEPS, env::VUMPSEnv)
    @unpack ACu, ARu, ACd, ARd, FLu, FRu, FLo, FRo = env
    Ni, Nj = size(ipeps)
    # total = 1
    # for j in 1:Nj, i in 1:Ni
    #     ir = mod1(i + 1, Ni)
    #     @tensoropt Z = FLo[i,j][6 5 4; 1] * ACu[i,j][1 2 3; -4] * ipeps[i,j][9; 2 -2 8 5] * 
    #     ipeps[i,j]'[3 -3 7 4; 9] * AC[ir,j]'[-1; 6 8 7] * FR[i,j][-4 -2 -3; -1]
    #     @tensor n = FL[i,j][1 2 3; 4] * C[i,j][4; 5] * FR[i,j][5 2 3; 6] * C[i,j]'[6; 1]
    #     total *= Z / n
    # end

    itp = InfiniteTransferPEPS(ipeps)
    λFL, _ = rightenv(ARu, adjoint.(ARd), itp; ifobs=true)
    # to do:
    λC, _ = rightCenv(ARu, adjoint.(ARd); ifobs=true)
    @show λC
    
    return prod(λFL ./ λC)^(1/Ni)
end
"""
    correlation_length(peps::InfinitePEPS, env::CTMRGEnv; howmany=2)

Compute the PEPS correlation length based on the horizontal and vertical
transfer matrices. Additionally the (normalized) eigenvalue spectrum is
returned. Specify the number of computed eigenvalues with `howmany`.
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

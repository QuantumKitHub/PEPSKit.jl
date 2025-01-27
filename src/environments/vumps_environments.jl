using MPSKit: InfiniteEnvironments

## Actual overloads: purely due to product structure in virtual spaces of 'effective' MPO
## tensors

function MPSKit.issamespace(
    envs::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS,
)
    L = MPSKit.check_length(above, operator, below)
    for i in 1:L
        space(envs.GLs[i]) == (
            left_virtualspace(below, i) ⊗
            _elementwise_dual(left_virtualspace(operator, i)) ← left_virtualspace(above, i)
        ) || return false
        space(envs.GRs[i]) == (
            right_virtualspace(above, i) ⊗ right_virtualspace(operator, i) ←
            right_virtualspace(below, i)
        ) || return false
    end
    return true
end

function MPSKit.allocate_GL(
    bra::InfiniteMPS, mpo::InfiniteTransferMatrix, ket::InfiniteMPS, i::Int
)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V =
        left_virtualspace(bra, i) ⊗ _elementwise_dual(left_virtualspace(mpo, i)) ←
        left_virtualspace(ket, i)
    TT = TensorMap{T}
    return TT(undef, V)
end

function MPSKit.allocate_GR(
    bra::InfiniteMPS, mpo::InfiniteTransferMatrix, ket::InfiniteMPS, i::Int
)
    T = Base.promote_type(scalartype(bra), scalartype(mpo), scalartype(ket))
    V = right_virtualspace(ket, i) ⊗ right_virtualspace(mpo, i) ← right_virtualspace(bra, i)
    TT = TensorMap{T}
    return TT(undef, V)
end

## All the rest: purely relaxation of type restriction of AbstractMPO{O<:MPOTensor}

function MPSKit.environment_alg(
    ::Union{InfiniteMPS,MultilineMPS},
    ::Union{InfiniteTransferMatrix,MultilineTransferMatrix},
    ::Union{InfiniteMPS,MultilineMPS};
    tol=MPSKit.Defaults.tol,
    maxiter=MPSKit.Defaults.maxiter,
    krylovdim=MPSKit.Defaults.krylovdim,
    verbosity=MPSKit.Defaults.VERBOSE_NONE,
    eager=true,
)
    return Arnoldi(; tol, maxiter, krylovdim, verbosity, eager)
end

function MPSKit.environments(
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS=above;
    kwargs...,
)
    GLs, GRs = MPSKit.initialize_environments(above, operator, below)
    envs = InfiniteEnvironments(GLs, GRs)
    return MPSKit.recalculate!(envs, above, operator, below; kwargs...)
end

function MPSKit.environments(
    above::MultilineMPS,
    operator::MultilineTransferMatrix,
    below::MultilineMPS=above;
    kwargs...,
)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    envs = map(1:rows) do row
        return environments(above[row], operator[row], below[row + 1]; kwargs...)
    end
    return Multiline(PeriodicVector(envs))
end

function MPSKit.recalculate!(
    envs::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS=above;
    kwargs...,
)
    if !MPSKit.issamespace(envs, above, operator, below)
        # TODO: in-place initialization?
        GLs, GRs = MPSKit.initialize_environments(above, operator, below)
        copy!(envs.GLs, GLs)
        copy!(envs.GRs, GRs)
    end

    alg = MPSKit.environment_alg(above, operator, below; kwargs...)

    @sync begin
        @spawn MPSKit.compute_leftenvs!(envs, above, operator, below, alg)
        @spawn MPSKit.compute_rightenvs!(envs, above, operator, below, alg)
    end
    MPSKit.normalize!(envs, above, operator, below)

    return envs
end

function MPSKit.recalculate!(
    envs::MultilineEnvironments,
    above::MultilineMPS,
    operator::MultilineTransferMatrix,
    below::MultilineMPS=above;
    kwargs...,
)
    (rows = size(above, 1)) == size(operator, 1) == size(below, 1) ||
        throw(ArgumentError("Incompatible sizes"))
    @threads for row in 1:rows
        MPSKit.recalculate!(envs[row], above[row], operator[row], below[row + 1]; kwargs...)
    end
    return envs
end

function MPSKit.initialize_environments(
    above::InfiniteMPS, operator::InfiniteTransferMatrix, below::InfiniteMPS=above
)
    L = MPSKit.check_length(above, operator, below)
    GLs = PeriodicVector([
        MPSKit.randomize!(MPSKit.allocate_GL(below, operator, above, i)) for i in 1:L
    ])
    GRs = PeriodicVector([
        MPSKit.randomize!(MPSKit.allocate_GR(below, operator, above, i)) for i in 1:L
    ])
    return GLs, GRs
end

function MPSKit.compute_leftenvs!(
    envs::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS,
    alg,
)
    # compute eigenvector
    T = MPSKit.TransferMatrix(above.AL, operator, below.AL)
    λ, envs.GLs[1] = MPSKit.fixedpoint(flip(T), envs.GLs[1], :LM, alg)
    # push through unitcell
    for i in 2:length(operator)
        envs.GLs[i] =
            envs.GLs[i - 1] *
            MPSKit.TransferMatrix(above.AL[i - 1], operator[i - 1], below.AL[i - 1])
    end
    return λ, envs
end

function MPSKit.compute_rightenvs!(
    envs::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS,
    alg,
)
    # compute eigenvector
    T = MPSKit.TransferMatrix(above.AR, operator, below.AR)
    λ, envs.GRs[end] = MPSKit.fixedpoint(T, envs.GRs[end], :LM, alg)
    # push through unitcell
    for i in reverse(1:(length(operator) - 1))
        envs.GRs[i] =
            MPSKit.TransferMatrix(above.AR[i + 1], operator[i + 1], below.AR[i + 1]) *
            envs.GRs[i + 1]
    end
    return λ, envs
end

function TensorKit.normalize!(
    envs::InfiniteEnvironments,
    above::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    below::InfiniteMPS,
)
    for i in 1:length(operator)
        λ = dot(below.C[i], MPSKit.∂C(above.C[i], envs.GLs[i + 1], envs.GRs[i]))
        scale!(envs.GLs[i + 1], inv(λ))
    end
    return envs
end

function MPSKit.leftenv(envs::MultilineEnvironments, r::Int, c::Int, state)
    return leftenv(envs[r], c, parent(state))
end
function MPSKit.rightenv(envs::MultilineEnvironments, r::Int, c::Int, state)
    return rightenv(envs[r], c, parent(state))
end

function MPSKit.expectation_value(
    ψ::MultilineMPS,
    O::MultilineTransferMatrix,
    envs::MultilineEnvironments=environments(ψ, O),
)
    return prod(product(1:size(ψ, 1), 1:size(ψ, 2))) do (i, j)
        GL = leftenv(envs, i, j, ψ)
        GR = rightenv(envs, i, j, ψ)
        return contract_mpo_expval(ψ.AC[i, j], GL, O[i, j], GR, ψ.AC[i + 1, j])
    end
end
function MPSKit.expectation_value(st::InfiniteMPS, transfer::InfiniteTransferMatrix)
    return expectation_value(Multiline([st]), Multiline([transfer]))
end

function MPSKit.calc_galerkin(
    pos::CartesianIndex{2},
    above::MultilineMPS,
    operator::MultilineTransferMatrix,
    below::MultilineMPS,
    envs::MultilineEnvironments,
)
    row, col = pos.I
    return MPSKit.calc_galerkin(col, above[row], operator[row], below[row + 1], envs[row])
end
function MPSKit.calc_galerkin(
    above::MultilineMPS,
    operator::MultilineTransferMatrix,
    below::MultilineMPS,
    envs::MultilineEnvironments,
)
    return maximum(
        pos -> MPSKit.calc_galerkin(pos, above, operator, below, envs),
        CartesianIndices(size(above)),
    )
end

function MPSKit.leading_boundary(
    state::InfiniteMPS,
    operator::InfiniteTransferMatrix,
    alg,
    envs=environments(state, operator),
)
    state_multi = convert(MultilineMPS, state)
    operator_multi = Multiline([operator])
    envs_multi = Multiline([envs])
    state_multi′, envs_multi′, err = leading_boundary(
        state_multi, operator_multi, alg, envs_multi
    )
    state′ = convert(InfiniteMPS, state_multi′)
    envs´ = convert(InfiniteEnvironments, envs_multi′)
    return state′, envs´, err
end

# TODO: remove this once it's in MPSKit
Base.convert(::Type{InfiniteEnvironments}, envs::MultilineEnvironments) = only(envs)

# this is a total pain...
function MPSKit._vumps_localupdate(
    col, ψ::MultilineMPS, O::MultilineTransferMatrix, envs, eigalg, factalg=QRpos()
)
    local AC′, C′
    if Defaults.scheduler[] isa Defaults.SerialScheduler
        _, AC′ = MPSKit.fixedpoint(MPSKit.∂∂AC(col, ψ, O, envs), ψ.AC[:, col], :LM, eigalg)
        _, C′ = MPSKit.fixedpoint(MPSKit.∂∂C(col, ψ, O, envs), ψ.C[:, col], :LM, eigalg)
    else
        @sync begin
            Threads.@spawn begin
                _, AC′ = MPSKit.fixedpoint(
                    MPSKit.∂∂AC(col, ψ, O, envs), ψ.AC[:, col], :LM, eigalg
                )
            end
            Threads.@spawn begin
                _, C′ = MPSKit.fixedpoint(
                    MPSKit.∂∂C(col, ψ, O, envs), ψ.C[:, col], :LM, eigalg
                )
            end
        end
    end
    return MPSKit.regauge!.(AC′, C′; alg=factalg)[:]
end

# Bind decomposition-owned warm starts to the existing projector implementations.
const SubspaceSVD = SVDAdjoint{<:SubspaceIteration}
const TruncatedSubspaceSVD = SVDAdjoint{<:TruncatedAlgorithm{<:SubspaceIteration}}

function _prepare_ctmrg(alg::CTMRGAlgorithm, env::CTMRGEnv)
    svd = decomposition_algorithm(alg.projector_alg)
    svd isa SubspaceSVD || return alg
    fwd = svd.fwd_alg
    fwd.cache isa AbstractArray && return alg
    entries = ignore_derivatives() do
        map(CartesianIndices(size(env))) do _
            local_alg = @set fwd.cache = Ref{Any}(nothing)
            return @set local_alg.rng = Random.Xoshiro(rand(fwd.rng, UInt))
        end
    end
    return @set alg.projector_alg.decomposition_alg.fwd_alg.cache = entries
end

function _select_decomposition(svd::SubspaceSVD, coordinate)
    isnothing(coordinate) && return svd
    return svd.fwd_alg.cache isa AbstractArray ? (@set svd.fwd_alg = svd.fwd_alg.cache[coordinate...]) : svd
end

function _rotate_decomposition(alg::SequentialCTMRG{<:Union{HalfInfiniteProjector{<:SubspaceSVD}, FullInfiniteProjector{<:SubspaceSVD}}})
    grid = alg.projector_alg.decomposition_alg.fwd_alg.cache
    rotated = reverse(permutedims(circshift(grid, (-1, 0, 0)), (1, 3, 2)); dims = 2)
    return @set alg.projector_alg.decomposition_alg.fwd_alg.cache = rotated
end

function _decomposition_metrics(alg::CTMRGAlgorithm)
    svd = decomposition_algorithm(alg.projector_alg)
    svd isa SubspaceSVD || return (;)
    error = maximum(svd.fwd_alg.cache) do entry
        return isnothing(entry.cache[]) ? Inf : entry.cache[].subspace_error
    end
    return (; subspace_error = error)
end

function _fix_decomposition_allocation(alg::CTMRGAlgorithm)
    svd = decomposition_algorithm(alg.projector_alg)
    svd isa SubspaceSVD || return alg
    entries = map(svd.fwd_alg.cache) do entry
        @set entry.charge_refresh = 0
    end
    return @set alg.projector_alg.decomposition_alg.fwd_alg.cache = entries
end

_environment_product(a, b, ::Union{SubspaceSVD, TruncatedSubspaceSVD}) = _factorized_environment(a, b)
function _projector_svd(M::FactorizedMap, alg::TruncatedSubspaceSVD)
    U, S, V, error = svd_trunc!(M, alg)
    # CTMRG removes this common scale; normalize only the small retained spectrum.
    scale = norm(S)
    Q = ignore_derivatives() do
        cache = alg.fwd_alg.alg.cache
        state = cache[]
        cache[] = (; state.X, state.iteration, state.subspace_error)
        return state.Q
    end
    # The reduced SVD already fixes phases; only its pullback needs this correction.
    if Zygote.isderiving()
        U, V = _subspace_svd_phase_correction(U, V, Q' * U)
    end
    return U, S / scale, V, error / scale
end

# These hooks select algorithm metadata and warm starts, not differentiable tensor data.
@non_differentiable _prepare_ctmrg(args...)
@non_differentiable _select_decomposition(args...)
@non_differentiable _rotate_decomposition(args...)
@non_differentiable _decomposition_metrics(args...)

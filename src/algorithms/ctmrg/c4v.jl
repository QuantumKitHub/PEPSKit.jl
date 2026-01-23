struct C4vCTMRG{P <: ProjectorAlgorithm} <: CTMRGAlgorithm
    tol::Float64
    maxiter::Int
    miniter::Int
    verbosity::Int
    projector_alg::P
end
function C4vCTMRG(; kwargs...)
    return CTMRGAlgorithm(; alg = :c4v, kwargs...)
end
CTMRG_SYMBOLS[:c4v] = C4vCTMRG

struct C4vEighProjector{S <: EighAdjoint, T} <: ProjectorAlgorithm
    alg::S
    trunc::T
    verbosity::Int
end
function C4vEighProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :c4v_eigh, kwargs...)
end
PROJECTOR_SYMBOLS[:c4v_eigh] = C4vEighProjector

# struct C4vQRProjector{S, T} <: ProjectorAlgorithm
#     alg::S
#     verbosity::Int
# end
# function C4vQRProjector(; kwargs...)
#     return ProjectorAlgorithm(; alg = :c4v_qr, kwargs...)
# end
# PROJECTOR_SYMBOLS[:c4v_qr] = C4vQRProjector

#
## C4v-symmetric CTMRG iteration (called through `leading_boundary`)
#

function ctmrg_iteration(
        network,
        env::CTMRGEnv,
        alg::C4vCTMRG,
    )
    enlarged_corner = c4v_enlarge(network, env, alg.projector_alg)
    corner′, projector, info = c4v_projector(enlarged_corner, alg.projector_alg)
    edge′ = c4v_renormalize(network, env, projector)
    return CTMRGEnv(corner′, edge′), info
end

function c4v_enlarge(network, env, ::C4vEighProjector)
    return TensorMap(EnlargedCorner(network, env, (NORTHWEST, 1, 1)))
end
# function c4v_enlarge(enlarged_corner, alg::C4vQRProjector)
#     # TODO
# end

function c4v_projector(enlarged_corner, alg::C4vEighProjector)
    hermitian_corner = 0.5 * (enlarged_corner + enlarged_corner') / norm(enlarged_corner)
    trunc = truncation_strategy(alg, enlarged_corner)
    D, U, info = eigh_trunc!(hermitian_corner, decomposition_algorithm(alg); trunc)

    # Check for degenerate eigenvalues
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(D)
            vals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(D))
            @warn("degenerate eigenvalues detected: ", vals)
        end
    end

    return D / norm(D), U, (; D, U, info...)
end
# function c4v_projector(enlarged_corner, alg::C4vQRProjector)
#     # TODO
# end

function c4v_renormalize(network, env, projector)
    new_edge = renormalize_north_edge(env.edges[1], projector, projector', network[1, 1])
    return new_edge / norm(new_edge)
end

function CTMRGEnv(
        corner::AbstractTensorMap{T, S, 1, 1}, edge::AbstractTensorMap{T′, S, N, 1}
    ) where {T, T′, S, N}
    corners = fill(corner, 4, 1, 1)
    edge_SW = physical_flip(edge)
    edges = reshape([edge, edge, edge_SW, edge_SW], (4, 1, 1))
    return CTMRGEnv(corners, edges)
end

#
## utility
#

# Adjoint of an edge tensor, but permutes the physical spaces back into the codomain.
# Intuitively, this conjugates a tensor and then reinterprets its 'direction' as an edge tensor.
function _dag(A::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    return permute(A', ((1, (3:(N + 1))...), (2,)))
end
function physical_flip(A::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    return flip(A, 2:N)
end
function project_hermitian(E::AbstractTensorMap{T, S, N, 1}) where {T, S, N}
    E´ = (E + physical_flip(_dag(E))) / 2
    return E´
end
function project_hermitian(C::AbstractTensorMap{T, S, 1, 1}) where {T, S}
    C´ = (C + C') / 2
    return C´
end

# should perform this check at the beginning of `leading_boundary` really...
function check_symmetry(state, ::RotateReflect; atol = 1.0e-10)
    @assert length(state) == 1 "check_symmetry only works for single site unit cells"
    @assert norm(state[1] - _fit_spaces(rotl90(state[1]), state[1])) /
        norm(state[1]) < atol "not rotation invariant"
    @assert norm(state[1] - _fit_spaces(herm_depth(state[1]), state[1])) /
        norm(state[1]) < atol "not hermitian-reflection invariant"
    return nothing
end

#
## environment initialization
#

# TODO: rewrite this using `initialize_environment` and C4v-specific initialization algorithms
# environment with dummy corner singlet(V) ← singlet(V) and identity edge V ← V, initialized at dim(Venv)
# function initialize_singlet_c4v_env(Vpeps::ElementarySpace, Venv::ElementarySpace, T = ComplexF64)
#     corner₀ = DiagonalTensorMap(zeros(real(T), Venv ← Venv))
#     corner₀.data[1] = one(real(T))
#     edge₀ = permute(id(T, Venv ⊗ Vpeps), ((1, 2, 4), (3,)))
#     return CTMRGEnv(corner₀, edge₀)
# end

function initialize_random_c4v_env(Vstate::ElementarySpace, Venv::ElementarySpace, T = ComplexF64)
    corner₀ = DiagonalTensorMap(randn(real(T), Venv ← Venv))
    edge₀ = randn(T, Venv ⊗ Vstate ← Venv)
    edge₀ = project_hermitian(edge₀)
    return CTMRGEnv(corner₀, edge₀)
end
function initialize_random_c4v_env(state::InfinitePEPS, Venv::ElementarySpace, T = scalartype(state))
    Vpeps = domain(state[1])[1]
    return initialize_random_c4v_env(Vpeps ⊗ Vpeps', Venv, T)
end
function initialize_random_c4v_env(state::InfinitePartitionFunction, Venv::ElementarySpace, T = scalartype(state))
    Vpf = domain(state[1])[1]
    return initialize_random_c4v_env(Vpf, Venv, T)
end

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
decomposition_algorithm(alg::C4vEighProjector) = alg.alg
PROJECTOR_SYMBOLS[:c4v_eigh] = C4vEighProjector

struct C4vQRProjector{S, T} <: ProjectorAlgorithm
    alg::S
    verbosity::Int
end
function C4vQRProjector(; kwargs...)
    return ProjectorAlgorithm(; alg = :c4v_qr, kwargs...)
end
decomposition_algorithm(alg::C4vEighProjector) = alg.alg
PROJECTOR_SYMBOLS[:c4v_qr] = C4vQRProjector

function ctmrg_iteration(
        network,
        env::CTMRGEnv,
        alg::C4vCTMRG,
    )
    enlarged_corner = TensorMap(EnlargedCorner(network, env, (NORTHWEST, 1, 1)))
    corner′, projector, info = c4v_projector(enlarged_corner, alg.projector_alg)
    edge′ = c4v_renormalize(network, env, projector)
    return _c4v_env(corner′, edge′), info
end

function c4v_projector(enlarged_corner, alg::C4vEighProjector)
    hermitian_corner = 0.5 * (enlarged_corner + enlarged_corner') / norm(enlarged_corner)
    trunc = truncation_strategy(alg, enlarged_corner)
    D, V, info = eigh_trunc!(hermitian_corner, decomposition_algorithm(alg); trunc)

    # Check for degenerate eigenvalues
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(D)
            vals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(D))
            @warn("degenerate eigenvalues detected: ", vals)
        end
    end

    return D / norm(D), V, (; D, V, info...)
end

function c4v_projector(enlarged_corner, alg::C4vQRProjector)
    # TODO
end

function c4v_renormalize(network, env, projector)
    new_edge = renormalize_north_edge(env.edges[1], projector, projector', network[1, 1])
    return new_edge / norm(new_edge)
end

# TODO: this won't differentiate properly probably due to custom CTMRGEnv rrule defined in PEPSKit
function CTMRGEnv(corner::CornerTensor, edge::EdgeTensor)
    corners = fill(corner, 4, 1, 1)
    edge_SW = physical_flip(edge)
    edges = reshape([edge, edge, edge_SW, edge_SW], (4, 1, 1))
    return CTMRGEnv(corners, edges)
end

function _c4v_env(corner::CornerTensor, edge::EdgeTensor)
    corners = fill(corner, 4, 1, 1)
    edge_SW = physical_flip(edge)
    edges = reshape([edge, edge, edge_SW, edge_SW], (4, 1, 1))
    return CTMRGEnv(corners, edges)
end

# environment with dummy corner singlet(V) ← singlet(V) and identity edge V ← V, initialized at dim(Venv)
function initialize_singlet_c4v_env(Vpeps::ElementarySpace, Venv::ElementarySpace, T = ComplexF64)
    corner₀ = DiagonalTensorMap(zeros(real(T), Venv ← Venv))
    corner₀.data[1] = one(real(T))
    edge₀ = permute(id(T, Venv ⊗ Vpeps), ((1, 2, 4), (3,)))
    return CTMRGEnv(corner₀, edge₀)
end

function initialize_random_c4v_env(Vpeps::ElementarySpace, Venv::ElementarySpace, T = ComplexF64)
    corner₀ = DiagonalTensorMap(randn(real(T), Venv ← Venv))
    edge₀ = randn(T, Venv ⊗ Vpeps ⊗ Vpeps' ← Venv)
    edge₀ = project_hermitian(edge₀)
    return CTMRGEnv(corner₀, edge₀)
end

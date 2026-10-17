"""Full-environment randomized SVD projector from PR 263."""
struct RandomizedProjector{S, T, R} <: ProjectorAlgorithm
    decomposition_alg::S
    trunc::T
    rng::R
    oversampling::Int
    max_full::Int
    n_subspace_iter::Int
    verbosity::Int
end

PROJECTOR_SYMBOLS[:randomized] = RandomizedProjector

decomposition_algorithm(alg::RandomizedProjector) = alg.decomposition_alg

_default_randomized_oversampling = 10
_default_randomized_max_full = 100
_default_n_subspace_iter = 2

# needed as default interface in PEPSKit.ProjectorAlgorithm
"""Construct the randomized projector with the original PR defaults."""
function RandomizedProjector(decomposition_alg, trunc, verbosity)
    return RandomizedProjector(
        decomposition_alg, trunc, Random.default_rng(), _default_randomized_oversampling,
        _default_randomized_max_full, _default_n_subspace_iter, verbosity
    )
end

"""Choose exact or oversampled sector dimensions using the previous environment space."""
function random_domain(alg::RandomizedProjector, full_space, last_space)
    sector_dims = map(sectors(full_space)) do s
        if dim(full_space, s) <= alg.max_full
            n = dim(full_space, s)
        else
            n = dim(last_space, s) + alg.oversampling
        end
        return s => n
    end
    return Vect[sectortype(last_space)](sector_dims)
end


"""Find each block range using exact bases for small blocks and randomized iteration otherwise."""
function randomized_range_finder(A::AbstractTensorMap, alg::RandomizedProjector, randomized_space)
    Q = TensorMap{eltype(A)}(undef, domain(A) ← randomized_space)
    foreach(blocks(Q)) do (s, b)
        Aad = A'
        m, n = size(b)
        if m <= alg.max_full
            b .= LinearAlgebra.I(m)
        else
            Ω = randn(alg.rng, eltype(A), domain(A) ← Vect[sectortype(A)](s => n))
            for _ in 1:alg.n_subspace_iter
                Y = A * Ω
                Qs, _ = left_orth!(Y)
                Ω, _ = left_orth!(Aad * Qs)
            end
            Y = A * Ω
            Qs, _ = left_orth!(Y)
            b .= block(Qs, s)
        end
    end
    return Q
end

# impose full env, could also be defined for half_infinite_environment, little gain
"""Build the CTMRG projectors from the lazy randomized full-environment SVD."""
function compute_projector(fq::FourQuadrants, last_space, alg::RandomizedProjector)
    full_space = fuse(domain(fq))
    randomized_space = random_domain(alg, full_space, last_space)
    Q = randomized_range_finder(fq, alg, randomized_space)
    B = Q' * fq
    normalize!(B)  # TODO better way?

    decomposition_alg = decomposition_algorithm(alg)
    U′, S, V, truncation_error = svd_trunc!(B, decomposition_alg)
    U = Q * U′
    foreach(blocks(S)) do (s, b)
        if size(b, 1) == dim(randomized_space, s) && size(b, 1) < dim(full_space, s)
            @warn("Sector is too small, kept all computed values: ", s)
        end
    end

    # Check for degenerate singular values; still needed for exact blocks
    Zygote.isderiving() && ignore_derivatives() do
        if alg.verbosity > 0 && is_degenerate_spectrum(S)
            svals = TensorKit.SectorDict(c => diag(b) for (c, b) in blocks(S))
            @warn("degenerate singular values detected: ", svals)
        end
    end

    truncation_error = truncation_error / norm(S) # normalize truncation error
    P_left, P_right = contract_projectors(U, S, V, fq)
    return (P_left, P_right), (; U, S, V, truncation_error)
end

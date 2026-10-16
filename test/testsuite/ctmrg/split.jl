using Test, Random, LinearAlgebra
using TensorKit, MatrixAlgebraKit, PEPSKit, Adapt, Zygote
using PEPSKit: ctmrg_iteration, WEST

"""Construct a dense Fishman cut projector independently from fourth roots of the Gram matrices."""
function _split_fishman_reference(Q::AbstractMatrix, Q_next::AbstractMatrix, rank::Int)
    A = normalize(Matrix(Hermitian(Q' * Q)^0.25))
    B = normalize(Matrix(Hermitian(Q_next * Q_next')^0.25))
    U, S, V = svd(normalize(A * B))
    return B * V[:, 1:rank] * Diagonal(1 ./ S[1:rank]) * U[:, 1:rank]' * A
end

function ctmrg_split_projectors(AT)
    Random.seed!(37913)
    return @testset "Split projector stability ($AT)" begin
        alg = SplitCTMRG(; verbosity = 0).projector_alg
        iterative = SplitProjector(; decomposition_alg = (; fwd_alg = (; alg = :GKL, krylovdim = 10)), verbosity = 0)
        trunc = truncspace(ComplexSpace(3))
        # Cover the two-index yellow and three-index red cuts.
        for cut in (ComplexSpace(3) ⊗ ComplexSpace(2), ComplexSpace(3) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2))
            Q = adapt(AT, randn(ComplexF64, ComplexSpace(12) ← cut))
            Q_next = adapt(AT, randn(ComplexF64, cut ← ComplexSpace(12)))
            reference, refinfo = PEPSKit.compute_projector((Q, Q_next), alg; trunc)
            dense = _split_fishman_reference(
                reshape(convert(Array, Q), 12, :), reshape(convert(Array, Q_next), :, 12), 3,
            )
            @test reshape(convert(Array, reference[1] * reference[2]), size(dense)) ≈ dense
            compressed, info = PEPSKit.compute_projector((Q, Q_next), alg; trunc, compress = true)
            # QR compression preserves the cut projector and the nonzero singular spectrum.
            @test compressed[1] * compressed[2] ≈ reference[1] * reference[2]
            @test info.S ≈ refinfo.S
            for compress in (false, true), scale in (1.0e200, 1.0e-200)
                scaled, = PEPSKit.compute_projector((scale * Q, scale * Q_next), alg; trunc, compress)
                @test scaled[1] * scaled[2] ≈ reference[1] * reference[2]
            end
        end

        # Check conditioning, projector pseudoinverse masking, and fixed output spaces with both final solvers.
        V = ComplexSpace(4)
        spectra = (([1.0, 1.0e-3, 1.0e-6, 1.0e-9], 4), ([1.0, 1.0e-6, 1.0e-15, 0.0], 2))
        for (values, retained) in spectra, (solver, rank) in ((alg, 4), (iterative, 1))
            Q = adapt(AT, TensorMap(Matrix(Diagonal(values)), V ← V))
            projectors, info = PEPSKit.compute_projector((Q, Q), solver; trunc = truncspace(ComplexSpace(rank)))
            spectrum = values
            support = [ones(min(retained, rank)); zeros(4 - min(retained, rank))]
            @test Array(info.S.data) ≈ normalize(spectrum)[1:rank]
            @test count(!iszero, info.S.data) == min(retained, rank)
            @test convert(Array, projectors[1] * projectors[2]) ≈ (norm(spectrum) / sum(spectrum)) * Diagonal(support)
            @test all(P -> isfinite(norm(P)), projectors)
        end
    end
end

"""Check that Fishman square roots suppress subthreshold modes in both values and derivatives."""
function ctmrg_split_fishman_cutoff(AT)
    Random.seed!(37914)
    return @testset "Fishman half-spectrum cutoff ($AT)" begin
        V = ComplexSpace(4)
        values = [1.0, 1.0e-6, 1.0e-15, 0.0]
        expected = [1.0, 1.0e-3, 0.0, 0.0]
        for scale in (1.0, 1.0e200, 1.0e-200)
            S = adapt(AT, DiagonalTensorMap(scale * values, V))
            root = PEPSKit._split_fishman_sqrt(S)
            @test Array(root.data) / sqrt(scale) ≈ expected
            @test space(root) == space(S)
        end
        input = adapt(AT, values)
        loss(x) = sum(abs2, PEPSKit._split_fishman_sqrt(DiagonalTensorMap(x, V)).data)
        gradient = only(Zygote.gradient(loss, input))
        @test Array(gradient) ≈ [1.0, 1.0, 0.0, 0.0]
        @test all(isfinite, gradient)

        # Tensor contractions send a dense cotangent into the diagonal square root.
        S = adapt(AT, DiagonalTensorMap(values, V))
        weight = adapt(AT, randn(ComplexF64, V ← V))
        weight += weight'
        gradient = only(Zygote.gradient(X -> real(dot(weight, PEPSKit._split_fishman_sqrt(X))), S))
        expected = real.(diag(convert(Array, weight))) .* [0.5, 500.0, 0.0, 0.0]
        @test gradient ≈ adapt(AT, DiagonalTensorMap(expected, V))
        @test all(isfinite, gradient.data)
    end
end

"""Check split compression and its pullback when unpivoted QR/LQ has internal zero pivots."""
function ctmrg_split_compression(AT)
    Random.seed!(37915)
    return @testset "Rank-deficient split compression ($AT)" begin
        alg = SplitCTMRG(; verbosity = 0).projector_alg
        for T in (Float64, ComplexF64), left in (true, false), deficient in (false, true)
            dense = randn(T, 8, 6)
            if deficient
                dense[:, 2] .= 0
                dense[:, 4] .= 0
            end
            left || (dense = Matrix(dense'))
            A = adapt(AT, TensorMap(dense, ComplexSpace(size(dense, 1)) ← ComplexSpace(size(dense, 2))))
            factor = PEPSKit._split_compress_half(A, alg, Val(left))
            gram(X) = left ? X' * X : X * X'
            @test gram(factor) ≈ gram(A)
            if !deficient
                reference = left ? last(left_orth(A, QRAdjoint())) : first(right_orth(A, LQAdjoint()))
                @test factor == reference
            end
            V = left ? domain(A)[1] : codomain(A)[1]
            H = adapt(AT, randn(T, V ← V))
            H += H'
            loss(X) = real(dot(H, gram(PEPSKit._split_compress_half(X, alg, Val(left)))))
            gradient = only(Zygote.gradient(loss, A))
            exact = left ? 2 * A * H : 2 * H * A
            @test isfinite(norm(gradient))
            @test gradient ≈ exact atol = 1.0e-10 rtol = 1.0e-10
            direction = normalize(adapt(AT, randn(T, space(A))))
            epsilon = 1.0e-5
            finite = (loss(A + epsilon * direction) - loss(A - epsilon * direction)) / (2 * epsilon)
            @test real(dot(gradient, direction)) ≈ finite atol = 1.0e-7 rtol = 1.0e-7
            if deficient
                trunc = truncspace(ComplexSpace(3))
                direct, = PEPSKit.compute_projector((A, A'), alg; trunc)
                compressed, = PEPSKit.compute_projector((A, A'), alg; trunc, compress = true)
                @test compressed[1] * compressed[2] ≈ direct[1] * direct[2]
                W = domain(A)[1]
                weight = adapt(AT, randn(T, W ← W))
                weight += weight'
                cut_loss = X -> begin
                    pair, = PEPSKit.compute_projector((X, X'), alg; trunc, compress = true)
                    return real(dot(weight, pair[1] * pair[2]))
                end
                gradient = only(Zygote.gradient(cut_loss, A))
                finite = (cut_loss(A + epsilon * direction) - cut_loss(A - epsilon * direction)) / (2 * epsilon)
                @test isfinite(norm(gradient))
                @test real(dot(gradient, direction)) ≈ finite atol = 1.0e-6 rtol = 1.0e-5
            end
        end
    end
end

function ctmrg_split_environment(AT)
    return @testset "Split environment ($AT)" begin
        Random.seed!(47119)
        psi = adapt(AT, InfinitePEPS(ComplexSpace(2), ComplexSpace(2); unitcell = (2, 3)))
        env = CTMRGEnv(psi, ComplexSpace(3))
        split = SplitCTMRGEnv(env)
        @test CTMRGEnv(split) ≈ env
        diagonal = CTMRGEnv(DiagonalTensorMap.(real.(env.corners)), copy.(env.edges))
        @test CTMRGEnv(SplitCTMRGEnv(diagonal)) ≈ diagonal
        @test CTMRGEnv(rotl90(split)) ≈ rotl90(env)
        rescaled = deepcopy(split)
        rescaled.edges_ket .*= 2
        rescaled.edges_bra ./= 2
        @test PEPSKit.calc_convergence(rescaled, PEPSKit.convergence_spectra(split, SplitCTMRG())..., SplitCTMRG())[1] < 1.0e-12
    end
end

function ctmrg_split_unitcell(AT)
    Random.seed!(72893)
    return @testset "Split sweeps ($AT)" begin
        unitcell = (2, 3)
        P = ComplexSpace.(rand(2:3, unitcell...))
        N, E = ComplexSpace.(rand(2:3, unitcell...)), ComplexSpace.(rand(2:3, unitcell...))
        chis = ntuple(_ -> ComplexSpace.(rand(3:5, unitcell...)), 4)
        psi = adapt(AT, InfinitePEPS(randn, ComplexF64, P, N, E))
        env = SplitCTMRGEnv(psi, chis...; interlayer_space = ComplexSpace(3))
        alg = SplitCTMRG(; verbosity = 0, projector_alg = :SplitProjector)

        # Non-self-dual charge content catches incorrect conjugations in reflected moves.
        V = U1Space(-1 => 1, 0 => 1, 1 => 1)
        chi = U1Space(-1 => 1, 0 => 2, 1 => 2)
        psi_u1 = adapt(AT, InfinitePEPS(randn, ComplexF64, U1Space(0 => 1, 1 => 1), V))
        env_u1 = SplitCTMRGEnv(psi_u1, chi)
        @test CTMRGEnv(SplitCTMRGEnv(CTMRGEnv(env_u1))) ≈ CTMRGEnv(env_u1)
        for (state, initial) in ((psi, env), (psi_u1, env_u1))
            final, = ctmrg_iteration(InfiniteSquareNetwork(state), initial, alg)
            for field in (:corners, :edges_ket, :edges_bra)
                @test space.(getproperty(final, field)) == space.(getproperty(initial, field))
            end
        end

        # The interlayer and outer refinement parameters can be set independently.
        resized, = ctmrg_iteration(
            InfiniteSquareNetwork(psi), env,
            SplitCTMRG(; verbosity = 0, trunc = truncspace(ComplexSpace(4)), interlayer_trunc = truncspace(ComplexSpace(2))),
        )
        @test all(K -> dim(codomain(K)[1]) == 2, resized.edges_ket[WEST, :, :])
        @test all(K -> dim(domain(K)[1]) == 4, resized.edges_ket[WEST, :, :])
    end
end

function ctmrg_split_expectation_values(AT)
    return @testset "Split expectation values ($T, $AT)" for T in (Float64, ComplexF64)
        Random.seed!(84109)
        psi = adapt(AT, product_peps(randn, T, ComplexSpace(2), ComplexSpace(2); noise_amp = 0.2, state_vector = fill([1.0, 0.0], (1, 1))))
        chi = ComplexSpace(4)
        initial = CTMRGEnv(psi, chi)
        reference, refinfo = leading_boundary(
            initial, psi, SequentialCTMRG(; verbosity = 0, tol = 1.0e-7, maxiter = 40, projector_alg = :FullInfiniteProjector)::SequentialCTMRG,
        )
        split_initial = SplitCTMRGEnv(initial; trunc = truncspace(chi))
        split, info = leading_boundary(split_initial, psi; verbosity = 0, tol = 1.0e-7, maxiter = 40)
        @test refinfo.converged
        @test info.converged
        joined = CTMRGEnv(split)
        @test abs(norm(psi, joined)) ≈ abs(norm(psi, reference)) rtol = 1.0e-4
        H = adapt(AT, heisenberg_XYZ(InfiniteSquare()))
        @test expectation_value(psi, H, joined) ≈ expectation_value(psi, H, reference) atol = 1.0e-4
    end
end

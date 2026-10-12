using Test, Random, LinearAlgebra
using TensorKit, MatrixAlgebraKit, PEPSKit, Adapt
using PEPSKit: ctmrg_iteration, WEST

function ctmrg_split_projectors(AT)
    Random.seed!(37913)
    return @testset "Split projector stability ($AT)" begin
        alg = SplitCTMRG(; verbosity = 0).projector_alg
        trunc = truncspace(ComplexSpace(3))
        # Cover the two-index yellow and three-index red cuts.
        for cut in (ComplexSpace(3) ⊗ ComplexSpace(2), ComplexSpace(3) ⊗ ComplexSpace(2) ⊗ ComplexSpace(2))
            Q = adapt(AT, randn(ComplexF64, ComplexSpace(12) ← cut))
            Q_next = adapt(AT, randn(ComplexF64, cut ← ComplexSpace(12)))
            reference, refinfo = PEPSKit.compute_projector((Q, Q_next), alg; trunc)
            compressed, info = PEPSKit.compute_projector((Q, Q_next), alg; trunc, compress = true)
            # QR compression preserves the cut projector and the nonzero singular spectrum.
            @test compressed[1] * compressed[2] ≈ reference[1] * reference[2]
            @test info.S ≈ refinfo.S
            for compress in (false, true), scale in (1.0e200, 1.0e-200)
                scaled, = PEPSKit.compute_projector((scale * Q, scale * Q_next), alg; trunc, compress)
                @test scaled[1] * scaled[2] ≈ reference[1] * reference[2]
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
